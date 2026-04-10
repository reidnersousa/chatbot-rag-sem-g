import numpy as np
import json
import logging
from sentence_transformers import util, SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class Searcher:
    """
    Busca semântica sobre os embeddings gerados na Fase 1.

    Args:
        embeddings_folder (str): pasta onde estão os arquivos .npy e chunks_metadata.json
        model_name (str): mesmo modelo usado na indexação
        top_k (int): número padrão de resultados retornados
    """

    def __init__(
            self,
            embeddings_folder: str,
            model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
            # model_name: str = "all-MiniLM-L6-v2",
            top_k: int = 5
    ):
        self.top_k = top_k
        self.embeddings_folder = embeddings_folder

        logging.info("Carregando embeddings ...")
        self.doc_embeddings = np.load("embedding/rag_embeddings_policies/doc_embeddings.npy")
        #self.doc_embeddings = np.load(f"{embeddings_folder}/doc_embeddings.npy")

        logging.info("Carregando metadados dos chunks...")
        with open("embedding/rag_embeddings_policies/chunks_metadata.json",encoding="utf-8") as f:
        #with open(f"{embeddings_folder}/chunks_metadata.json", encoding="utf-8") as f:
            self.chunks = json.load(f)

        logging.info("Construindo índice BM25...")
        self.bm25 = BM25Okapi([
            c["text"].lower().split() for c in self.chunks
        ])
        logging.info("BM25 pronto.")
        ### carrega o mapeamento de resposta pré-definidas
        responses_path = "embedding/rag_embeddings_policies/responses.json"
        #responses_path = f"{embeddings_folder}/responses.json"
        logging.info("Carregando base de respostas..")
        with open(responses_path, encoding="utf-8") as f:
            self.response_map = json.load(f)
        logging.info(f"{len(self.response_map)} respostas mapeadas carregadas.")

        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        logging.info("CrossEncoder pronto")

        logging.info(f"Carregando modelo {model_name}...")
        self.model = SentenceTransformer(model_name)
        logging.info("Searcher pronto")

    def _busca_bm25(self, query: str, top_k: int = 20) -> list[dict]:
        """Busca por termos exatos usando BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i],
                             reverse=True)[:top_k]
        return [
            {
                "bi_score": round(float(scores[i]), 4),
                "text": self.chunks[i]["text"],
                "source": self.chunks[i].get("source"),
                "page": self.chunks[i].get("page"),
                "section": self.chunks[i].get("section"),
            }
            for i in top_indices if scores[i] > 0  ## ignora chunks sem  nenhum termo
        ]

    def _rrf(self, *rankings: list[dict], k: int = 60) -> list[dict]:
        """ Funde N  rankings usando Reciprocal Rank Fusion."""
        combined = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking):
                chave = f"{chunk['source']}:: {chunk['section']}"
                if chave not in combined:
                    combined[chave] = {"chunk": chunk, "rrf_score": 0.0}
                combined[chave]['rrf_score'] += 1 / (k + rank + 1)
        sorted_items = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        return [item["chunk"] for item in sorted_items]

    def _sigmoid(self, x: float) -> float:
        """Converte logit bruto do CrossEncoder para probabilidade 0-1"""
        return float(1 / (1 + np.exp(-x)))

    def _confianca_hibrida(self, bi_score: float, rerank_score: float) -> float:
        """
            Combina bi_score e rerank score para uma confiança mais robousta
            Util quando cross-encode é inseguro mas bi -encode acertou
        """
        return round(0.3 * bi_score + 0.7 * rerank_score, 4)

    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> dict:
        """
        Busca os chunks mais similares À  query.

            Args :
                query (str):  pergunta do usuário
                top_k (int):  sobrescreve  o padrão se fornecido

            Returns:
                List[Dict]: lista com score, text, source, page, section
        """
        candidates_k = min(20 if rerank else top_k, len(self.doc_embeddings))
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        top_results = scores.topk(k=candidates_k)

        embedding_candidates = []
        for score, idx in zip(top_results.values, top_results.indices):
            chunk = self.chunks[int(idx)]
            embedding_candidates.append({
                "bi_score": round(float(score), 4),  ## score de embedding
                "text": chunk["text"],
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
            })
        # --- camada 1b: buscar BM25 (termos exatos) ---
        bm25_candidates = self._busca_bm25(query, top_k=candidates_k)

        # --- fusão RRF
        candidates = self._rrf(embedding_candidates, bm25_candidates)

        # --- camada 2: re-ranker ---
        if rerank:
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for candidate, rs in zip(candidates, rerank_scores):
                # aplica sigmoid - agora é comparável e interpretável
                candidate["rerank_score"] = round(self._sigmoid(rs), 4)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # chunk vencedor — o melhor após re-rank
        best = candidates[0]
        # --- camada 3: matcher ---
        resposta = self.match_response(best)

        return {
            "resposta": resposta["texto"],
            "categoria": resposta["categoria"],
            "confianca": self._confianca_hibrida(best["bi_score"], best.get("rerank_score", 0)),
            "fonte": best["source"],
            "chunk_original": best["text"],
            "fallback": resposta["fallback"],
        }

    def match_response(self, chunk: dict) -> dict:
        """
            Busca a resposta pré-definida para o chunk vencedor.
            Chave de busca: source + section.
            Retorna fallback se não houver mapeamento.
        """
        chave = f"{chunk['source']}::{chunk['section']}"

        if chave in self.response_map:
            return {**self.response_map[chave], "fallback": False}

        return {
            "texto": "Não encontrei uma resposta específica para essa pergunta. Por favor, consulte o RH.",
            "categoria": "fallback",
            "fallback": True,
        }

    def search_with_threshold(self, query: str, min_score: float = 0.30, low_confidence: float = 0.10) -> dict:
        """
            Três comportamentos possíveis:
            - confianca >= min_score      → resposta normal
            - low_confidence <= confianca < min_score → resposta com aviso de incerteza
            - confianca < low_confidence  → fallback total
        """
        result = self.search(query)
        confianca = result["confianca"]

        if confianca >= min_score:
            return result

        if confianca >= low_confidence:
            result["resposta"] = (
                f"Encontrei uma informação que pode ser relevante, "
                f"mas não tenho certeza se responde sua pergunta: "
                f"{result['resposta']} — Para mais detalhes, consulte o RH."
            )
            result["fallback"] = True
            return result

        # abaixo de low_confidence — fallback total
        logging.warning(f"Confiança {confianca:.2f} muito baixa para: '{query}'")
        return {
            "resposta": "Não encontrei informações sobre esse tema nas políticas disponíveis. Por favor, consulte o RH diretamente.",
            "categoria": "fallback",
            "confianca": confianca,
            "fonte": None,
            "chunk_original": None,
            "fallback": True,
        }


def teste(searcher: Searcher, queries: list[str], min_score: float = None):
    """
        Roda uma lista de queries e imprime os resultados formatados.
    """
    seperador = "=" * 80
    for query in queries:
        print(seperador)
        print(f"Pergunta: {query}")
        print()

        if min_score:
            results = searcher.search_with_threshold(query, min_score=min_score)
            if result is None:
                print(" [confiança abaixo do threshold - sem resposta]")
                print()
                continue
        else:
            result = searcher.search(query)

        print(f" Resposta  : {result['resposta']}")
        print(f" Categoria : {result['categoria']}")
        print(f" Confiança : {result['confianca']}")
        print(f" Fonte     : {result['fonte']}")
        print(f" Fallback  : {result['fallback']}\n")