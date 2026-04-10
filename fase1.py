import os
import re
import json
import logging
import unicodedata
import fitz


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def clean_text(texto: str) -> str:
    texto = unicodedata.normalize("NFKC",texto)
    texto = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]","",texto)
    texto = re.sub(r"\s+"," ",texto)
    return texto.strip()


def get_politica(texto: str) -> str | None:
    """
    Extrai o tipo de política mencionada no texto.

    Returns:
        str com o tipo da política, ou None se não encontrar.
    """
    padrao = r"(?i)política\s+(?:de|da|do)?\s*([\wÀ-ú]+)"
    resultado = re.search(padrao, texto)
    return resultado.group(1) if resultado else None


def get_section(texto: str) -> list[tuple[str, str]]:
    """
    Divide o texto em seções numeradas (ex: '1. Título...').

    Returns:
        Lista de tuplas (número_seção, conteúdo).
        Retorna [(\"0\", texto_completo)] se nenhuma seção for encontrada — fallback
        para não perder conteúdo de páginas sem numeração.
    """
    padrao = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"
    resultados = re.findall(padrao, texto, flags=re.DOTALL)
    resultados = [(num, conteudo.strip()) for num, conteudo in resultados]

    if not resultados:
        # Fallback: trata a página inteira como um chunk único (seção "0")
        texto_limpo = texto.strip()
        if texto_limpo:
            return [("0", texto_limpo)]
        return []

    return resultados



def load_pdfs(pdf_folder: str) -> list[dict]:
    """
    Carrega todos os PDFs da pasta, extrai texto, divide em seções e aplica limpeza.

    Args:
        pdf_folder: caminho para a pasta contendo os PDFs.

    Returns:
        Lista de chunks, cada um com source, page, section, politica e text.
    """
    chunks = []
    documentos = os.listdir(pdf_folder)

    for documento in documentos:
        # Ignora arquivos que não são PDF
        if not documento.lower().endswith(".pdf"):
            logging.warning(f"Ignorando arquivo não-PDF: {documento}")
            continue

        path = os.path.join(pdf_folder, documento)

        try:
            doc = fitz.open(path)
        except Exception as e:
            logging.error(f"Não foi possível abrir {documento}: {e}")
            continue

        logging.info(f"Processando: {documento} ({doc.page_count} páginas)")

        for page_num in range(doc.page_count):
            texto = doc[page_num].get_text()

            if not texto.strip():
                logging.debug(f"  Página {page_num} vazia, pulando.")
                continue

            politica = get_politica(texto)
            secoes = get_section(texto)

            for num, conteudo in secoes:
                texto_limpo = clean_text(conteudo)

                # Ignora chunks muito curtos — provavelmente ruído
                if len(texto_limpo) < 20:
                    continue

                chunks.append({
                    "source": documento,
                    "page": page_num,
                    "section": num,
                    "politica": politica,
                    "text": texto_limpo,
                })

        doc.close()

    logging.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks


def generate_embeddings(
    chunks: list[dict],
    model_name = "paraphrase-multilingual-MiniLM-L12-v2",
    #model_name: str = "all-MiniLM-L6-v2",
    save_folder: str = "/kaggle/working/embedding/rag_embeddings_policies"
) -> tuple:
    """
    Gera embeddings para os chunks e salva em disco.

    Salva:
        - doc_embeddings.npy  → matriz de embeddings
        - chunks_metadata.json → lista completa de dicts com text + metadados

    Returns:
        tuple: (texts, embeddings)
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    if not chunks:
        raise ValueError("Chunks estão vazios. Rode load_pdfs primeiro.")

    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    os.makedirs(save_folder, exist_ok=True)

    # Salva embeddings
    np.save(os.path.join(save_folder, "doc_embeddings.npy"), embeddings)

    # Salva metadados completos (substitui o texts.npy que perdia source/page/section)
    metadata_path = os.path.join(save_folder, "chunks_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logging.info(f"Embeddings e metadados salvos em {save_folder}/")
    return texts, embeddings


# --- Execução direta no Kaggle ---
if __name__ == "__main__":
    root = "pdf"
    chunks = load_pdfs(root)
    texts, embeddings = generate_embeddings(chunks)