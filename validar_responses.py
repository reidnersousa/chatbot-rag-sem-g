

import json
from fase2 import Searcher

searcher = Searcher("embedding/rag_embeddings_policies")

queries = {
    "home_office": [
        "Posso trabalhar 4 dias por semana de casa sem ir ao escritório?",
        "A empresa oferece algum subsídio para internet no home office?",
        "Quantos dias preciso ir presencialmente na empresa?",
        "Existe algum apoio financeiro para quem trabalha fora do escritório?",
    ],
    "seguranca": [
        "Posso enviar arquivos da empresa para meu e-mail pessoal?",
        "O que devo fazer se receber um e-mail suspeito?",
        "Como enviar arquivos confidenciais para fora da empresa?",
    ],
    "reembolso": [
        "Qual o prazo para pedir reembolso de despesas?",
        "Existe limite diário para alimentação em viagens?",
        "Posso reembolsar bebidas alcoólicas?",
        "Posso reembolsar um curso sem avisar o gestor antes?",
    ],
    "fora_do_dominio": [
        "Qual é a política de férias da empresa?",
        "Como solicitar aumento de salário?",
        "A empresa tem plano de saúde?",
        "Posso usar o notebook da empresa para uso pessoal?",
    ],
}

resultados = {"ok": 0, "fallback_correto": 0, "fallback_incorreto": 0}

for categoria, perguntas in queries.items():
    print(f"\n{'+'*30} {categoria.upper()} {'+'*30}")
    for query in perguntas:
        result = searcher.search_with_threshold(query, min_score=0.30, low_confidence=0.10)
        #result = searcher.search(query)
        is_fora = categoria == "fora_do_dominio"

        if result["fallback"]:
            status = "FALLBACK OK" if is_fora else "FALLBACK INCORRETO"
            if is_fora:
                resultados["fallback_correto"] += 1
            else:
                resultados["fallback_incorreto"] += 1
        else:
            status = "FORA DO DOMINIO SEM FALLBACK" if is_fora else "OK"
            if not is_fora:
                resultados["ok"] += 1
            else:
                resultados["fallback_incorreto"] += 1

        print(f"\n  [{status}]")
        print(f"  Pergunta  : {query}")
        print(f"  Resposta  : {result['resposta']}")
        print(f"  Confiança : {result['confianca']} | Categoria: {result['categoria']} | Fonte: {result['fonte']}")

print(f"\n{'='*60}")
print(f"RESUMO")
print(f"{'='*60}")
print(f"  Respostas corretas          : {resultados['ok']}")
print(f"  Fallback correto (fora dom) : {resultados['fallback_correto']}")
print(f"  Problemas                   : {resultados['fallback_incorreto']}")