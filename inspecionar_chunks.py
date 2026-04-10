

# inspecionar_chunks.py
import json

path = "embedding/rag_embeddings_policies/responses.json"

with open(path, encoding="utf-8") as f:
    responses = json.load(f)

categorias = {
    "home_office": [],
    "seguranca": [],
    "reembolso": [],
    "geral": []
}

for chave, valor in responses.items():
    doc = chave.split("::")[0]
    if "Home Office" in doc:
        categorias["home_office"].append((chave, valor))
    elif "E-mail" in doc or "Segura" in doc:
        categorias["seguranca"].append((chave, valor))
    elif "Reembolso" in doc or "Despesa" in doc:
        categorias["reembolso"].append((chave, valor))
    else:
        categorias["geral"].append((chave, valor))

total_ok = 0
total_pendente = 0

for categoria, itens in categorias.items():
    ok = [i for i in itens if "[PREENCHER]" not in i[1]["texto"]]
    pendentes = [i for i in itens if "[PREENCHER]" in i[1]["texto"]]
    total_ok += len(ok)
    total_pendente += len(pendentes)

    print(f"\n{'='*60}")
    print(f"CATEGORIA: {categoria.upper()} — {len(ok)} OK / {len(pendentes)} pendentes")
    print('='*60)

    for chave, valor in itens:
        status = "OK" if "[PREENCHER]" not in valor["texto"] else "PENDENTE"
        print(f"\n  [{status}] {chave}")
        print(f"  Resposta : {valor['texto'][:120]}{'...' if len(valor['texto']) > 120 else ''}")

print(f"\n{'='*60}")
print(f"RESUMO FINAL: {total_ok} OK — {total_pendente} pendentes")
print('='*60)