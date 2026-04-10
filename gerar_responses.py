import json


with open("embedding/rag_embeddings_policies/chunks_metadata.json",encoding="utf-8") as f:
#with open("/kaggle/working/embedding/rag_embeddings_policies/chunks_metadata.json", encoding="utf-8") as f:
    chunks = json.load(f)

response_map = {}
for chunk in chunks:
    chave = f"{chunk['source']}::{chunk['section']}"
    if chave not in response_map:  # evita duplicatas
        response_map[chave] = {
            "texto": f"[PREENCHER] {chunk['text'][:100]}",
            "categoria": "geral"
        }

with open("embedding/rag_embeddings_policies/responses.json", "w", encoding="utf-8") as f:
    json.dump(response_map, f, ensure_ascii=False, indent=2)

print(f"{len(response_map)} chaves geradas.")