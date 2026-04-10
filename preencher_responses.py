import json

path = "embedding/rag_embeddings_policies/responses.json"

with open(path, encoding="utf-8") as f:
    responses = json.load(f)

# ─────────────────────────────────────────
# Preencha aqui as respostas por categoria
# Chave = mesmo formato do responses.json: "nome_arquivo.pdf::numero_secao"
# ─────────────────────────────────────────

respostas_definidas = {

    # HOME OFFICE
    "Polticas de Home Office.pdf::1": {
        "texto": "A política de home office é híbrida: cada colaborador deve estar presencialmente pelo menos 2 dias por semana, salvo exceções aprovadas pelo RH.",
        "categoria": "home_office"
    },
    "Polticas de Home Office.pdf::2": {
        "texto": "A empresa fornece notebook e periféricos para o trabalho remoto. O colaborador deve cuidar da conservação dos equipamentos.",
        "categoria": "home_office"
    },
    "Polticas de Home Office.pdf::3": {
        "texto": "É obrigatório usar VPN e manter a tela bloqueada. Documentos confidenciais não podem ser impressos em casa.",
        "categoria": "home_office"
    },
    "Polticas de Home Office.pdf::4": {
        "texto": "Recomenda-se cadeira adequada e suporte de monitor. O RH pode avaliar pedidos de apoio ergonômico.",
        "categoria": "home_office"
    },
    "Polticas de Home Office.pdf::5": {
        "texto": "Há subsídio mensal de até R$ 100 para internet domiciliar, conforme política de home office.",
        "categoria": "home_office"
    },
    "Polticas de Home Office.pdf::6": {
        "texto": "Para solicitar exceção (como 4-5 dias remotos), é necessário abrir chamado no RH com justificativa.",
        "categoria": "home_office"
    },

    # SEGURANÇA — preencha abaixo
    "Poltica de Uso de E-mail e Segurana da Informao.pdf::1": {
        "texto": "É proibido encaminhar documentos confidenciais para endereços de e-mail pessoais. Em caso de dúvida, consulte a política de segurança da informação.",
        "categoria": "seguranca"
    },
    "Poltica de Uso de E-mail e Segurana da Informao.pdf::2": {
        "texto": "Anexos externos só podem ser enviados se estiverem criptografados e com senha compartilhada por canal separado. Isso garante a proteção dos dados.",
        "categoria": "seguranca"
    },
    "Poltica de Uso de E-mail e Segurana da Informao.pdf::3": {
        "texto": "Para evitar phishing, verifique sempre o remetente e domínios suspeitos. Mensagens suspeitas devem ser reportadas imediatamente ao time de Segurança da Informação.",
        "categoria": "seguranca"
    },
    "Poltica de Uso de E-mail e Segurana da Informao.pdf::4": {
        "texto": "Mensagens que contenham dados pessoais devem seguir as diretrizes de retenção definidas pela empresa. Consulte o manual de retenção para detalhes.",
        "categoria": "seguranca"
    },
    "Poltica de Uso de E-mail e Segurana da Informao.pdf::5": {
        "texto": "Solicitações de liberação de anexos ou domínios devem ser abertas via chamado, com justificativa do gestor. O time de Segurança avaliará cada caso.",
        "categoria": "seguranca"
    },

    # REEMBOLSO — preencha abaixo
    "Poltica de Reembolsos (Viagens e Despesas).pdf::1": {
        "texto": "Para solicitar reembolso é necessário apresentar nota fiscal e enviar o pedido em até 10 dias corridos após a despesa.",
        "categoria": "reembolso"
    },
    "Poltica de Reembolsos (Viagens e Despesas).pdf::2": {
        "texto": "Em viagens, o limite de alimentação é de R$ 70 por dia por pessoa. Bebidas alcoólicas não são reembolsadas.",
        "categoria": "reembolso"
    },
    "Poltica de Reembolsos (Viagens e Despesas).pdf::3": {
        "texto": "Transporte por táxi ou aplicativo é permitido quando não houver alternativa viável. É obrigatório anexar os comprovantes.",
        "categoria": "reembolso"
    },

    "Poltica de Reembolsos (Viagens e Despesas).pdf::4": {
        "texto": "Internet para home office é reembolsada via subsídio mensal de até R$ 100, conforme política específica de home office.",
        "categoria": "reembolso"
    },
    "Poltica de Reembolsos (Viagens e Despesas).pdf::5": {
        "texto": "Cursos e certificações só podem ser reembolsados com aprovação prévia do gestor e orçamento do time.",
        "categoria": "reembolso"
    },
    "Poltica de Reembolsos (Viagens e Despesas).pdf::6": {
        "texto": "Custos excepcionais, como franquia de bagagem extra, devem ser justificados no chamado e aprovados pelo gestor",
        "categoria": "reembolso"
    }
}

# ─────────────────────────────────────────
# Aplica as respostas definidas
# ─────────────────────────────────────────

atualizadas = 0
nao_encontradas = []

for chave, valor in respostas_definidas.items():
    if chave in responses:
        responses[chave] = valor
        atualizadas += 1
    else:
        nao_encontradas.append(chave)

with open(path, "w", encoding="utf-8") as f:
    json.dump(responses, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────
# Relatório
# ─────────────────────────────────────────

pendentes = [k for k, v in responses.items() if "[PREENCHER]" in v["texto"]]

print(f"Atualizadas : {atualizadas}")
print(f"Pendentes   : {len(pendentes)}")

if nao_encontradas:
    print(f"\nChaves não encontradas no responses.json (verifique o nome exato):")
    for c in nao_encontradas:
        print(f"  {c}")

if pendentes:
    print(f"\nAinda faltam preencher:")
    for p in pendentes:
        print(f"  {p}")