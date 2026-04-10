from fase2 import Searcher, teste




searcher = Searcher(
    embeddings_folder="embedding/rag_embeddings_policies"
)

queries_home_office = [
    "Posso trabalhar 4 dias por semana de casa sem ir ao escritório?",
    "A empresa oferece algum subsídio para internet no home office?",
    "Quantos dias preciso ir presencialmente na empresa?",
]

queries_seguranca_email =[
"Posso enviar arquivos da empresa para meu e-mail pessoal?",
"O que devo fazer se receber um e-mail suspeito?",
"Como enviar arquivos confidenciais para fora da empresa?"
]

queries_reembolso = [
"Qual o prazo para pedir reembolso de despesas?",
"Existe limite diário para alimentação em viagens?",
"Posso reembolsar bebidas alcoólicas?"
]
# sem threshold — retorna top 5
print("+"*30+"Query home_office"+"+"*30)
teste(searcher, queries_home_office)
print("+"*30+"seguranca_email"+"+"*30)
teste(searcher, queries_seguranca_email)
print("+"*30+"Query reembolso"+"+"*30)
teste(searcher, queries_reembolso)
print("+"*30+"Query difícil"+"+"*30)
queries_hard= ["Existe algum apoio financeiro para quem trabalha fora do escritório?"]

teste(searcher,queries_hard)

#print("Com threshold 0.45")
# com threshold — só retorna se for relevante de verdade
#teste(searcher, queries_home_office, min_score=0.45)