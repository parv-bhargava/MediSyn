## user_query ----> LLM ----> response
from src.agent import LLM
user_query = input("Enter your query: ")
model = LLM()
response = model.get_response("meta.llama3-70b-instruct-v1:0", {"input": user_query})
print(response)


## user_query ----> RAG ----> LLM ----> response
from src.rag import RAG
user_query = input("Enter your query: ")
model = LLM()
rag = RAG()
aug = rag.get_augmentation(user_query)
response = model.get_response("meta.llama3-70b-instruct-v1:0", {"input": user_query + aug})



## user_query ----> MultiAgent ----> response