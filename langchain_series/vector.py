from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import os
import pandas as pd


df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'realistic_restaurant_reviews.csv'))
df = df.dropna()
print(df.head())

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
db_location = './vectordb/chroma_reviews.db'
add_documents = not os.path.exists(db_location)
if add_documents:
    documents = []
    document_ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=row['Title'] + " " + row['Review'], 
            metadata={"rating": row['Rating'], "date": row['Date']},
            id = str(i)
        )
        document_ids.append(str(i))
        documents.append(document)


vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=db_location,
    collection_name="restaurant_reviews"
)


if add_documents:
    vectorstore.add_documents(documents=documents, ids=document_ids)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


if __name__ == "__main__":
    while True:
        print("\n\n --------------------------------\n\n")
        print("Welcome to the Vector DB of Restaurant Reviews Assistant!")
        question = input("Enter your question (or 'q'/exit to quit): ")
        if question.lower() in ['q', 'exit']:
            break

        reviews = retriever.invoke(question)
        result = reviews
        print(result)