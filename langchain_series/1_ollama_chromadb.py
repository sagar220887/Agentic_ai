from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

EMBEDDING_MODEL = 'nomic-embed-text:latest'
OLLAMA_LLM_MODEL = 'deepseek-r1:1.5b'

model = OllamaLLM(
    model=OLLAMA_LLM_MODEL,
    temperature=0.1,
    max_tokens=200,
    stream=True,
    verbose=True
)

template = """
"You are a helpful assistant. You are expert in answering questions about a pizza restaurant.
Answer the question based on the context provided.
Here are some relevant reviews {reviews}
Here is the question to provide the answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# # Define the question and reviews
# question = "What do you think about the pizza crust?"
# reviews = [
#     "I love the pepperoni pizza. It's the best I've ever had!",
#     "The crust was too thick for my liking.",
#     "The service was excellent and the staff were very friendly.",
#     "The cheese was melted to perfection."
# ]

chain = prompt | model

while True:
    print("\n\n --------------------------------\n\n")
    print("Welcome to the Pizza Restaurant Assistant!")
    question = input("Enter your question (or 'q'/exit to quit): ")
    if question.lower() in ['q', 'exit']:
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({
        "reviews":reviews,
        "question": question
    })
    print(result)