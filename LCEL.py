from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

embedding_model = FastEmbedEmbeddings()

vector_store = Chroma(persist_directory="./db", embedding_function=embedding_model)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template("""
    Answer from the proivded context only, say insufficient information if the data is not reasonable

    context:{context}
    question:{question}
""")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

chain = (
    RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
    | prompt
    | llm
)

query = "What is RAG"

response = chain.invoke(query)

print(response.content)
