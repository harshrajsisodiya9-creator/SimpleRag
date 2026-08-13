from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq

embedding_model = FastEmbedEmbeddings()
vector_store = Chroma(persist_directory="./db", embedding_function=embedding_model)
retriver = vector_store.as_retriever(search_kwargs={"k": 2})


def format_doc(document):
    return "\n\n".join(doc.page_content for doc in document)


query = "What is RAG"

retrieval = RunnableParallel(docs=retriver, question=RunnablePassthrough())

retrieval_with_context = retrieval | RunnableLambda(
    lambda x: {
        "docs": x["docs"],  # type: ignore
        "context": format_doc(x["docs"]),  # type: ignore
        "question": x["question"],  # type: ignore
    }
)

prompt = ChatPromptTemplate.from_template("""Answer from the proivded context only,
say insufficient information if the data is not reasonable

    context:{context}
    question:{question}""")

llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

generation = retrieval_with_context | RunnableParallel(
    result=prompt | llm, docs=RunnableLambda(lambda x: x["docs"])
)

message = generation.invoke(query)
docs = message["docs"]
for doc in docs:
    print(doc.metadata)
print(message["result"].content)
