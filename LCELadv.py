# previous code of LCEL didnt preserve the document so we couldnt have document as a whole(reason we also wanted doc.meta_data)
# so in here instead of having format_docs we dont format it in the form of string instead we
# directly get the documents and create context from it later using lambda's

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

embedding_model = FastEmbedEmbeddings()

vector_store = Chroma(persist_directory="./db", embedding_function=embedding_model)

retreiver = vector_store.as_retriever(search_kwargs={"k": 2})


def format_docs(document):
    return "\n".join(doc.page_content for doc in document)


prompt = ChatPromptTemplate.from_template("""Answer from the proivded context only, say insufficient information if the data is not reasonable

    context:{context}
    question:{question}""")

retrival = RunnableParallel(docs=retreiver, question=RunnablePassthrough())

# retriever returns , i.e docs = [
#     Document1, Document2, Document3 etc...                  ]
# current dictionary from retrival {docs= object of Documents, question = query} so we should add context too


def add_context(x):
    return {
        "docs": x["docs"],
        "context": format_docs(x["docs"]),
        "question": x["question"],
    }


# x is the dictonary that retrival create which is x = {"docs": [Document1,Document2,...], "question": query}
# to make a chain work in langchain we write .invoke after that chain right but the functions inside it must
# be runnables so one such func is RunnableLambda usually we provide it with a lambda function but it also
# works with normal ones
# variable = lambda argumnets : expression
# square = lambda x : x*x
# print(square(x))
# add = lambda a,b : a+b; print(add(a,b))
# add_context = lambda x : {"docs": x["docs"], "context": format_docs(x["docs"]), "question": x["question"]}

retrival_with_context = retrival | RunnableLambda(add_context)
# output will be now with a new added field context

# another way for above line

another_retrival = retrival | RunnableLambda(
    lambda x: {
        "docs": x["docs"],  # type: ignore
        "context": format_docs(x["docs"]),  # type: ignore
        "question": x["question"],  # type: ignore
    }
)

query = "What is RAG"

result = another_retrival.invoke(query)
print(result)
