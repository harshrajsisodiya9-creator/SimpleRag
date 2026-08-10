from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/sample.txt")
document = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = splitter.split_documents(documents=document)

# chunks = [
#       Document1,
#       Document2,
#       ....
# #]
# Each document will consist of metadata and page_content

# print(f"Number of chunks: {len(chunks)}")
# print(f"Number of chunks: {len(chunks)}")

# # for i, chunk in enumerate(chunks):
# #     print(f"\nCHUNK {i + 1}")
# #     print(repr(chunk.page_content))

embedding_model = FastEmbedEmbeddings()

vector = embedding_model.embed_query(chunks[0].page_content)

# vector_store is the reference to the db created by Chroma
ids = [f"chunk-{i}" for i in range(len(chunks))]

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    ids=ids,
    persist_directory="db",  # db directory
)

# internal working of above line
# chunk_texts = [chunk.page_content for chunk in chunks]
# vector = embedding_model.embed_documents(chunk_texts)
# zip combines two things which can be accessed in for loops
# for chunk,embedding in zip(chunks,vector):
#       chroma.insert(
#       embedding = embedding,
#       document = chunk.page_content
#       metadata = chunk.metadata
#       )
# #

quer = "What is LangChain?"


results = vector_store.similarity_search(query=quer, k=2)

# print(len(results))

# for i, doc in enumerate(results):
#     print(f"\n{i + 1}")
#     print(repr(doc.page_content))

context = "\n\n".join(doc.page_content for doc in results)

prompt = f"""
    Answer the questions using:

    Context: {context}

    Query: {quer}

    Answer:
"""

llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

repsonse = llm.invoke(prompt)
print(repsonse.content)
