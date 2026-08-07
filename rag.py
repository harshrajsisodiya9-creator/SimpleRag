from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/sample.txt")
document = loader.load()

print(document)
