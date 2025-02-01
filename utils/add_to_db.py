from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import os

print('Loading Documents....')
file_path = 'c++.pdf'
loader=PyPDFLoader(file_path)
docs=loader.load()
print("Splitting the docs....")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2')


vector_store = Chroma.from_documents(
    chunks,
    collection_name="document_store",
    embedding=embedder,
    persist_directory="chromadb_store",
)


vector_store=Chroma(persist_directory="chromadb_store",embedding_function=embedder,collection_name="document_store")
results=vector_store.similarity_search(query="Explain c++ pointers",k=3)
print(results)