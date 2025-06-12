import os
import atexit
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter

load_dotenv()
AZURE_AI_SEARCH_SERVICE_NAME = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
AZURE_AI_SEARCH_INDEX_NAME = "aula-index"
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_EMBEDDINGS_MODEL = "text-embedding-3-small"
AZURE_OPENAI_MODEL = "gpt-4o"


embeddings = AzureOpenAIEmbeddings(
    model= AZURE_EMBEDDINGS_MODEL,
    azure_endpoint= AZURE_OPENAI_ENDPOINT,
    openai_api_key= AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

vector_store: AzureSearch = AzureSearch(
    embedding_function=embeddings.embed_query,
    azure_search_endpoint= AZURE_AI_SEARCH_SERVICE_NAME,
    azure_search_key= AZURE_AI_SEARCH_API_KEY,
    index_name= AZURE_AI_SEARCH_INDEX_NAME    
)

def load_documents_from_directory(directory_path: str):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Diretório '{directory_path}' não encontrado. Caminho atual: {os.getcwd()}")

    loader = DirectoryLoader(directory_path, glob="*.docx", show_progress=True)
    docs = loader.load()
    vector_store.add_documents(documents=docs)
    print("Documentos processados e adicionados ao vetor.")


def chat_on_files():
    azure_endpoint=AZURE_OPENAI_ENDPOINT
    api_key=AZURE_OPENAI_API_KEY
    openai_api_version="2024-12-01-preview"

    llm = AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_MODEL, 
        model=AZURE_OPENAI_MODEL,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

    
    qa = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vector_store.as_retriever(),
      return_source_documents=True
    )
    query = f""" Estou com uma vaga de diretor de Tecnologia no IPT quem é o melhor candidado e porque ?"""
    response = qa({"query": query})
    print(f"Resposta: {response['result']}")
    print("Documentos de origem:")
    for doc in response['source_documents']:
        print(f" - {doc.metadata['source']}")

#load_documents_from_directory("livro")
chat_on_files()