
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
AZURE_EMBEDDINGS_MODEL = "text-embedding-3-large"
AZURE_OPENAI_MODEL = "gpt-4o-mini"

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

def load_documents():
    folder_path = r"F:\APP-GenAI\Livro"
    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                all_docs.extend(docs)
                
                print(f"✔️ Documento carregado: {file}")
            except Exception as e:
                print(f"❌ Falha ao carregar {file}: {e}")

        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(all_docs)
        vector_store.add_documents(split_docs)
        
    print("📥 Todos os documentos válidos foram adicionados ao vetor.")

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
        deployment_name=AZURE_OPENAI_MODEL,  # isso é "gpt-4o-mini"
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

    query = f"""
        Tenho uma vaga de Analista de RH você indicaria para essa vaga e porque não sabe caso não saiba ?"""

    result = qa({"query": query})
  
    print("Resposta: ", result["result"])


#load_documents()

chat_on_files()

#load_documents_from_directory("Livro")