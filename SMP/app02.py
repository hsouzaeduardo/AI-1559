# Importa a biblioteca Streamlit para criar interfaces web interativas em Python
import streamlit as st
# Importa a biblioteca os para manipulação de variáveis de ambiente e arquivos
import os
# Importa a biblioteca tempfile para criar arquivos temporários
import tempfile
# Importa a biblioteca time para usar funções relacionadas ao tempo (como sleep)
import time
# Importa a biblioteca gc para gerenciamento de memória (garbage collection)
import gc
# Importa BytesIO para manipular arquivos em memória (sem salvar no disco)
from io import BytesIO
# Importa o carregador de PDF do LangChain
from langchain_community.document_loaders import PyPDFLoader
# Importa o AzureSearch do LangChain para busca vetorial no Azure
from langchain_community.vectorstores import AzureSearch
# Importa classes para embeddings e LLM do Azure OpenAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# Importa o chain de Pergunta e Resposta do LangChain
from langchain.chains import RetrievalQA
# Importa o divisor de texto por tokens
from langchain_text_splitters import TokenTextSplitter
# Outro carregador de PDF do LangChain
from langchain_community.document_loaders import UnstructuredPDFLoader
# Importa a classe Document do LangChain
from langchain.schema import Document
# Importa função para carregar variáveis de ambiente de um arquivo .env
from dotenv import load_dotenv
# Importa biblioteca para ler PDFs
import PyPDF2

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do Azure (pega as variáveis do ambiente)
AZURE_AI_SEARCH_SERVICE_NAME = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")
AZURE_AI_SEARCH_INDEX_NAME = "reactor-index"  # Nome fixo do índice
AZURE_AI_SEARCH_API_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
AZURE_EMBEDDINGS_MODEL = "text-embedding-3-small"
AZURE_OPENAI_MODEL = "gpt-4o-mini"

def initialize_azure_services():
    """Inicializa os serviços do Azure"""
    try:
        # Cria o objeto de embeddings (transforma texto em vetores)
        embeddings = AzureOpenAIEmbeddings(
            model=AZURE_EMBEDDINGS_MODEL,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=AZURE_OPENAI_API_VERSION,
        )

        # Cria o objeto de busca vetorial no Azure
        vector_store = AzureSearch(
            embedding_function=embeddings.embed_query,
            azure_search_endpoint=AZURE_AI_SEARCH_SERVICE_NAME,
            azure_search_key=AZURE_AI_SEARCH_API_KEY,
            index_name=AZURE_AI_SEARCH_INDEX_NAME,
            # Configurações opcionais
            fields=None,
            vector_search_dimensions=None,
            vector_search_profile_name=None,
            semantic_configuration_name=None,
        )

        # Cria o modelo de linguagem (LLM) do Azure OpenAI
        llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_MODEL,
            model=AZURE_OPENAI_MODEL,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.1,  # Controla a criatividade das respostas
        )
        
        return embeddings, vector_store, llm
    except Exception as e:
        st.error(f"Erro ao inicializar serviços Azure: {str(e)}")
        return None, None, None

def process_uploaded_file(uploaded_file, vector_store):
    """Processa arquivo PDF carregado usando PyPDF2 para evitar arquivos temporários"""
    try:
        # Lê o PDF diretamente da memória usando PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
        
        # Extrai o texto de todas as páginas do PDF
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                full_text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
            except Exception as page_error:
                st.warning(f"Erro na página {page_num + 1}: {page_error}")
                continue
        
        if not full_text.strip():
            st.error("Não foi possível extrair texto do PDF.")
            return 0, ""
        
        # Cria um documento do LangChain com o texto extraído
        doc = Document(
            page_content=full_text,
            metadata={
                "source": uploaded_file.name,
                "pages": len(pdf_reader.pages)
            }
        )
        
        # Divide o texto em pedaços menores (chunks) para facilitar a busca
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents([doc])
        
        # Adiciona os pedaços ao vector store (banco de vetores)
        vector_store.add_documents(split_docs)
        
        return len(split_docs), full_text[:500]
        
    except Exception as e:
        # Se falhar, tenta outro método usando arquivo temporário
        st.warning(f"Método PyPDF2 falhou: {e}. Tentando método alternativo...")
        return process_uploaded_file_fallback(uploaded_file, vector_store)

def process_uploaded_file_fallback(uploaded_file, vector_store):
    """Método alternativo usando arquivo temporário"""
    tmp_file_path = None
    try:
        # Cria um arquivo temporário para salvar o PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            tmp_file_path = tmp_file.name
        
        # Espera um pouco para garantir que o arquivo foi salvo
        time.sleep(0.1)
        
        # Carrega o PDF usando PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        
        # Divide o texto em pedaços menores
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        
        # Adiciona ao vector store
        vector_store.add_documents(split_docs)
        
        return len(split_docs), docs[0].page_content[:500] if docs else ""
        
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return 0, ""
    finally:
        # Tenta apagar o arquivo temporário criado
        if tmp_file_path and os.path.exists(tmp_file_path):
            for attempt in range(5):  # Tenta até 5 vezes
                try:
                    time.sleep(0.1 * (attempt + 1))  # Espera um pouco mais a cada tentativa
                    os.unlink(tmp_file_path)
                    break
                except (PermissionError, OSError) as cleanup_error:
                    if attempt == 4:
                        st.warning(f"Arquivo temporário será limpo pelo sistema: {cleanup_error}")
                    continue
        # Força o garbage collector a liberar memória
        gc.collect()

def custom_search_and_answer(question, vector_store, llm, max_docs=3):
    """Busca documentos relevantes e gera resposta usando o LLM"""
    try:
        # Busca os documentos mais parecidos com a pergunta
        docs = vector_store.similarity_search(question, k=max_docs)
        
        if not docs:
            return "Não foram encontrados documentos relevantes para responder sua pergunta.", []
        
        # Junta o conteúdo dos documentos encontrados
        context = "\n\n".join([f"Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Cria o prompt para o modelo de linguagem
        prompt = f"""
Com base no contexto fornecido, responda à pergunta de forma clara e precisa.

Contexto:
{context}

Pergunta: {question}

Resposta:
"""
        
        # Chama o modelo de linguagem para gerar a resposta
        response = llm.invoke(prompt)
        
        # Extrai o texto da resposta
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return answer, docs
        
    except Exception as e:
        st.error(f"Erro na busca personalizada: {e}")
        return f"Erro ao processar a pergunta: {str(e)}", []

def create_qa_chain(llm, vector_store):
    """Cria um chain de Pergunta e Resposta usando LangChain"""
    try:
        # Configura o retriever para buscar por similaridade
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Erro ao criar QA chain: {e}")
        # Fallback sem parâmetros de busca específicos
        retriever = vector_store.as_retriever()
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

def main():
    # Configura a página do Streamlit
    st.set_page_config(
        page_title="RAG com Azure OpenAI",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🤖 Sistema de Consulta de Documentos")
    st.markdown("---")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Verifica se as variáveis de ambiente estão configuradas
        env_status = {
            "AZURE_AI_SEARCH_SERVICE_NAME": bool(AZURE_AI_SEARCH_SERVICE_NAME),
            "AZURE_AI_SEARCH_API_KEY": bool(AZURE_AI_SEARCH_API_KEY),
            "AZURE_OPENAI_ENDPOINT": bool(AZURE_OPENAI_ENDPOINT),
            "AZURE_OPENAI_API_KEY": bool(AZURE_OPENAI_API_KEY)
        }
        
        st.subheader("Status das Configurações:")
        for var, status in env_status.items():
            if status:
                st.success(f"✅ {var}")
            else:
                st.error(f"❌ {var}")
        
        if not all(env_status.values()):
            st.warning("⚠️ Configure todas as variáveis de ambiente no arquivo .env")
            return
    
    # Inicializa os serviços Azure apenas uma vez
    if 'services_initialized' not in st.session_state:
        with st.spinner("Inicializando serviços Azure..."):
            embeddings, vector_store, llm = initialize_azure_services()
            if embeddings and vector_store and llm:
                st.session_state.embeddings = embeddings
                st.session_state.vector_store = vector_store
                st.session_state.llm = llm
                st.session_state.services_initialized = True
                st.success("✅ Serviços Azure inicializados com sucesso!")
            else:
                st.error("❌ Falha ao inicializar serviços Azure")
                return
    
    # Divide a tela em duas colunas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload de Documentos")
        
        # Permite o upload de múltiplos arquivos PDF
        uploaded_files = st.file_uploader(
            "Faça upload dos seus documentos PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Limite: 200MB por arquivo • Formatos: PDF"
        )
        
        if uploaded_files:
            if st.button("🔄 Processar Documentos", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chunks = 0
                processed_files = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processando: {uploaded_file.name}")
                    
                    chunks, preview = process_uploaded_file(
                        uploaded_file, 
                        st.session_state.vector_store
                    )
                    
                    if chunks > 0:
                        total_chunks += chunks
                        processed_files.append({
                            'name': uploaded_file.name,
                            'chunks': chunks,
                            'preview': preview
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Processamento concluído!")
                
                if processed_files:
                    st.success(f"📥 {len(processed_files)} arquivos processados com {total_chunks} chunks!")
                    
                    with st.expander("Ver detalhes dos arquivos processados"):
                        for file_info in processed_files:
                            st.write(f"**{file_info['name']}** - {file_info['chunks']} chunks")
                            if file_info['preview']:
                                st.text_area(
                                    f"Preview de {file_info['name']}:",
                                    file_info['preview'],
                                    height=100,
                                    key=f"preview_{file_info['name']}"
                                )
    
    with col2:
        st.header("💬 Consultas aos Documentos")
        
        # Perguntas pré-definidas para facilitar o uso
        st.subheader("🎯 Perguntas Frequentes")
        
        predefined_questions = [
            "Em qual empresa Lucas trabalhou antes da Titantech?",
            "Tenho uma vaga de Analista de RH, você indicaria para essa vaga e por quê?",
            "Quais são as principais qualificações mencionadas no documento?",
            "Faça um resumo das experiências profissionais mencionadas."
        ]
        
        selected_question = st.selectbox(
            "Escolha uma pergunta:",
            [""] + predefined_questions,
            index=0
        )
        
        # Campo para pergunta personalizada
        st.subheader("✍️ Pergunta Personalizada")
        custom_question = st.text_area(
            "Digite sua pergunta:",
            value=selected_question if selected_question else "",
            height=100,
            placeholder="Ex: Quais são as principais habilidades de Lucas?"
        )
        
        if st.button("🔍 Fazer Pergunta", type="primary"):
            if not custom_question.strip():
                st.warning("⚠️ Por favor, digite uma pergunta.")
            elif 'services_initialized' not in st.session_state:
                st.error("❌ Serviços não inicializados.")
            else:
                with st.spinner("🤔 Buscando resposta..."):
                    try:
                        # Usa o método personalizado para buscar resposta
                        answer, source_docs = custom_search_and_answer(
                            custom_question,
                            st.session_state.vector_store,
                            st.session_state.llm,
                            max_docs=3
                        )
                        
                        st.subheader("📋 Resposta:")
                        st.write(answer)
                        
                        # Mostra os documentos fonte usados na resposta
                        if source_docs:
                            with st.expander(f"📚 Fontes ({len(source_docs)} documentos)"):
                                for i, doc in enumerate(source_docs):
                                    st.write(f"**Fonte {i+1}:**")
                                    st.text_area(
                                        f"Conteúdo {i+1}:",
                                        doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                        height=150,
                                        key=f"source_{i}_{time.time()}"  # Key único para evitar conflitos
                                    )
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        st.json(doc.metadata)
                                    st.markdown("---")
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao buscar resposta: {str(e)}")
                        
                        # Se falhar, tenta método alternativo usando LangChain QA Chain
                        try:
                            st.info("🔄 Tentando método alternativo...")
                            qa_chain = create_qa_chain(
                                st.session_state.llm,
                                st.session_state.vector_store
                            )
                            
                            result = qa_chain({"query": custom_question})
                            
                            st.subheader("📋 Resposta (Método Alternativo):")
                            st.write(result["result"])
                            
                            # Mostra os documentos fonte
                            if result.get("source_documents"):
                                with st.expander(f"📚 Fontes ({len(result['source_documents'])} documentos)"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        st.write(f"**Fonte {i+1}:**")
                                        st.text_area(
                                            f"Conteúdo {i+1}:",
                                            doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                            height=150,
                                            key=f"alt_source_{i}_{time.time()}"
                                        )
                                        if hasattr(doc, 'metadata') and doc.metadata:
                                            st.json(doc.metadata)
                                        st.markdown("---")
                            
                        except Exception as alt_error:
                            st.error(f"❌ Ambos os métodos falharam: {str(alt_error)}")
                            st.info("💡 Tente reformular sua pergunta ou verifique se os documentos foram processados corretamente.")
    
    # Rodapé da página
    st.markdown("---")
    st.markdown("🔧 **Sistema RAG** com Azure OpenAI, LangChain e Streamlit")

if __name__ == "__main__":
    main()