import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import tempfile
import os
import json
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
# Configurar a página do Streamlit
st.set_page_config(page_title="Chat com Documentos", page_icon="💬", layout="centered")

#variáveis de ambiente
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDING")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

st.markdown("<h2 style='text-align: center;'>Use os atalhos abaixo</h2>", unsafe_allow_html=True)

# col1 = st.columns(1)
# with col1:
#   if st.button("Buscar Personalidades"):
#       st.session_state.user_input = "Quais são as personalidades do ChatGPT?"

uploaded_file = st.file_uploader("📄 Faça upload do documento (PDF, TXT ou JSON)", type=["pdf", "txt", "json"])

if uploaded_file:
    try:
        with st.spinner("Processando documento..."):
            # Salvar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            # Carregar documento baseado no tipo
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif uploaded_file.name.endswith(".json"):
                with open(tmp_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                docs = [Document(page_content=json.dumps(json_data, indent=2))]
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()

            # Dividir documentos em chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            # Verificar se há chunks para processar
            if not chunks:
                st.error("❌ Nenhum conteúdo foi extraído do documento.")
                st.stop()
            
            try:
                embeddings = AzureOpenAIEmbeddings(
                    model=AZURE_OPENAI_DEPLOYMENT_EMBEDDING,
                    api_key=AZURE_OPENAI_API_KEY,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_version=AZURE_OPENAI_VERSION
                )
                
                test_embedding = embeddings.embed_query("teste")
            except Exception as e:
                st.error(f"❌ Erro ao configurar embeddings: {str(e)}")
                st.info("Verifique suas credenciais do Azure OpenAI e se o deployment de embedding está correto.")
                st.stop()

            try:
                vectorstore = Chroma.from_documents(chunks, embeddings)
            
            except Exception as e:
                st.error(f"❌ Erro ao criar base vetorial: {str(e)}")
                st.info("Possíveis causas: credenciais inválidas, quota excedida, ou problema de conectividade.")
                st.stop()
            # Criar agente RAG
            try:
                retriever = vectorstore.as_retriever()
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=AzureChatOpenAI(
                        api_key=AZURE_OPENAI_API_KEY,
                        azure_endpoint=AZURE_OPENAI_ENDPOINT,
                        api_version=AZURE_OPENAI_VERSION,
                        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                        temperature=0.7
                    ),
                    retriever=retriever,
                    return_source_documents=False
                )

                st.session_state.qa_chain = qa_chain
                st.success("✅ Documento processado e indexado com sucesso!")
            except Exception as e:
                st.error(f"❌ Erro ao criar cadeia de conversação: {str(e)}")
                st.stop()
            # Limpar arquivo temporário
            os.unlink(tmp_path)
    except Exception as e:
        st.error(f"❌ Erro ao processar documento: {str(e)}")
        st.info("Verifique se o arquivo está no formato correto e não está corrompido.")

# === HISTÓRICO ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

# === INPUT DO USUÁRIO ===
user_input = st.text_area("Digite sua pergunta ou prompt:", key="user_input", height=200, label_visibility="collapsed")

if st.button("Enviar") and user_input.strip():
    if "qa_chain" in st.session_state:
        try:
            with st.spinner("Processando pergunta..."):
                result = st.session_state.qa_chain({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })
                resposta = result["answer"]
        except Exception as e:
            resposta = f"❌ Erro ao processar pergunta: {str(e)}"
    else:
        resposta = "⚠️ Nenhum documento carregado ainda. Faça upload para ativar o chat com contexto."

    st.session_state.chat_history.append(("Você", user_input))
    st.session_state.chat_history.append(("Assistente", resposta))

# === CONVERSA ===
if st.session_state.chat_history:
    st.markdown("### 💬 Conversa")
    for remetente, mensagem in st.session_state.chat_history:
        with st.chat_message("user" if remetente == "Você" else "assistant"):
            st.markdown(mensagem)

# === BOTÃO PARA LIMPAR HISTÓRICO ===
if st.session_state.chat_history:
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.chat_history = []
        st.rerun()

# === Rodapé ===
st.markdown("<div style='text-align: right; color: gray;'>Aula do Henrique</div>", unsafe_allow_html=True)