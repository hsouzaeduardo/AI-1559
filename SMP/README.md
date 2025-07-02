# Sistema RAG com Azure OpenAI, LangChain e Streamlit

Este projeto é um sistema de consulta de documentos PDF utilizando RAG (Retrieval-Augmented Generation) com Azure OpenAI, LangChain e Streamlit.

## Pré-requisitos

- Python 3.9 ou superior
- Git
- Conta no Azure com acesso ao Azure OpenAI e Azure Cognitive Search

## Passo a passo para rodar o projeto

### 1. Clonar o repositório

Abra o terminal/powershell e execute:

```sh
git clone <URL_DO_REPOSITORIO>
cd AI-1559/SMP
```

Substitua `<URL_DO_REPOSITORIO>` pela URL do seu repositório.

### 2. Criar e ativar um ambiente virtual (opcional, mas recomendado)

No Windows PowerShell:

```sh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux/Mac:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

Com o ambiente virtual ativado, execute:

```sh
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

Crie um arquivo `.env` na pasta `SMP` com as seguintes variáveis (preencha com seus dados do Azure):

```
AZURE_AI_SEARCH_SERVICE_NAME=<nome-do-servico-search>
AZURE_AI_SEARCH_API_KEY=<sua-api-key-search>
AZURE_OPENAI_ENDPOINT=<endpoint-openai>
AZURE_OPENAI_API_KEY=<sua-api-key-openai>
```

### 5. Executar o sistema

No terminal, ainda na pasta `SMP`, rode:

```sh
streamlit run app02.py
```

O Streamlit abrirá uma página web no navegador. Siga as instruções na tela para fazer upload de PDFs e realizar consultas.

---

## Observações
- O sistema utiliza serviços pagos do Azure. Certifique-se de que suas chaves e endpoints estão corretos.
- Para dúvidas sobre configuração do Azure, consulte a [documentação oficial](https://learn.microsoft.com/pt-br/azure/).
- Para instalar dependências adicionais, edite o arquivo `requirements.txt`.

---

Desenvolvido para fins educacionais.
