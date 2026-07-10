import os
import streamlit as st

import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Load the values from the .env file into the environment
load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

st.title("Chat With Your PDF")

# Access the hidden variables
OPENAI_API_KEY = os.getenv("API_SECRET_KEY")

uploaded_pdfs = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

db_chroma = st.session_state.db

if uploaded_pdfs and db_chroma is None:

    all_pages = []

    for uploaded_file in uploaded_pdfs:

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            all_pages.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(all_pages)


    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )


    db_chroma = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=f"./chroma_db/{uuid.uuid4()}"
    )

    st.session_state.db = db_chroma

query = st.chat_input(
    "Ask a question..."
)

if query:
    
    st.session_state.messages.append(
    {
        "role": "user",
        "content": query
    }
    )

    if db_chroma is None:
        st.warning("Please upload one or more PDFs first.")
        st.stop()
        
    docs_chroma = db_chroma.similarity_search_with_score(
        query,
        k=5
    )

    context_text = "\n\n".join(
        [doc.page_content for doc, _score in docs_chroma]
    )

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}

    Answer the question based on the above context:
    {question}.

    Provide a detailed answer.

    Don’t justify your answers.

    Don’t give information not mentioned in the CONTEXT INFORMATION.

    Do not say "according to the context".
    """

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt = prompt_template.format(
        context=context_text,
        question=query
    )

    model = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY
    )

    response_text = model.predict(prompt)

    st.session_state.messages.append(
    {
        "role": "assistant",
        "content": response_text
    }
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])