import os
import shutil
import hashlib
import streamlit as st

import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# Load the values from the .env file into the environment
load_dotenv()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "chroma_dir" not in st.session_state:
    st.session_state.chroma_dir = None

def get_pdf_hash(uploaded_files):
    hasher = hashlib.sha256()

    # Sort so upload order doesn't matter
    for pdf in sorted(uploaded_files, key=lambda f: f.name):
        pdf.seek(0)
        hasher.update(pdf.read())
        pdf.seek(0)

    return hasher.hexdigest()

st.title("Chat With Your PDF")

# Access the hidden variables
OPENAI_API_KEY = os.getenv("API_SECRET_KEY")

uploaded_pdfs = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

current_hash = None

if uploaded_pdfs:
    current_hash = get_pdf_hash(uploaded_pdfs)

db_chroma = st.session_state.db

if (
    uploaded_pdfs
    and current_hash != st.session_state.pdf_hash
    ):

    if (
        st.session_state.chroma_dir
        and os.path.exists(st.session_state.chroma_dir)
    ):
        shutil.rmtree(st.session_state.chroma_dir)

    all_pages = []

    for uploaded_file in uploaded_pdfs:

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Add the uploaded filename to every page
            for page in pages:
                page.metadata["source_pdf"] = uploaded_file.name

            all_pages.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(all_pages)

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    persist_directory = f"./chroma_db/{uuid.uuid4()}"

    db_chroma = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory
    )

    st.session_state.db = db_chroma
    st.session_state.pdf_hash = current_hash
    st.session_state.chroma_dir = persist_directory

query = st.chat_input(
    "Ask a question..."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query:
    
    st.session_state.messages.append(
    {
        "role": "user",
        "content": query
    }
    )

    with st.chat_message("user"):
        st.markdown(query)

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

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(prompt)
            response_text = response.content

        st.markdown(response_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
)
