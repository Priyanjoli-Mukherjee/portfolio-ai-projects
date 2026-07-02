import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load the values from the .env file into the environment
load_dotenv()

# Access the hidden variables
secret_key = os.getenv("API_SECRET_KEY")

print(f"Using API key: {secret_key}")

loader = PyPDFLoader(DOC_PATH)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)