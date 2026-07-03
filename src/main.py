import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Load the values from the .env file into the environment
load_dotenv()

# Access the hidden variables
OPENAI_API_KEY = os.getenv("API_SECRET_KEY")

loader = PyPDFLoader("./data/alphabet_10K_2022.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

query = 'what are the top risks mentioned in the document?'

docs_chroma = db_chroma.similarity_search_with_score(query, k=5)

context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
response_text = model.predict(prompt)

print(response_text)