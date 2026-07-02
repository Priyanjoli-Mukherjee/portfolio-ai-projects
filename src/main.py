import os
from dotenv import load_dotenv

# Load the values from the .env file into the environment
load_dotenv()

# Access the hidden variables
secret_key = os.getenv("API_SECRET_KEY")

print(f"Using API key: {secret_key}")

loader = PyPDFLoader(DOC_PATH)
pages = loader.load()