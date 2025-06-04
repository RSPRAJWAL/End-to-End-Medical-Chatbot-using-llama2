from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
from dotenv import load_dotenv  
import os

# Load environment variables from .env file
load_dotenv()

# Get Pinecone API credentials from environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

# Print to confirm they loaded (optional for debugging, remove in production)
#print("PINECONE_API_KEY:", PINECONE_API_KEY)
#print("PINECONE_API_ENV:", PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index_name="medical-chatbot"

# Createating embeding for each of the text chunks & storing them
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=embeddings)