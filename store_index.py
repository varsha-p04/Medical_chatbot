from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_community.vectorstores import Pinecone  # Corrected import
from pinecone import Pinecone as PineconeClient, ServerlessSpec  # Corrected Pinecone import
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Ensure your Pinecone API key is correctly loaded
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# If you want to hardcode the API key (not recommended for production), make sure it matches
# os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key_here"

# Extract data from PDF
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)

# Download Hugging Face Embeddings
embeddings = download_hugging_face_embeddings()

# Create a Pinecone client instance using the correct method
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Define the name of the index
index_name = "medicalbot"

# Check if the index already exists, otherwise create a new one
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,  # Ensure this matches your embeddings dimensions
        metric="cosine",  # Ensure this is the correct similarity metric for your use case
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Using the Pinecone vector store from LangChain
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

