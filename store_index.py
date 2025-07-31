from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_community.vectorstores import Pinecone  
from pinecone import Pinecone as PineconeClient, ServerlessSpec  
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')



extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)


embeddings = download_hugging_face_embeddings()


pc = PineconeClient(api_key=PINECONE_API_KEY)


index_name = "medicalbot"


if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine",  
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

