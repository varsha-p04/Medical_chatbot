from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from dotenv import load_dotenv
from src.prompt import system_prompt
from pinecone import Pinecone as PineconeClient
from groq import Groq
from pydantic import PrivateAttr  # For Groq class fix
import os


# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""

# Download Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone
index_name = "medicalbot"
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Set up the retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

from groq import Groq

class LangChainGroq(LLM):
    _model: any = PrivateAttr()

    def __init__(self, model):
        super().__init__()
        self._model = model  # Store Groq client

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        """Call the Groq model and return the response."""
        # Use the correct API call for Groq
        chat_completion = self._model.chat.completions.create(
            messages=[
                {"role": "system", "content": "you are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            model="mixtral-8x7b-32768",  # Ensure you use the correct model
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        
        # Return the content of the response
        return chat_completion.choices[0].message.content


# Instantiate the Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = LangChainGroq(model=groq_client)

# Set up the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/chatbot")
def chatbot_ui():
    return render_template('chat.html') 
@app.route("/appointments")
def appointments():
    return render_template("dashboard__appointments.html")

@app.route("/doctors")
def view_doctors():
    return render_template("dashboard__doctors.html")




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print("User Input:", input_text)
    response = rag_chain.invoke({"input": input_text})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
