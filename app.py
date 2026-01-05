from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt


# 🔹 Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Safety check (important)
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

# 🔹 Flask app (template path fix)
app = Flask(__name__, template_folder="src/templates")

# 🔹 Load embeddings
embeddings = download_hugging_face_embeddings()

# 🔹 Load existing Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 🔹 Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 🔹 Groq LLM (FREE)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

# 🔹 Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# 🔹 RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# 🔹 Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "Please enter a question"

    response = rag_chain.invoke({"input": msg})
    return response["answer"]


# 🔹 Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
