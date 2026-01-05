from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# 1️⃣ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # replace OpenAI key with Groq

# 2️⃣ Validate keys
if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY not found in .env")

if GROQ_API_KEY is None:
    print("Warning: GROQ_API_KEY not found, but Groq LLM will need it when querying.")

# 3️⃣ Set environment variables
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found in .env")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 4️⃣ Load and process PDFs
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

print(f"Number of text chunks extracted: {len(text_chunks)}")

# 5️⃣ Load embeddings
embeddings = download_hugging_face_embeddings()  # must return embedding object

# 6️⃣ Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# 7️⃣ Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # make sure it matches embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# 8️⃣ Create Pinecone vector store from documents
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("✅ Pinecone index created and documents stored successfully!")
