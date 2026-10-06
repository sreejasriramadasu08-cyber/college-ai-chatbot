import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Gemini API
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
db_client = chromadb.PersistentClient(path="./chroma_db")

collection = db_client.get_collection(
    name="college_documents"
)

# Streamlit page
st.set_page_config(
    page_title="College AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 College AI Chatbot")
st.caption("Ask questions from your college documents 📚")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask your question...")

if question:

    # Display user message
    st.chat_message("user").markdown(question)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search relevant PDF content
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=8
    )
    
    context = "\n\n".join(
        results["documents"][0]
    )
    
    # Gemini prompt
    prompt = f"""
You are a helpful college student chatbot.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say: "I couldn't find this information in the document."

Context:
{context}

Question:
{question}

Give a simple and clear answer.
"""

    # Generate answer
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt
    )

    answer = response.text

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

