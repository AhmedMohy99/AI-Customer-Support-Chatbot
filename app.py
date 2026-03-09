# =========================
# AI Customer Support Chatbot
# Streamlit + LangChain + OpenAI + FAISS
# =========================

# Import Python built-in libraries
import os
import tempfile

# Import Streamlit for the web app UI
import streamlit as st

# Import PDF reader to extract text from uploaded PDF files
from pypdf import PdfReader

# Import LangChain document object
from langchain_core.documents import Document

# Import text splitter to break long text into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import OpenAI embeddings and chat model
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Import FAISS vector database
from langchain_community.vectorstores import FAISS


# =========================
# PAGE CONFIG
# =========================

# Set browser tab title and page layout
st.set_page_config(
    page_title="AI Customer Support Agent",
    page_icon="🤖",
    layout="wide"
)

# Main app title
st.title("🤖 AI Customer Support Agent")

# Small description under the title
st.write(
    "Upload your business PDF files, build a knowledge base, and ask questions about your business."
)


# =========================
# API KEY SETUP
# =========================

# Read the OpenAI API key from Streamlit secrets
# You will add this later inside Streamlit Cloud settings
# Example:
# OPENAI_API_KEY = "your_api_key_here"
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = None

# If there is no API key, show an error and stop the app
if not OPENAI_API_KEY:
    st.error(
        "OpenAI API key not found. Please add OPENAI_API_KEY to your Streamlit secrets."
    )
    st.stop()

# Put the API key into environment variables so LangChain can use it
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# =========================
# HELPER FUNCTION:
# READ TEXT FROM PDF
# =========================

def extract_text_from_pdf(uploaded_file):
    """
    Reads one uploaded PDF file and returns all extracted text as one long string.
    """

    # Create a PDF reader object from the uploaded file
    reader = PdfReader(uploaded_file)

    # Create an empty list to store text from each page
    pages_text = []

    # Loop through every page in the PDF
    for page in reader.pages:
        # Extract text from the page
        page_text = page.extract_text()

        # If text exists, save it
        if page_text:
            pages_text.append(page_text)

    # Join all page texts into one big string
    return "\n".join(pages_text)


# =========================
# HELPER FUNCTION:
# CONVERT PDFs TO DOCUMENTS
# =========================

def build_documents_from_uploaded_pdfs(uploaded_files):
    """
    Converts uploaded PDF files into LangChain Document objects.
    Each document contains the full text of one PDF plus metadata.
    """

    # Create an empty list for all documents
    documents = []

    # Loop through all uploaded PDF files
    for uploaded_file in uploaded_files:
        # Extract the text from the current PDF
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Only add the file if it contains text
        if pdf_text and pdf_text.strip():
            # Create a Document object with text + source filename metadata
            doc = Document(
                page_content=pdf_text,
                metadata={"source": uploaded_file.name}
            )

            # Add the document to our list
            documents.append(doc)

    # Return all documents
    return documents


# =========================
# HELPER FUNCTION:
# SPLIT DOCUMENTS INTO CHUNKS
# =========================

def split_documents(documents):
    """
    Splits large documents into smaller chunks for better retrieval quality.
    """

    # Create a recursive text splitter
    # chunk_size controls chunk length
    # chunk_overlap keeps some repeated context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Return the smaller chunks
    return chunks


# =========================
# HELPER FUNCTION:
# BUILD VECTOR DATABASE
# =========================

def create_vector_store(chunks):
    """
    Creates a FAISS vector store from document chunks using OpenAI embeddings.
    """

    # Create embedding model
    # text-embedding-3-small is a current OpenAI embedding model option
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Convert chunks into vectors and store them in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Return the vector database
    return vector_store


# =========================
# HELPER FUNCTION:
# GENERATE FINAL ANSWER
# =========================

def answer_question(vector_store, question):
    """
    Searches the vector store for relevant chunks, then asks the chat model
    to answer using only the retrieved context.
    """

    # Retrieve the most relevant chunks from FAISS
    retrieved_docs = vector_store.similarity_search(question, k=4)

    # Join the retrieved chunks into one context block
    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )

    # Create the chat model
    # Temperature 0 makes answers more consistent
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # Build a strict prompt to reduce hallucinations
    prompt = f"""
You are a professional AI customer support assistant.

Answer the user's question using ONLY the context below.
If the answer is not found in the context, say:
"I could not find that information in the uploaded documents."

Be clear, professional, and concise.
If helpful, use bullet points.

Context:
{context}

User question:
{question}
"""

    # Ask the model for a final answer
    response = llm.invoke(prompt)

    # Return both the answer and the retrieved sources
    return response.content, retrieved_docs


# =========================
# SESSION STATE
# =========================

# Save vector store in session so it stays available after reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Save whether indexing is complete
if "knowledge_ready" not in st.session_state:
    st.session_state.knowledge_ready = False


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    # Sidebar title
    st.header("⚙️ Setup")

    # File uploader for one or many PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Button to build the knowledge base
    build_button = st.button("Build Knowledge Base")

    # Small help text
    st.caption("Upload your FAQ, policy, manual, product, or support documents.")


# =========================
# BUILD KNOWLEDGE BASE
# =========================

if build_button:
    # Check whether the user uploaded files
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        # Show a loading spinner while processing
        with st.spinner("Reading PDFs, splitting text, and building the knowledge base..."):
            # Convert PDFs into documents
            documents = build_documents_from_uploaded_pdfs(uploaded_files)

            # If no text could be extracted, stop
            if not documents:
                st.error("No readable text was found in the uploaded PDF files.")
                st.stop()

            # Split into chunks
            chunks = split_documents(documents)

            # Build FAISS vector database
            vector_store = create_vector_store(chunks)

            # Save to session state
            st.session_state.vector_store = vector_store
            st.session_state.knowledge_ready = True

        # Success message after building
        st.success("Knowledge base created successfully.")


# =========================
# MAIN CHAT AREA
# =========================

# Show instructions when database is not ready
if not st.session_state.knowledge_ready:
    st.info("Upload PDF files and click 'Build Knowledge Base' to start.")
else:
    # Question input box
    user_question = st.text_input("Ask a question about your uploaded documents:")

    # When user asks something
    if user_question:
        # Show loading spinner while generating answer
        with st.spinner("Searching documents and generating answer..."):
            # Get answer from the AI
            answer, source_docs = answer_question(
                st.session_state.vector_store,
                user_question
            )

        # Show the final answer
        st.subheader("Answer")
        st.write(answer)

        # Show the sources used
        st.subheader("Sources Used")
        for i, doc in enumerate(source_docs, start=1):
            st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
            st.write(doc.page_content[:500] + "...")
            st.markdown("---")
