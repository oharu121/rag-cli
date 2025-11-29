"""
RAG (Retrieval-Augmented Generation) Practice

A simple RAG implementation for learning purposes.
- Embedding: sentence-transformers (free, local)
- Vector DB: Chroma (with persistence)
- LLM: Google Gemini Flash (free tier)
"""
print("Starting RAG Practice...", flush=True)

import os
from pathlib import Path

from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Good for multilingual
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents():
    """Load all text documents from the documents directory."""
    if not DOCUMENTS_DIR.exists():
        print(f"Creating documents directory: {DOCUMENTS_DIR}")
        DOCUMENTS_DIR.mkdir(parents=True)
        return []

    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents):
    """Split documents into chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def get_embeddings():
    """Initialize the embedding model (runs locally, no API cost)."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}", flush=True)
    print("(First run downloads ~500MB model. Please wait...)", flush=True)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have GPU
    )


def get_or_create_vectorstore(chunks=None, embeddings=None):
    """Get existing vectorstore or create new one from chunks."""
    if CHROMA_DB_DIR.exists() and chunks is None:
        # Load existing vectorstore
        print(f"Loading existing vectorstore from: {CHROMA_DB_DIR}")
        return Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings,
        )

    if chunks is None:
        raise ValueError("No existing vectorstore and no chunks provided")

    # Create new vectorstore
    print(f"Creating new vectorstore at: {CHROMA_DB_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )
    return vectorstore


def get_llm():
    """Initialize Gemini Flash LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Get your free API key at: https://aistudio.google.com/apikey"
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
    )


def create_rag_chain(vectorstore, llm):
    """Create the RAG chain."""
    # Create retriever (get top 3 most relevant chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG prompt template
    template = """Answer the question based on the following context.
If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Format retrieved documents into a single string
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Build the chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    print("=" * 50, flush=True)
    print("RAG Practice - Learning Setup", flush=True)
    print("=" * 50, flush=True)

    # Initialize embeddings (local, free)
    embeddings = get_embeddings()
    print("Embeddings ready!", flush=True)

    # Check if we need to rebuild the vectorstore
    rebuild = not CHROMA_DB_DIR.exists()

    if rebuild:
        # Load and process documents
        documents = load_documents()
        if not documents:
            print("\nNo documents found!")
            print(f"Add .txt files to: {DOCUMENTS_DIR}")
            return

        chunks = split_documents(documents)
        vectorstore = get_or_create_vectorstore(chunks, embeddings)
    else:
        vectorstore = get_or_create_vectorstore(embeddings=embeddings)

    # Initialize LLM
    try:
        llm = get_llm()
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo fix this:")
        print("1. Copy .env.example to .env")
        print("2. Add your Google API key")
        return

    # Create RAG chain
    chain = create_rag_chain(vectorstore, llm)

    print("\n" + "=" * 50)
    print("Ready! Ask questions about your documents.")
    print("Type 'quit' to exit, 'rebuild' to reload documents.")
    print("=" * 50 + "\n")

    # Interactive loop
    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() == "quit":
                print("Goodbye!")
                break

            if question.lower() == "rebuild":
                print("Rebuilding vectorstore...")
                documents = load_documents()
                chunks = split_documents(documents)
                # Delete old vectorstore
                import shutil
                if CHROMA_DB_DIR.exists():
                    shutil.rmtree(CHROMA_DB_DIR)
                vectorstore = get_or_create_vectorstore(chunks, embeddings)
                chain = create_rag_chain(vectorstore, llm)
                print("Done! Vectorstore rebuilt.\n")
                continue

            # Get response from RAG chain
            response = chain.invoke(question)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
