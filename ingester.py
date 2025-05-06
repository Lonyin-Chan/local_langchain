import os
import warnings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.
    
    This function loads data from Lilian Weng's blog post about agents,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.
    """
    # Initialize web loader for Lilian Weng's blog post
    web_loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    loaded_documents = web_loader.load()
    print(f"Loaded {len(loaded_documents)} document(s) from the website")
    
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    chunked_documents = text_splitter.split_documents(loaded_documents)
    print(f"Split into {len(chunked_documents)} chunks")
    
    # Initialize Ollama Embeddings
    ollama_embeddings = OllamaEmbeddings(model="llama3")
    
    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=ollama_embeddings,
        persist_directory=DB_DIR,
    )
    
    vector_database.persist()
    print(f"Vector database created and persisted to {DB_DIR}")
    
    # Example query to test the database
    query = "What are the key components of an agent?"
    docs = vector_database.similarity_search(query, k=2)
    
    # Print results of the example query
    print("\nExample query results:")
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(doc.page_content)

if __name__ == "__main__":
    # Ensure the database directory exists
    os.makedirs(DB_DIR, exist_ok=True)
    create_vector_database()