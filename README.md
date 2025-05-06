## Setup Instructions

1. Make sure you have Ollama installed and running with the llama3 model:
   ```
   ollama pull llama3
   ```

2. Install the required Python packages:
   ```
   pip install langchain langchain_community chainlit chromadb
   ```

3. Run the ingester to create the vector database:
   ```
   python -c "from ingester import create_vector_database; create_vector_database()"
   ```

4. Start the Chainlit application:
   ```
   chainlit run app.py
   ```

5. Open your browser and navigate to http://localhost:8000

## How it Works

1. The ingester fetches content from Lilian Weng's blog post about agents
2. The content is split into chunks and embedded using Ollama's llama3 model
3. The embeddings are stored in a Chroma vector database
4. The Chainlit interface allows you to ask questions about the content
5. When you ask a question, the most relevant chunks are retrieved and used to generate a response

## Files

- `ingester.py`: Fetches and processes the web content
- `app.py`: Chainlit interface for the RAG application
"""