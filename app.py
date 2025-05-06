"""
RAG Application with Chainlit Interface
--------------------------------------
This application has two main components:

1. ingester.py - Fetches and processes web content from Lilian Weng's blog post on agents
2. app.py - Chainlit interface for chatting with the processed content

Run the ingester first to create the vector database, then run the app to start the chat interface.
"""
import os
from typing import List
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA

# Path to the vector database
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

# Set up RAG prompt template
rag_prompt_llama3 = hub.pull("rlm/rag-prompt-llama3")

def load_model():
    """Initialize and return the Ollama LLM with the llama3 model."""
    llm = Ollama(
        model="llama3",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

def retrieval_qa_chain(llm, vectorstore):
    """Create a retrieval QA chain using the provided LLM and vector store."""
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_llama3},
        return_source_documents=True,
    )
    return qa_chain

def qa_bot():
    """Initialize the QA bot with the LLM and vector store."""
    llm = load_model()
    
    # Check if DB exists, if not, create it
    if not os.path.exists(DB_DIR):
        from ingester import create_vector_database
        create_vector_database()
        print("Created new vector database as it didn't exist.")
    
    # Load the vector store
    vectorstore = Chroma(
        persist_directory=DB_DIR, 
        embedding_function=OllamaEmbeddings(model="llama3")
    )

    # Create the QA chain
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa

@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Welcome to Chat With Lilian Weng's Agent Blog Post using Ollama (llama3 model) and LangChain.\n\n"
        "Ask me anything about AI agents from the blog post: https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    
    # Process the message
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
