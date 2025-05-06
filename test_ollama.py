# test_ollama.py
from langchain_ollama import OllamaLLM

print("Initializing Ollama...")
llm = OllamaLLM(model="gemma3")
print("Connection successful!")

print("Testing generation...")
response = llm.invoke("Hello")
print(f"Response: {response}")