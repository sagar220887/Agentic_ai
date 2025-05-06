from fastapi import FastAPI
import ollama

app = FastAPI()
ollama_client = ollama.OllamaClient()
