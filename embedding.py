
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  
        google_api_key="mykey"
    )
    return embeddings