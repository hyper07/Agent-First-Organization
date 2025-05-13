from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import os

def get_huggingface_llm(model, **kwargs):
    llm = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        **kwargs
    )
    return ChatHuggingFace(llm=llm)

def get_ollama_llm(model, **kwargs):
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:8434")
    return ChatOllama(model=model, base_url=ollama_base_url, **kwargs)

LLM_PROVIDERS = ["openai", "gemini", "anthropic", "huggingface", "ollama"]

PROVIDER_MAP = {
    "anthropic": ChatAnthropic,
    "gemini": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
    "huggingface": get_huggingface_llm,
    "ollama": get_ollama_llm
}

PROVIDER_EMBEDDINGS = {
    "anthropic": HuggingFaceEmbeddings,
    "gemini": GoogleGenerativeAIEmbeddings,
    "openai": OpenAIEmbeddings ,
    "huggingface": HuggingFaceEmbeddings,
    "ollama": HuggingFaceEmbeddings
}
PROVIDER_EMBEDDING_MODELS = {
    "anthropic": "sentence-transformers/sentence-t5-base",
    "gemini": "models/embedding-001",
    "openai": "text-embedding-ada-002",
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
    "ollama": "sentence-transformers/all-mpnet-base-v2",
}