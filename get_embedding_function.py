from langchain_community.embeddings import OllamaEmbeddings
#from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings


def get_embedding_function():
    """embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")   #274mb
    return embeddings

#we can use LangSmith later to evaluate teh performance 
