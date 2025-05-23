import getpass
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv


api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")





if not os.environ.get("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")



llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=chat_deployment,
        openai_api_version=api_version,
    )


embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        azure_deployment=embedding_deployment,
        openai_api_version=api_version,
            )


client = QdrantClient(":memory:")
vector_store = QdrantVectorStore(
            client=client,
            collection_name="test",
            embedding=embeddings,
        )


