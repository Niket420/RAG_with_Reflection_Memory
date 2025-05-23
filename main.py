import getpass
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate
import asyncio 
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore


load_dotenv()

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


# embeddings = AzureOpenAIEmbeddings(
#         azure_endpoint=endpoint,
#         azure_deployment=embedding_deployment,
#         openai_api_version=api_version,
#         model="text-embedding-3-large" 
#             )

embeddings = AzureOpenAIEmbeddings(
            azure_deployment = os.getenv("OPENAI_DEPLOYMENT_NAME"),
            azure_endpoint = os.getenv("OPENAI_ENDPOINT"),
            api_key = os.getenv("OPENAI_API_KEY")
        )


client = QdrantClient(":memory:")
embedding_dim = 3072  
collection_name = "test2"


# Create the collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(
            client=client,
            collection_name="test2",
            embedding=embeddings,
        )



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# loading the pdf file slowly ,one by one
async def read_pdf(file_path):
  loader = PyPDFLoader(file_path)
  pages = []
  async for page in loader.alazy_load():
    pages.append(page)
  return pages



def splitting_pdfs(pages):
  all_splits = text_splitter.split_documents(pages)
  return all_splits


def get_embeddings(all_splits):
   document_ids = vector_store.add_documents(documents=all_splits)
   return document_ids


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str



# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"],k=5)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    print(response.content)
    return {"answer": response.content}


# make sure prompt.invoke works correctly| dekh le bhai same hi hai
prompt = PromptTemplate.from_template("you are a helpful assistant. Answer the question based on the context provided."
                                      "If you don't know the answer, still answer something related to it, but give empty reply'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:")




if __name__=="__main__":
  file_path = "/Users/niketanand/Documents/MLOps/RAG_with_Reflection_Memory/copilot_tuning.pdf"
  pages  = asyncio.run(read_pdf(file_path))
  all_splits = splitting_pdfs(pages)
  embeddings = get_embeddings(all_splits)

  question = "can you tell me about the name of this paper?"
  state = {"question": question}


  state.update(retrieve(state))
  state.update(generate(state))
  
 




