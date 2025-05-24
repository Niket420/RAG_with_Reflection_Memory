import asyncio
from datetime import datetime
import uuid

from main import read_pdf, splitting_pdfs, get_embeddings, retrieve, generate, self_reflection, GetModel
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configs
collection_name = "reflection_memory"
embedding_dim = 3072
reflection_start = 5  # Activate reflection memory after this many turns

# Initialize Qdrant in-memory
client = QdrantClient(":memory:")
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
)

# Initialize models
model = GetModel()
embedding_function = model.embedding

# Setup reflection memory vector store
reflection_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embedding_function,
)

# Load and index document
async def prepare_knowledge_base(file_path):
    pages = await read_pdf(file_path)
    all_splits = splitting_pdfs(pages)
    embeddings = get_embeddings(all_splits)
    return all_splits, embeddings

# Store a past interaction into reflection memory
def log_to_reflection_memory(question, answer, context, embedding_fn):
    embedding = embedding_fn.embed_query(question + answer)
    reflection_store.add_texts(
        texts=[answer],
        metadatas=[{
            "question": question,
            "context": context,
            "timestamp": str(datetime.now())
        }],
        ids=[str(uuid.uuid4())]
    )

# Full flow for a single query
def rag_with_reflection(state, turn):
    query = state["question"]

    if turn >= reflection_start:
        # Retrieve from reflection memory
        reflection_results = reflection_store.similarity_search(query, k=3)
        state["reflection_context"] = [doc.page_content for doc in reflection_results]

        # Now combine reflection + document retrieval
        factual_state = retrieve(state)
        state["context"] = "\n\n".join(state["reflection_context"] + factual_state["context"])
    else:
        state.update(retrieve(state))

    # Generate final answer using combined context
    state.update(generate(state))
    return state

# Run interaction
async def run_chat():
    file_path = "/Users/niketanand/Documents/MLOps/RAG_with_Reflection_Memory/copilot_tuning.pdf"
    all_splits, _ = await prepare_knowledge_base(file_path)

    history = []
    turn = 0

    while True:
        question = input("\n💬 You: ")
        if question.lower() in ["exit", "quit"]:
            break

        state = {"question": question}
        state = rag_with_reflection(state, turn)

        answer = state["answer"]
        print(f"🤖 RAG: {answer}")
        log_to_reflection_memory(question, answer, state["context"], embedding_function)

        turn += 1

# Run main
if __name__ == "__main__":
    asyncio.run(run_chat())