from main import read_pdf, splitting_pdfs, get_embeddings, retrieve, generate
from openevals.llm import create_llm_as_judge
from openevals.prompts import RAG_HELPFULNESS_PROMPT
import os
import asyncio
from main import GetModel


file_path = "/Users/niketanand/Documents/MLOps/RAG_with_Reflection_Memory/copilot_tuning.pdf"
pages  = asyncio.run(read_pdf(file_path))
all_splits = splitting_pdfs(pages)
embeddings = get_embeddings(all_splits)

question = "can you tell me about the name of this paper?"
state = {"question": question}


state.update(retrieve(state))
state.update(generate(state))


llm_model = GetModel()
llm = llm_model.llm
print(llm.invoke("who is sachine tendulkar?"))



helpfulness_evaluator = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    model="openai:o3-mini",
)

inputs = {
    "question": "Where was the first president of FoobarLand born?",
}

outputs = {
    "answer": "The first president of FoobarLand was Bagatur Askaryan.",
}

eval_result = helpfulness_evaluator(
  inputs=inputs,
  outputs=outputs,
)

print(eval_result)