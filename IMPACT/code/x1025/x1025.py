#
# conda create --name x1025 python=3.1
# conda activate x1025
# pip install chromadb sentence-transformers transformers accelerate torch
#

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -----------------------------
# Step 1: Setup device
# -----------------------------
def step1_setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    return device


# -----------------------------
# Step 2: Load embedding model
# -----------------------------
def step2_load_embedding_model(device):
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return model


# -----------------------------
# Step 3: Initialize vector DB
# -----------------------------
def step3_init_vector_db():
    client = chromadb.Client()
    collection = client.create_collection(name="x1025")
    return collection


# -----------------------------
# Step 4: Ingest documents
# -----------------------------
def step4_ingest_documents(collection, embed_model):
    docs = [
        "Fuel transfer procedure: Ensure valves are closed before pump start.",
        "Emergency protocol: In case of fire, activate CO2 suppression system.",
        "Noon report includes fuel consumption, position, and speed.",
    ]

    embeddings = embed_model.encode(docs).tolist()

    for i, d in enumerate(docs):
        collection.add(
            documents=[d],
            embeddings=[embeddings[i]],
            ids=[str(i)]
        )

    print("[INFO] Documents ingested")


# -----------------------------
# Step 5: Load LLM on GPU
# -----------------------------
def step5_load_llm(device):
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"   # uses your MIG GPU
    )

    print("[INFO] LLM loaded")

    return tokenizer, model


# -----------------------------
# Step 6: Query + RAG pipeline
# -----------------------------
def step6_ask(query, embed_model, collection, tokenizer, model, device):

    # Embed query
    q_emb = embed_model.encode([query]).tolist()

    # Retrieve context
    results = collection.query(
        query_embeddings=q_emb,
        n_results=2
    )

    context = "\n".join(results["documents"][0])

    print("\n[DEBUG] Retrieved Context:")
    print(context)

    # Prompt
    prompt = f"""
    You are a maritime assistant.
    Answer based ONLY on the context.

    Context:
    {context}

    Question:
    {query}
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer



def main():

    # Step 1
    device = step1_setup_device()

    # Step 2
    embed_model = step2_load_embedding_model(device)

    # Step 3
    collection = step3_init_vector_db()

    # Step 4
    step4_ingest_documents(collection, embed_model)

    # Step 5
    tokenizer, model = step5_load_llm(device)

    # Step 6 (test query)
    query = "What is included in a noon report?"
    answer = step6_ask(query, embed_model, collection, tokenizer, model, device)

    print("\n[FINAL ANSWER]")
    print(answer)


if __name__ == "__main__":
    main()
