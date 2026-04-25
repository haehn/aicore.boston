#
# conda create --name x1025 python=3.10
# conda activate x1025
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# pip install -r requirements.txt

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -----------------------------
# Step 1: Setup device
# -----------------------------
def step1_setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
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

    print("Documents ingested")


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

    return tokenizer, model


# -----------------------------
# Step 6: Query + RAG pipeline
# -----------------------------
def step6_ask(query, embed_model, collection, tokenizer, model, device):

    q_emb = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=2
    )

    context = "\n".join(results["documents"][0])

    print("\nRetrieved Context:")
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
        max_new_tokens=200
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer



def main():

    device = step1_setup_device()

    embed_model = step2_load_embedding_model(device)

    collection = step3_init_vector_db()

    step4_ingest_documents(collection, embed_model)

    tokenizer, model = step5_load_llm(device)

    query = "Q: What is included in a noon report?"
    answer = step6_ask(query, embed_model, collection, tokenizer, model, device)

    print(answer)


if __name__ == "__main__":
    main()
