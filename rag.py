import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import ollama
import os


class CodeBERTEmbedder:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
        return np.array(embeddings)


def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def save_faiss(index, index_path, id_map, idmap_path):
    faiss.write_index(index, index_path)
    with open(idmap_path, "w") as f:
        json.dump(id_map, f)


def load_faiss(index_path, idmap_path):
    index = faiss.read_index(index_path)
    with open(idmap_path, "r") as f:
        id_map = json.load(f)
    return index, id_map


def build_index(dataset, embedder, index_path="index.faiss", idmap_path="id_map.json"):
    if os.path.exists(index_path) and os.path.exists(idmap_path):
        return load_faiss(index_path, idmap_path)

    texts = [item["code"] for item in dataset]
    embeddings = embedder.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    id_map = {i: dataset[i]["code"] for i in range(len(dataset))}

    save_faiss(index, index_path, id_map, idmap_path)
    return index, id_map


def retrieve_context(query, embedder, index, id_map, top_k=2):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [id_map[str(i)] for i in I[0]]


def generate_answer(question, context_snippets):
    context = "\n\n".join(context_snippets)
    prompt = f"""# Context
{context}

# Question
{question}

# Answer
"""
    response = ollama.chat(
        model="deepseek-coder",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


def batch_answer(dataset, embedder, index, id_map):
    results = []
    for item in dataset:
        question = item["question"]
        context = retrieve_context(question, embedder, index, id_map)
        answer = generate_answer(question, context)
        results.append({
            "question": question,
            "expected": item.get("answer", ""),
            "generated": answer.strip()
        })
    return results


if __name__ == "__main__":
    dataset = load_dataset("dataset.json")
    embedder = CodeBERTEmbedder()
    index, id_map = build_index(dataset, embedder)

    answers = batch_answer(dataset, embedder, index, id_map)

    print("\nGenerated Answers:\n")
    for item in answers:
        print(f"Q: {item['question']}")
        print(f"Expected: {item['expected']}")
        print(f"Generated: {item['generated']}")
        print("-" * 60)

    # Optional: save to file
    with open("generated_answers.json", "w") as f:
        json.dump(answers, f, indent=2)
