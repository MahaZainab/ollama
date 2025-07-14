import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


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


def load_dataset(path="complete_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)


def save_index_and_idmap(dataset, embedder, index_path="index.faiss", idmap_path="id_map.json"):
    texts = [item["code"] for item in dataset]
    embeddings = embedder.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    id_map = {i: dataset[i]["code"] for i in range(len(dataset))}

    faiss.write_index(index, index_path)
    with open(idmap_path, "w") as f:
        json.dump(id_map, f)


if __name__ == "__main__":
    dataset = load_dataset()
    embedder = CodeBERTEmbedder()
    save_index_and_idmap(dataset, embedder)
    print("Saved FAISS index to 'index.faiss' and ID map to 'id_map.json'.")
