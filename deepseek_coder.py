import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_tfidf_vectors(data):
    documents = [entry['question'] + ' ' + entry['code'] for entry in data]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

def retrieve_documents(query, data, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_n_indices = np.argsort(-similarities)[:5]
    return [data[i] for i in top_n_indices if similarities[i] > 0]

def generate_response_with_chain_of_thought(query, documents):
    reasoning = 'Based on the provided code and context, '
    context = ' '.join([f"{doc['question']} Hence, {doc['answer']}." for doc in documents])
    reasoning += context
    combined_input = f"Question: {query}\n\n{reasoning}\n\nTherefore,"
    response = ollama.chat(
        model='deepseek-coder',
        messages=[{"role": "user", "content": combined_input}]
    )
    return response['message']['content']

from tqdm import tqdm

json_path = 'smallest_dataset.json'
data = load_json_data(json_path)
vectorizer, tfidf_matrix = create_tfidf_vectors(data)

updated_data = []
for entry in tqdm(data, desc='Generating predictions'):
    question = entry['question']
    code = entry['code']
    query = f"{code}\n{question}"
    relevant_docs = retrieve_documents(query, data, vectorizer, tfidf_matrix)
    if not relevant_docs:
        prediction = 'No relevant documents found.'
    else:
        try:
            prediction = generate_response_with_chain_of_thought(query, relevant_docs)
        except Exception as e:
            prediction = f'Error: {str(e)}'
    entry['prediction'] = prediction
    updated_data.append(entry)

output_path = 'output_data_with_predictions.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, indent=2)

print(f"\n✅ Predictions saved to: {output_path}")

def main():
    json_path = 'smallest_dataset.json'
    data = load_json_data(json_path)
    vectorizer, tfidf_matrix = create_tfidf_vectors(data)

    updated_data = []
    for entry in tqdm(data, desc='Generating predictions'):
        question = entry['question']
        code = entry['code']
        query = f"{code}\n{question}"
        relevant_docs = retrieve_documents(query, data, vectorizer, tfidf_matrix)
        if not relevant_docs:
            prediction = 'No relevant documents found.'
        else:
            try:
                prediction = generate_response_with_chain_of_thought(query, relevant_docs)
            except Exception as e:
                prediction = f'Error: {str(e)}'
        entry['prediction'] = prediction
        updated_data.append(entry)

    output_path = 'output_data_with_predictions.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2)

    print(f"\n✅ Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()
