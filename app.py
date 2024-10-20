import os
import re
import math
import logging
from typing import List, Dict
from collections import Counter
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

class BM25Model:
    def __init__(self, dataset_directory: str, k1: float = 1.5, b: float = 0.75):
        self.dataset_directory = dataset_directory
        self.documents: Dict[str, str] = {}
        self.document_vectors: Dict[str, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.avg_doc_length = 0
        self.k1 = k1
        self.b = b
        self.load_documents()
        self.calculate_bm25()

    def load_documents(self):
        """Load all .txt files from the specified directory."""
        total_length = 0
        for filename in os.listdir(self.dataset_directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.dataset_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.documents[filename] = file.read()
                    total_length += len(self.preprocess(self.documents[filename]))
        
        self.avg_doc_length = total_length / len(self.documents)
        print(f"Loaded {len(self.documents)} documents. Average document length: {self.avg_doc_length:.2f}")

    def preprocess(self, text: str) -> List[str]:
        """Preprocess the text by converting to lowercase and tokenizing."""
        return re.findall(r'\w+', text.lower())

    def calculate_bm25(self):
        """Calculate BM25 for all terms in all documents."""
        # Calculate document frequency (DF) for each term
        df = Counter()
        for doc in self.documents.values():
            df.update(set(self.preprocess(doc)))

        # Calculate IDF for BM25
        num_docs = len(self.documents)
        self.idf = {term: math.log((num_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

        # Calculate BM25 score for each document
        for filename, doc in self.documents.items():
            term_freq = Counter(self.preprocess(doc))
            doc_length = sum(term_freq.values())
            self.document_vectors[filename] = {
                term: self.bm25_score(term_freq[term], doc_length, self.idf.get(term, 0))
                for term in term_freq
            }

    def bm25_score(self, term_freq: int, doc_length: int, idf: float) -> float:
        """Calculate BM25 score for a term in a document."""
        return idf * ((term_freq * (self.k1 + 1)) / (term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))))

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)

        sum1 = sum(vec1[x]**2 for x in vec1.keys())
        sum2 = sum(vec2[x]**2 for x in vec2.keys())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        return numerator / denominator

    def search(self, query: str) -> List[Dict[str, str]]:
        """Search for the query and return ranked results."""
        query_terms = self.preprocess(query)
        query_vector = {term: self.idf.get(term, 0) for term in query_terms}

        results = []
        for filename, doc_vector in self.document_vectors.items():
            similarity = self.cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                snippet = self.get_snippet(self.documents[filename], query)
                results.append({
                    "filename": filename,
                    "snippet": snippet,
                    "similarity": similarity
                })

        # Sort results by similarity score in descending order
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:10]  # Return top 10 results

    def get_snippet(self, content: str, query: str, context_length: int = 50) -> str:
        """Extract a snippet of text around the query match."""
        query_terms = self.preprocess(query)
        content_lower = content.lower()
        
        for term in query_terms:
            match = re.search(r'\b' + re.escape(term) + r'\b', content_lower)
            if match:
                start = max(0, match.start() - context_length)
                end = min(len(content), match.end() + context_length)
                return f"...{content[start:end]}..."
        
        return content[:100] + "..."  # Return first 100 characters if no match found

# Initialize the retrieval system with BM25
dataset_directory = os.getenv('DATASET_DIR','E:\Assignment\AI&ML\TECH 400\Project\data')
bm25 = BM25Model(dataset_directory)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form['query']
        results = bm25.search(query)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
