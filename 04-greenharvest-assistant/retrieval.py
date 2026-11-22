"""
Part 1: RETRIEVAL
-----------------
A chatbot should not guess. Before asking the AI model anything, we first FIND the
products that match the customer's question, then give only those to the model.
This idea is called RAG (Retrieval-Augmented Generation).

How the search works (TF-IDF, a classic technique):
  - Turn every product into a vector of word weights.
  - Turn the question into a vector the same way.
  - Products whose vectors point in a similar direction to the question are the best matches.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PRODUCTS_FILE = Path(__file__).parent / "products.json"


def load_products():
    """Read the product catalog from products.json."""
    return json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))


def product_to_text(product):
    """Combine the useful fields of a product into one searchable sentence."""
    return (
        f"{product['name']}. {product['description']} "
        f"Category: {product['category']}. Origin: {product.get('origin', 'India')}."
    )


class ProductSearch:
    def __init__(self):
        self.products = load_products()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([product_to_text(p) for p in self.products])

    def search(self, question, top_k=3):
        """Return the top_k products most similar to the question, with a score."""
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        best = scores.argsort()[::-1][:top_k]
        return [(self.products[i], float(scores[i])) for i in best if scores[i] > 0]


if __name__ == "__main__":
    search = ProductSearch()
    for question in [
        "Do you have cardamom from Kerala?",
        "I need millets for a healthy diet",
        "Which rice do you export?",
    ]:
        print(f"\nQ: {question}")
        for product, score in search.search(question):
            print(f"   {score:.2f}  {product['name']}  ({product['price']}, min {product.get('minOrderQuantity', '?')} {product.get('unit', '')})")
