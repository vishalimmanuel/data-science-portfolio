"""
Part 2: THE ASSISTANT
---------------------
Flow for every customer question:
  1. RETRIEVE  - find the 3 most relevant products (retrieval.py)
  2. BUILD A PROMPT - give the AI only those products, and strict rules
  3. ASK CLAUDE - get a friendly answer based ONLY on that catalog data

Without an API key the script still works in "dry-run" mode: it shows what it found and
the exact prompt it WOULD send. With a key (in a .env file) it gets a real answer.

Run:   python assistant.py
"""

import os
from pathlib import Path

from retrieval import ProductSearch

MODEL = "claude-sonnet-5-5"

SYSTEM_RULES = """You are the sales assistant for GreenHarvest, an exporter of premium Indian agricultural products.
Rules:
- Answer ONLY using the product information provided below. Never invent prices, stock, certifications or shipping terms.
- If the information needed is not provided, say you don't have it and suggest the customer use the contact form.
- Mention price, minimum order quantity and origin when relevant.
- Keep answers short, friendly and professional."""


def load_env():
    """Read API key from a local .env file (never committed to GitHub)."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def format_context(matches):
    """Turn the retrieved products into plain text for the prompt."""
    if not matches:
        return "No matching products found."
    lines = []
    for product, _score in matches:
        lines.append(
            f"- {product['name']}: {product['description']} "
            f"Price: {product['price']} per {product.get('unit', 'unit')}. "
            f"Minimum order: {product.get('minOrderQuantity', 'not stated')} {product.get('unit', '')}. "
            f"Origin: {product.get('origin', 'India')}."
        )
    return "\n".join(lines)


def answer(question, search, client=None):
    matches = search.search(question, top_k=3)
    context = format_context(matches)
    system = f"{SYSTEM_RULES}\n\nPRODUCT INFORMATION:\n{context}"

    if client is None:  # dry-run mode
        return f"[DRY RUN - no API key found]\n\nRetrieved products:\n{context}\n\nPrompt that would be sent:\n{system}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


if __name__ == "__main__":
    load_env()
    search = ProductSearch()

    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    else:
        print("No ANTHROPIC_API_KEY found: running in dry-run mode (see README to add a key).\n")

    print("GreenHarvest assistant. Type a question, or 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit", ""}:
            break
        print("\nAssistant:", answer(question, search, client))
