# GreenHarvest Assistant (RAG chatbot)

A customer-support chatbot for [GreenHarvest](https://github.com/vishalimmanuel/project-mom), an Indian agricultural exports website. It answers questions about the 18 products in the catalog and is told never to invent prices, stock or terms.

## How it works (RAG: Retrieval-Augmented Generation)
```
customer question
      |
 1. RETRIEVE  -> find the 3 most relevant products (TF-IDF similarity)   [retrieval.py]
      |
 2. PROMPT    -> give the AI ONLY those products + strict rules          [assistant.py]
      |
 3. ANSWER    -> Claude replies using only the supplied catalog data
```
Giving the model only retrieved facts reduces made-up answers, which matters when prices are involved.

## Files
- `products.json`: the 18 products, extracted from the project-mom website's code
- `retrieval.py`: the search step (works without any API key)
- `assistant.py`: builds the prompt and calls Claude

## Run it
```bash
pip install -r requirements.txt
python retrieval.py      # test the search step
python assistant.py      # chat (dry-run mode without an API key)
```
**Without an API key** the assistant runs in dry-run mode: it shows the products it found and the exact prompt it would send.

**With an API key:** copy `.env.example` to `.env`, paste your key from the [Anthropic console](https://console.anthropic.com), and run again. `.env` is git-ignored, so the key is never uploaded.

## Status and honesty notes
- Retrieval and prompt building are tested. The **live model call has not been tested yet** because it needs an API key
- Search is keyword-based (TF-IDF), so "something for a healthy diet" works partly by word overlap and may miss the best match. Upgrading to embeddings is a good next step
- The catalog is sample data from the website, not live inventory

## Practice exercises (do these yourself to learn)
1. In `retrieval.py`, change `top_k=3` to `5` and see how the matches change
2. Add a new product to `products.json` and search for it
3. In `assistant.py`, edit `SYSTEM_RULES` so the assistant also replies in Tamil when the customer writes in Tamil
4. Write 10 test questions with expected answers and check how many the assistant gets right. This is called an evaluation, and it's what employers look for

## Skills shown
RAG pattern, TF-IDF retrieval, prompt design with guardrails, API integration, secrets handling (`.env`), dry-run mode for testing without cost
