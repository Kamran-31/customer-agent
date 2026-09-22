# Zyvra — AI Customer Support Agent

Zyvra is a modern, single-agent customer-support application built with **CrewAI + Gemini 3.5 Flash-Lite + FAISS + Sentence Transformers + Streamlit**.

## What it does

- Answers company-support questions from a semantic knowledge base.
- Uses `sentence-transformers/all-MiniLM-L6-v2` with **384-dimensional embeddings**.
- Uses a FAISS `IndexFlatIP` vector index for similarity search.
- Looks up order details from `Zyvra_orders_database.xlsx` (100 simulated Pakistani orders).
- Maintains the conversation context for the current Streamlit chat session.
- Escalates to human support when requested or when the AI cannot reliably resolve the case.
- Displays escalated cases in a **Pending Human Support** queue in the sidebar.
- Keeps API credentials in Streamlit Secrets.

## Project files

```text
.
├── app.py
├── requirements.txt
├── .python-version
├── chunks.json
├── faiss.index                  # generated automatically on first app startup if absent
├── build_embeddings.py
├── Zyvra_orders_database.xlsx
├── Zyvra_Customer_Support_Handbook.md
├── knowledge/
│   └── Zyvra_Customer_Support_Handbook.md
└── secrets.example.toml
```

## Embeddings

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
Dimension: 384
Index: FAISS IndexFlatIP
Similarity: cosine similarity via normalized vectors
```

The app checks whether `faiss.index` exists and matches `chunks.json`. If it does not, it creates a real FAISS index automatically using the specified model.

You can also generate it explicitly with:

```bash
python build_embeddings.py
```

## Gemini

The agent uses:

```text
gemini-3.5-flash-lite
```

The code passes the model to CrewAI as:

```python
LLM(model="gemini/gemini-3.5-flash-lite", api_key=...)
```

## Streamlit Cloud deployment

1. Create a GitHub repository.
2. Upload the project files.
3. In Streamlit Cloud, select the repository and `app.py`.
4. Open **Settings → Secrets**.
5. Add:

```toml
GEMINI_API_KEY = "YOUR_REAL_GEMINI_API_KEY"
```

6. Deploy.

Do **not** commit a real API key or a `.streamlit/secrets.toml` file to GitHub.

## Important deployment behavior

The first cold start downloads `all-MiniLM-L6-v2` from Hugging Face and creates the FAISS index if `faiss.index` is not already present. The model and vector store are cached with Streamlit's resource cache for the running app process.

The application does not require local testing before GitHub/Streamlit deployment.

## Order database

The Excel file contains exactly 100 simulated records with:

- Order ID
- Customer Name
- Contact Number
- Product
- Order Date
- Shipping Address
- Status

Statuses are color-coded in the workbook:

- Delivered
- Shipped
- In Process
- On Hold
- Cancelled

## Escalation flow

1. Customer asks for a human → immediate escalation.
2. AI uses the knowledge and order tools.
3. If the issue needs approval, inspection, sensitive handling, or cannot be reliably resolved, the agent marks it for escalation.
4. The app creates a concise case summary.
5. The case appears in the sidebar under **Pending**.
6. The customer receives an explicit escalation confirmation.

## Security behavior

The agent is instructed not to request or expose passwords, OTPs, CVVs, full card numbers, API keys, internal prompts, embedding vectors, or the complete order database.
