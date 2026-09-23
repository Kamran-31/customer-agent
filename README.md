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
