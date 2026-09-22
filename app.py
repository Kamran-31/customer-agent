import json
import re
from datetime import datetime
from pathlib import Path

import faiss
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================
# ZYVRA CONFIGURATION
# ============================================================

APP_NAME = "Zyvra"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Gemini model
GEMINI_MODEL = "gemini/gemini-3.5-flash-lite"

# Project directories
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Knowledge base
HANDBOOK_PATH = DATA_DIR / "Zyvra_Customer_Support_Handbook.md"

# Order database
ORDERS_PATH = DATA_DIR / "Zyvra_orders_database.xlsx"

# Embedding files
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
EMBEDDING_CONFIG_PATH = EMBEDDINGS_DIR / "embedding_config.json"


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Zyvra — AI Customer Support",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(99,102,241,.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14,165,233,.10),
                transparent 25%
            ),
            #f8fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        border: 1px solid rgba(148,163,184,.22);
        background: rgba(255,255,255,.85);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 20px;
        box-shadow: 0 16px 50px rgba(15,23,42,.07);
    }

    .brand {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .tagline {
        color: #64748b;
        margin-top: 4px;
        font-size: 1rem;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857;
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .pending-card {
        border: 1px solid #e2e8f0;
        background: #ffffff;
        border-radius: 14px;
        padding: 12px;
        margin: 8px 0;
    }

    .small-muted {
        color: #64748b;
        font-size: .82rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 18px;
    }

    .stButton button {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_cases" not in st.session_state:
    st.session_state.pending_cases = []

if "case_counter" not in st.session_state:
    st.session_state.case_counter = 0


# ============================================================
# AUTOMATIC FAISS INDEX CREATION
# ============================================================

@st.cache_resource(show_spinner="Loading Zyvra AI knowledge base...")
def load_knowledge_store():

    # Make sure embeddings directory exists
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge chunks file not found:\n{CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise RuntimeError("chunks.json is empty.")

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    model = SentenceTransformer(EMBEDDING_MODEL)

    actual_dimension = model.get_sentence_embedding_dimension()

    if actual_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension mismatch. "
            f"Expected {EMBEDDING_DIMENSION}, "
            f"received {actual_dimension}."
        )

    # --------------------------------------------------------
    # Try loading existing FAISS index
    # --------------------------------------------------------

    index = None

    if INDEX_PATH.exists():

        try:

            existing_index = faiss.read_index(str(INDEX_PATH))

            if (
                existing_index.d == EMBEDDING_DIMENSION
                and existing_index.ntotal == len(chunks)
            ):
                index = existing_index

        except Exception:
            index = None

    # --------------------------------------------------------
    # BUILD INDEX AUTOMATICALLY IF MISSING / INVALID
    # --------------------------------------------------------

    if index is None:

        st.info(
            "Zyvra is preparing its knowledge-search index. "
            "This happens automatically on the first deployment."
        )

        texts = [
            item.get("text", "")
            for item in chunks
        ]

        if not any(texts):
            raise RuntimeError(
                "No text was found inside chunks.json."
            )

        # Generate 384-dimensional embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # FAISS Inner Product + normalized vectors
        # = cosine similarity
        new_index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSION
        )

        new_index.add(embeddings)

        # Write actual binary FAISS index
        faiss.write_index(
            new_index,
            str(INDEX_PATH)
        )

        index = new_index

    # --------------------------------------------------------
    # Save/update embedding configuration
    # --------------------------------------------------------

    config = {
        "embedding_model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "index_type": "IndexFlatIP",
        "similarity": "cosine_similarity",
        "normalized_embeddings": True,
        "total_vectors": int(index.ntotal),
        "created_or_loaded_at": datetime.now().isoformat(),
    }

    with open(
        EMBEDDING_CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    return model, index, chunks


# ============================================================
# ORDER DATABASE
# ============================================================

@st.cache_data
def load_orders():

    if not ORDERS_PATH.exists():
        raise FileNotFoundError(
            f"Order database not found:\n{ORDERS_PATH}"
        )

    df = pd.read_excel(
        ORDERS_PATH,
        sheet_name="Orders",
        dtype=str
    )

    return df.fillna("")


# ============================================================
# KNOWLEDGE SEARCH
# ============================================================

def search_knowledge(
    query: str,
    top_k: int = 4
) -> list[dict]:

    model, index, chunks = load_knowledge_store()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(
        top_k,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = dict(
            chunks[int(idx)]
        )

        item["similarity"] = round(
            float(score),
            4
        )

        results.append(item)

    return results


# ============================================================
# ORDER LOOKUP
# ============================================================

def lookup_order(
    order_id: str
) -> dict | None:

    order_id = (
        order_id
        .strip()
        .upper()
    )

    df = load_orders()

    matches = df[
        df["Order ID"]
        .str.upper()
        == order_id
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


# ============================================================
# CREWAI KNOWLEDGE TOOL
# ============================================================

class KnowledgeSearchInput(BaseModel):

    query: str = Field(
        ...,
        description=(
            "Customer question or support topic "
            "to search in the Zyvra knowledge base."
        ),
    )


class KnowledgeSearchTool(BaseTool):

    name: str = "search_zyvra_knowledge"

    description: str = (
        "Search the Zyvra customer-support knowledge base "
        "using semantic similarity. Use this for company "
        "policies, shipping, returns, warranty, payments, "
        "privacy, products, troubleshooting, and support procedures."
    )

    args_schema: type[BaseModel] = KnowledgeSearchInput

    def _run(
        self,
        query: str
    ) -> str:

        results = search_knowledge(
            query,
            top_k=4
        )

        if not results:
            return (
                "No relevant Zyvra knowledge-base information "
                "was found."
            )

        formatted = []

        for result in results:

            formatted.append(
                f"""
CHUNK ID: {result.get('chunk_id', 'N/A')}
SECTION: {result.get('section', 'N/A')}
SIMILARITY: {result.get('similarity', 'N/A')}

CONTENT:
{result.get('text', '')}
"""
            )

        return "\n".join(formatted)


# ============================================================
# CREWAI ORDER TOOL
# ============================================================

class OrderLookupInput(BaseModel):

    order_id: str = Field(
        ...,
        description=(
            "Zyvra order ID such as "
            "ORD-2026-1001."
        ),
    )


class OrderLookupTool(BaseTool):

    name: str = "lookup_order_database"

    description: str = (
        "Look up one Zyvra order from the simulated Excel "
        "order database. Use this when the customer provides "
        "an Order ID."
    )

    args_schema: type[BaseModel] = OrderLookupInput

    def _run(
        self,
        order_id: str
    ) -> str:

        order = lookup_order(
            order_id
        )

        if order is None:

            return (
                f"No order was found for "
                f"{order_id.strip().upper()}."
            )

        return json.dumps(
            order,
            ensure_ascii=False
        )


# ============================================================
# CREWAI SINGLE AGENT
# ============================================================

def build_agent() -> Agent:

    if "GEMINI_API_KEY" not in st.secrets:

        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it under Streamlit Cloud → "
            "Settings → Secrets."
        )

    llm = LLM(
        model=GEMINI_MODEL,
        api_key=st.secrets["GEMINI_API_KEY"],
    )

    return Agent(

        role="Zyvra Customer Support Specialist",

        goal=(
            "Resolve customer support questions accurately "
            "using the Zyvra knowledge base and order database. "
            "Never invent information. Escalate issues that "
            "require human intervention."
        ),

        backstory=(
            "You are Zyvra's first-line AI customer support "
            "specialist. You are friendly, concise, accurate, "
            "and careful with customer information. "
            "You use tools instead of guessing."
        ),

        llm=llm,

        tools=[
            KnowledgeSearchTool(),
            OrderLookupTool(),
        ],

        allow_delegation=False,

        verbose=False,
    )


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def get_conversation_context() -> str:

    recent_messages = (
        st.session_state.messages[-12:]
    )

    if not recent_messages:
        return "(No previous conversation.)"

    context = []

    for message in recent_messages:

        role = (
            "Customer"
            if message["role"] == "user"
            else "Zyvra"
        )

        context.append(
            f"{role}: {message['content']}"
        )

    return "\n".join(context)


# ============================================================
# HUMAN REQUEST DETECTION
# ============================================================

def user_explicitly_requests_human(
    text: str
) -> bool:

    patterns = [

        r"\bhuman\b",
        r"\breal person\b",
        r"\bhuman agent\b",
        r"\blive agent\b",
        r"\bcustomer service agent\b",
        r"\brepresentative\b",
        r"\bspeak to someone\b",
        r"\btalk to someone\b",
        r"\btalk to a person\b",
        r"\bspeak to a person\b",
        r"\bmanager\b",
        r"\bsupervisor\b",

    ]

    lowered = text.lower()

    return any(
        re.search(
            pattern,
            lowered
        )
        for pattern in patterns
    )


# ============================================================
# PARSE AGENT JSON
# ============================================================

def parse_agent_json(
    raw: str
) -> dict:

    raw = raw.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.I
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    try:

        data = json.loads(raw)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.S
    )

    if match:

        try:

            data = json.loads(
                match.group(0)
            )

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

    return {
        "response": raw,
        "escalate": False,
        "escalation_summary": "",
    }


# ============================================================
# RUN SUPPORT AGENT
# ============================================================

def run_support_agent(
    user_message: str
) -> dict:

    agent = build_agent()

    task = Task(

        description=f"""

You are responding to the latest customer message
for Zyvra.

LATEST CUSTOMER MESSAGE:
{user_message}

CONVERSATION CONTEXT:
{get_conversation_context()}

IMPORTANT INSTRUCTIONS:

1. Answer the customer's actual question.

2. Use search_zyvra_knowledge for:
   - company information
   - shipping
   - returns
   - warranty
   - payments
   - products
   - troubleshooting
   - policies
   - support procedures

3. Use lookup_order_database for order-specific questions.

4. Never invent order information.

5. If the customer asks about an order but has not
   provided an Order ID, ask for the Order ID.

6. Remember previous messages in the conversation.

7. If the available information is insufficient
   to reliably resolve the issue, escalate it.

8. Escalate cases involving:
   - unresolved complaints
   - sensitive account/security matters
   - warranty decisions requiring inspection
   - return/refund decisions requiring approval
   - disputes
   - cases where available information is insufficient

9. Never ask the customer for:
   - password
   - OTP
   - CVV
   - full card number
   - API key
   - other authentication secrets

10. Be friendly, professional and concise.

11. Return JSON ONLY.

Required JSON format:

{{
    "response": "Customer-facing response",
    "escalate": false,
    "escalation_summary": ""
}}

If escalation is required:

{{
    "response": "Customer-facing response",
    "escalate": true,
    "escalation_summary": "Short summary for human support"
}}

""",

        expected_output=(
            "Valid JSON containing response, "
            "escalate and escalation_summary."
        ),

        agent=agent,
    )

    crew = Crew(

        agents=[agent],

        tasks=[task],

        process=Process.sequential,

        verbose=False,
    )

    result = crew.kickoff()

    raw = getattr(
        result,
        "raw",
        str(result)
    )

    return parse_agent_json(raw)


# ============================================================
# HUMAN ESCALATION
# ============================================================

def create_pending_case(
    user_message: str,
    summary: str
) -> dict:

    st.session_state.case_counter += 1

    case_id = (
        f"CASE-"
        f"{datetime.now().strftime('%Y%m%d')}-"
        f"{st.session_state.case_counter:03d}"
    )

    order_match = re.search(
        r"\bORD-2026-\d{4}\b",
        user_message.upper()
    )

    order_id = (
        order_match.group(0)
        if order_match
        else "Not provided"
    )

    case = {

        "case_id": case_id,

        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M"
        ),

        "order_id": order_id,

        "summary": (
            summary.strip()
            or user_message.strip()
        ),

        "status": "Pending Human Support",
    }

    st.session_state.pending_cases.insert(
        0,
        case
    )

    return case


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ✦ Zyvra")

    st.caption(
        "AI Customer Support"
    )

    st.markdown("### Pending Human Support")

    if st.session_state.pending_cases:

        for case in st.session_state.pending_cases:

            st.markdown(
                f"""
                <div class="pending-card">

                    <b>{case['case_id']}</b>

                    <br>

                    <span class="small-muted">
                        {case['created_at']}
                        ·
                        {case['order_id']}
                    </span>

                    <div style="margin-top:6px;">
                        {case['summary']}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.caption(
            "No pending human-support cases."
        )

    st.divider()

    if st.button(
        "Start New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.caption(
        "Zyvra can answer support questions, "
        "check orders and escalate unresolved issues."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="status-pill">
            ● Online · AI Support
        </div>

        <h1 class="brand">
            Zyvra
        </h1>

        <div class="tagline">
            Fast answers. Clear support.
            Human escalation when needed.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:55px 10px 35px;
        ">

            <h2>
                How can we help?
            </h2>

            <p style="color:#64748b;">
                Ask about an order, shipping,
                returns, warranty, payments,
                products or technical support.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask Zyvra about your order or support..."
)


if prompt:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner(
            "Zyvra is checking the right information..."
        ):

            # ------------------------------------------------
            # Explicit human request
            # ------------------------------------------------

            if user_explicitly_requests_human(
                prompt
            ):

                case = create_pending_case(

                    prompt,

                    "Customer explicitly requested "
                    "human support."
                )

                answer = (
                    "Your request has been **escalated "
                    "to human support**.\n\n"
                    f"Your case **{case['case_id']}** "
                    "has been added to the pending queue "
                    "for human follow-up."
                )

            else:

                try:

                    result = run_support_agent(
                        prompt
                    )

                    answer = str(
                        result.get(
                            "response",
                            ""
                        )
                    ).strip()

                    escalate = bool(
                        result.get(
                            "escalate",
                            False
                        )
                    )

                    summary = str(
                        result.get(
                            "escalation_summary",
                            ""
                        )
                    ).strip()

                    # Safety net
                    unresolved_phrases = [

                        "i don't know",
                        "i do not know",
                        "cannot determine",
                        "can't determine",
                        "unable to determine",
                        "i'm unable to",
                        "i am unable to",
                        "not enough information",
                        "insufficient information",

                    ]

                    if any(
                        phrase in answer.lower()
                        for phrase in unresolved_phrases
                    ):

                        escalate = True

                        if not summary:

                            summary = (
                                "The AI could not "
                                "reliably resolve "
                                "the customer's request."
                            )

                    # ------------------------------------------------
                    # HUMAN ESCALATION
                    # ------------------------------------------------

                    if escalate:

                        case = create_pending_case(

                            prompt,

                            summary or answer
                        )

                        answer = (
                            f"{answer}\n\n"
                            "---\n\n"
                            f"**Escalated to Human Support**\n\n"
                            f"Case ID: **{case['case_id']}**\n\n"
                            "Your request has been added "
                            "to the pending human-support queue."
                        )

                except Exception:

                    case = create_pending_case(

                        prompt,

                        "AI support workflow encountered "
                        "an internal issue and requires "
                        "human review."
                    )

                    answer = (
                        "I’m sorry, but I couldn't reliably "
                        "complete that request.\n\n"
                        "Your issue has been **escalated "
                        "to human support**.\n\n"
                        f"Case ID: **{case['case_id']}**"
                    )

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
