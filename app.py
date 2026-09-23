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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

GEMINI_MODEL = "gemini/gemini-3.5-flash-lite"

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

HANDBOOK_PATH = DATA_DIR / "Zyvra_Customer_Support_Handbook.md"
ORDERS_PATH = DATA_DIR / "Zyvra_orders_database.xlsx"

CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
EMBEDDING_CONFIG_PATH = EMBEDDINGS_DIR / "embedding_config.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Zyvra — AI Customer Support",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STREAMLIT NATIVE STYLING
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .zyvra-header {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 26px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
    }

    .online-status {
        display: inline-block;
        background: #ecfdf5;
        color: #047857;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .zyvra-logo {
        font-size: 38px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }

    .zyvra-tagline {
        color: #64748b;
        font-size: 16px;
        margin-top: 6px;
    }

    .welcome-box {
        text-align: center;
        padding: 45px 20px 30px;
    }

    .welcome-title {
        font-size: 30px;
        font-weight: 700;
        color: #0f172a;
    }

    .welcome-text {
        color: #64748b;
        font-size: 16px;
        line-height: 1.6;
    }

    .pending-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 9px;
    }

    .case-id {
        font-weight: 700;
        color: #0f172a;
    }

    .case-meta {
        font-size: 12px;
        color: #64748b;
    }

    .case-summary {
        font-size: 13px;
        color: #334155;
        margin-top: 6px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
    }

    .stButton > button {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_cases" not in st.session_state:
    st.session_state.pending_cases = []

if "case_counter" not in st.session_state:
    st.session_state.case_counter = 0


# ============================================================
# KNOWLEDGE BASE + FAISS
# ============================================================

@st.cache_resource(show_spinner="Preparing Zyvra's knowledge base...")
def load_knowledge_store():

    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json was not found at:\n{CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    if not chunks:
        raise RuntimeError(
            "chunks.json is empty."
        )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    dimension = model.get_sentence_embedding_dimension()

    if dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension mismatch. "
            f"Expected {EMBEDDING_DIMENSION}, "
            f"got {dimension}."
        )

    index = None

    # --------------------------------------------------------
    # Load existing index
    # --------------------------------------------------------

    if INDEX_PATH.exists():

        try:

            existing_index = faiss.read_index(
                str(INDEX_PATH)
            )

            if (
                existing_index.d
                == EMBEDDING_DIMENSION
                and existing_index.ntotal
                == len(chunks)
            ):

                index = existing_index

        except Exception:
            index = None

    # --------------------------------------------------------
    # Automatically create FAISS index
    # --------------------------------------------------------

    if index is None:

        texts = [
            item.get("text", "")
            for item in chunks
        ]

        if not any(texts):
            raise RuntimeError(
                "No text was found in chunks.json."
            )

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSION
        )

        index.add(
            embeddings
        )

        faiss.write_index(
            index,
            str(INDEX_PATH)
        )

        config = {
            "embedding_model": EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
            "index_type": "IndexFlatIP",
            "similarity": "cosine_similarity",
            "normalized_embeddings": True,
            "total_vectors": int(index.ntotal),
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
            f"Order database was not found at:\n{ORDERS_PATH}"
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
):

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
):

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
# KNOWLEDGE SEARCH TOOL
# ============================================================

class KnowledgeSearchInput(BaseModel):

    query: str = Field(
        ...,
        description=(
            "Customer question or support topic "
            "to search in the Zyvra knowledge base."
        )
    )


class KnowledgeSearchTool(BaseTool):

    name: str = "search_zyvra_knowledge"

    description: str = (
        "Search the Zyvra company knowledge base. "
        "Use for policies, shipping, returns, warranty, "
        "payments, products, troubleshooting and support."
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
            return "No relevant knowledge was found."

        output = []

        for result in results:

            output.append(
                f"""
CHUNK ID: {result.get("chunk_id", "N/A")}
SIMILARITY: {result.get("similarity", "N/A")}

CONTENT:
{result.get("text", "")}
"""
            )

        return "\n".join(output)


# ============================================================
# ORDER LOOKUP TOOL
# ============================================================

class OrderLookupInput(BaseModel):

    order_id: str = Field(
        ...,
        description="Zyvra order ID such as ORD-2026-1001."
    )


class OrderLookupTool(BaseTool):

    name: str = "lookup_order_database"

    description: str = (
        "Look up an order from Zyvra's simulated Excel "
        "order database."
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
# SINGLE CREWAI AGENT
# ============================================================

def build_agent():

    if "GEMINI_API_KEY" not in st.secrets:

        raise RuntimeError(
            "GEMINI_API_KEY is missing from "
            "Streamlit Secrets."
        )

    llm = LLM(
        model=GEMINI_MODEL,
        api_key=st.secrets["GEMINI_API_KEY"],
    )

    return Agent(

        role="Zyvra Customer Support Specialist",

        goal=(
            "Resolve customer support questions accurately "
            "using Zyvra's knowledge base and order database. "
            "Never invent information and escalate when "
            "human intervention is required."
        ),

        backstory=(
            "You are Zyvra's first-line AI customer support "
            "specialist. You are friendly, professional, "
            "accurate and concise. You always use the available "
            "tools instead of guessing."
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
# CONVERSATION CONTEXT
# ============================================================

def get_conversation_context():

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

def user_requests_human(
    text: str
):

    patterns = [

        r"\bhuman\b",
        r"\breal person\b",
        r"\bhuman agent\b",
        r"\blive agent\b",
        r"\brepresentative\b",
        r"\bspeak to someone\b",
        r"\btalk to someone\b",
        r"\btalk to a person\b",
        r"\bspeak to a person\b",
        r"\bmanager\b",
        r"\bsupervisor\b",

    ]

    text = text.lower()

    return any(
        re.search(
            pattern,
            text
        )
        for pattern in patterns
    )


# ============================================================
# PARSE AGENT RESPONSE
# ============================================================

def parse_agent_json(
    raw: str
):

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

        result = json.loads(raw)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.S
    )

    if match:

        try:

            result = json.loads(
                match.group(0)
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    return {
        "response": raw,
        "escalate": False,
        "escalation_summary": "",
    }


# ============================================================
# RUN AGENT
# ============================================================

def run_support_agent(
    user_message: str
):

    agent = build_agent()

    task = Task(

        description=f"""
Respond to the latest Zyvra customer message.

LATEST CUSTOMER MESSAGE:
{user_message}

PREVIOUS CONVERSATION:
{get_conversation_context()}

RULES:

1. Answer the customer's actual question.

2. Use the knowledge-base tool for company information,
   policies, shipping, returns, warranty, payments,
   products and troubleshooting.

3. Use the order database tool for order-specific questions.

4. Never invent order information.

5. If an order question does not contain an Order ID,
   ask the customer for their Order ID.

6. Maintain conversation context.

7. If the issue cannot be reliably resolved,
   escalate it to human support.

8. Escalate disputes, unresolved complaints,
   sensitive account issues, approval-dependent
   refunds/returns, warranty decisions requiring
   inspection, or insufficient-information cases.

9. Never request passwords, OTPs, CVVs,
   full card numbers or other authentication secrets.

10. Keep the response professional and concise.

Return JSON ONLY:

{{
    "response": "customer-facing response",
    "escalate": false,
    "escalation_summary": ""
}}

For escalation:

{{
    "response": "customer-facing response",
    "escalate": true,
    "escalation_summary": "short summary for human support"
}}
""",

        expected_output=(
            "Valid JSON with response, escalate "
            "and escalation_summary."
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
# CREATE HUMAN SUPPORT CASE
# ============================================================

def create_pending_case(
    user_message: str,
    summary: str
):

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
            if summary.strip()
            else user_message.strip()
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

    st.divider()

    st.markdown(
        "### Pending Human Support"
    )

    if st.session_state.pending_cases:

        for case in st.session_state.pending_cases:

            st.markdown(
                f"""
                <div class="pending-card">

                    <div class="case-id">
                        {case["case_id"]}
                    </div>

                    <div class="case-meta">
                        {case["created_at"]}
                        ·
                        {case["order_id"]}
                    </div>

                    <div class="case-summary">
                        {case["summary"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.caption(
            "No pending human-support cases."
        )

    st.divider()

    if st.button(
        "＋ Start New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "Zyvra can answer support questions, "
        "check orders and escalate unresolved "
        "issues to human support."
    )


# ============================================================
# MAIN ZYVRA HEADER
# ============================================================

st.markdown(
    """
    <div class="zyvra-header">

        <div class="online-status">
            ● Online · AI Support
        </div>

        <div class="zyvra-logo">
            Zyvra
        </div>

        <div class="zyvra-tagline">
            Fast answers. Clear support.
            Human escalation when needed.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-box">

            <div class="welcome-title">
                How can we help?
            </div>

            <div class="welcome-text">
                Ask about an order, shipping, returns,
                warranty, payments, products or
                technical support.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Native Streamlit suggestion buttons
    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "📦 Check my order",
            use_container_width=True
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "I want to check my order."
                }
            )

            st.rerun()

    with col2:

        if st.button(
            "🚚 Shipping information",
            use_container_width=True
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "What are your shipping policies?"
                }
            )

            st.rerun()

    with col3:

        if st.button(
            "↩️ Returns & warranty",
            use_container_width=True
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "What is your return and warranty policy?"
                }
            )

            st.rerun()


# ============================================================
# CHAT HISTORY
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
            # Direct human request
            # ------------------------------------------------

            if user_requests_human(prompt):

                case = create_pending_case(

                    prompt,

                    "Customer explicitly requested "
                    "human support."
                )

                answer = (
                    "Your request has been **escalated "
                    "to human support**.\n\n"
                    f"Your case ID is **{case['case_id']}**.\n\n"
                    "The request has been added to the "
                    "pending human-support queue."
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
                                "The AI could not reliably "
                                "resolve the customer's request."
                            )

                    if escalate:

                        case = create_pending_case(

                            prompt,

                            summary or answer
                        )

                        answer = (
                            f"{answer}\n\n"
                            "---\n\n"
                            "**Escalated to Human Support**\n\n"
                            f"Case ID: **{case['case_id']}**\n\n"
                            "Your request has been added "
                            "to the pending human-support queue."
                        )

                except Exception as error:

                    case = create_pending_case(

                        prompt,

                        "The AI support workflow encountered "
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

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
