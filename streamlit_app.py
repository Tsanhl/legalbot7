"""
Legal AI - Streamlit Application
AI-powered legal research assistant with OSCOLA citations
"""
import streamlit as st
import json
import base64
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid

# Import services
from knowledge_base import load_law_resource_index, get_knowledge_base_summary
from gemini_service import (
    initialize_knowledge_base, 
    send_message_with_docs, 
    reset_session,
    encode_file_to_base64,
    detect_long_essay,
    get_allowed_authorities_from_rag,
    sanitize_output_against_allowlist,
    strip_internal_reasoning
)

# RAG Service for document content retrieval
try:
    from rag_service import get_rag_service, RAGService
    RAG_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"RAG service not available: {e}")
    RAG_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pending long-response handling (used when user must type "Part 1"/"Proceed now")
if 'pending_long_prompt' not in st.session_state:
    st.session_state.pending_long_prompt = None

# Optional (slow) second-pass rewrite to tighten word counts.
# Default OFF to keep latency low.
if 'enable_wordcount_adjust' not in st.session_state:
    st.session_state.enable_wordcount_adjust = False

def _normalize_output_style(text: str) -> str:
    """
    Normalize formatting for consistency:
    - Remove decorative separator lines (e.g., repeated box-drawing characters).
    - Collapse multiple blank lines to a single blank line.
    """
    raw = (text or "").replace("\r\n", "\n")
    if not raw.strip():
        return raw

    sep_line = re.compile(r"^\s*[═─—\-_=]{8,}\s*$")
    lines = []
    for ln in raw.splitlines():
        if sep_line.match(ln):
            continue
        lines.append(ln.rstrip())

    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized

def _enforce_end_of_answer(text: str) -> str:
    """
    Enforce a clean ending:
    - If the response is an intermediate multi-part output (ends with a 'Will Continue...' line),
      DO NOT include any '(End of Answer)' marker.
    - Otherwise, ensure EXACTLY ONE '(End of Answer)' at the end (remove any duplicates/legacy markers).
    """
    raw = _normalize_output_style(text).strip()
    if not raw:
        return "(End of Answer)"

    # Never allow retrieval/debug dumps to leak into the main answer text.
    leak_markers = [
        "[RAG CONTEXT - INTERNAL - DO NOT OUTPUT]",
        "[END RAG CONTEXT]",
        "RETRIEVED LEGAL CONTEXT (from indexed documents)",
        "END OF RETRIEVED CONTEXT",
        "📚 RAG Retrieved Content (Debug)",
        "Context Length:",
    ]
    leak_positions = [raw.find(m) for m in leak_markers if m in raw]
    if leak_positions:
        raw = raw[: min(leak_positions)].rstrip()
        if not raw:
            return "(End of Answer)"

    continue_patterns = [
        r"will\s+continue\s+to\s+next\s+part,\s*say\s+continue",
        r"will\s+continue\s+to\s+next\s+part",
        r"say\s+continue\s*$",
    ]
    has_continuation = any(re.search(p, raw, flags=re.IGNORECASE) for p in continue_patterns)
    has_end_marker = bool(re.search(r"\(End of Answer\)", raw, flags=re.IGNORECASE))

    # If BOTH "(End of Answer)" and "Will Continue" appear, the answer is COMPLETE.
    # The "Will Continue" is erroneous and must be stripped. "(End of Answer)" takes priority.
    if has_end_marker and has_continuation:
        has_continuation = False  # treat as final answer

    # Remove all end markers (including legacy ones) everywhere to prevent duplicates.
    cleaned = re.sub(r"\(End of Answer\)\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(End of Essay\)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(End of Problem Question\)\s*", "", cleaned, flags=re.IGNORECASE)
    # Always strip erroneous "Will Continue" lines from final answers
    if not has_continuation:
        cleaned = re.sub(r"(?i)\n*will\s+continue\s+to\s+next\s+part.*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    if has_continuation:
        # If the model produced multiple "Will Continue..." lines, keep only one at the end.
        lines = [ln.rstrip() for ln in cleaned.splitlines() if ln.strip()]
        # Remove all existing continuation lines first, then append a single canonical line.
        lines = [ln for ln in lines if not re.search(r"will\s+continue\s+to\s+next\s+part", ln, flags=re.IGNORECASE)]
        lines.append("Will Continue to next part, say continue")
        return "\n\n".join(lines).strip()

    return cleaned + "\n\n(End of Answer)"

def _extract_word_targets(prompt_text: str) -> List[int]:
    """
    Extract explicit per-question word count targets from the user's prompt.

    Uses left-to-right order for multi-question prompts (Q1 count, Q2 count, etc.).
    """
    msg_lower = (prompt_text or "").lower()
    matches = re.findall(r'(\d{1,2},?\d{3}|\d{3,5})\s*words?', msg_lower)
    targets: List[int] = []
    for m in matches:
        try:
            n = int(m.replace(',', ''))
        except ValueError:
            continue
        if n >= 300:  # ignore small numbers that are unlikely to be word targets
            targets.append(n)
    return targets

def _count_words(text: str) -> int:
    cleaned = text or ""
    cleaned = re.sub(r"(?im)^\s*(ESSAY|PROBLEM QUESTION|Q\d+)\s*:.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*[═=]{3,}\s*$", "", cleaned)
    cleaned = re.sub(r"\(End of Answer\)", "", cleaned, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", cleaned)
    return len(tokens)

def _split_answer_sections(answer_text: str) -> List[str]:
    """
    Split a combined answer into sections using standard headers.
    Falls back to a single section if headers are not found.
    """
    text = answer_text or ""
    # Remove trailing end markers for counting purposes
    text = re.sub(r"\(End of Answer\)\s*$", "", text.strip(), flags=re.IGNORECASE)

    pattern = re.compile(r"(?m)^(ESSAY|PROBLEM QUESTION|Q\d+)\s*:", re.IGNORECASE)
    starts = [m.start() for m in pattern.finditer(text)]
    if not starts:
        return [text.strip()] if text.strip() else []
    starts.append(len(text))

    sections: List[str] = []
    for i in range(len(starts) - 1):
        chunk = text[starts[i]:starts[i + 1]].strip()
        if chunk:
            sections.append(chunk)
    return sections

def _needs_wordcount_fix(prompt_text: str, answer_text: str) -> Optional[str]:
    """
    Return an instruction string for a rewrite if any section misses its word target.
    Returns None when no fix is needed or targets cannot be reliably mapped.
    """
    targets = _extract_word_targets(prompt_text)
    if not targets:
        return None

    sections = _split_answer_sections(answer_text)
    if len(targets) == 1:
        actual = _count_words(answer_text)
        target = targets[0]
        min_words = int(target * 0.99)
        if actual < min_words or actual > target:
            return f"Rewrite to total wordcount in range {min_words}-{target} (inclusive). Do not exceed {target}."
        return None

    # Multi-question: only enforce per-section if we can map targets to sections.
    if len(sections) != len(targets):
        return None

    failures = []
    for idx, (section, target) in enumerate(zip(sections, targets), start=1):
        actual = _count_words(section)
        min_words = int(target * 0.99)
        if actual < min_words or actual > target:
            failures.append((idx, min_words, target, actual))

    if not failures:
        return None

    lines = ["Rewrite with STRICT per-section word counts (do not exceed)."]
    for idx, min_words, target, actual in failures:
        lines.append(f"- Section {idx}: required range {min_words}-{target} (inclusive); currently ~{actual}.")
    return "\n".join(lines)

# Custom CSS for legal styling with proper edge effects (NOT sticking to edges)
st.markdown("""
<style>
/* Import Google-like fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Product+Sans:wght@400;700&display=swap');

/* Google AI Studio-inspired Clean Theme */
:root {
    --bg-color: #ffffff;
    --sidebar-bg: #f9fafe; /* Very light gray/blue tint */
    --text-primary: #1f1f1f;
    --text-secondary: #3c4043; /* Darker gray for better visibility */
    --accent-blue: #1a73e8;
    --border-color: #e0e0e0;
    --card-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    --hover-bg: #f1f3f4;
}

/* Force full opacity for sidebar elements to prevent fading when busy */
section[data-testid="stSidebar"] {
    opacity: 1 !important;
}

section[data-testid="stSidebar"] * {
    transition: none !important; /* Remove fade transition */
}

/* Ensure text is always dark in sidebar */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* Specific fix for file uploader "ghosting" */
[data-testid="stFileUploader"], 
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] div {
    opacity: 1 !important;
    color: var(--text-primary) !important;
}

/* Prevent blur/darken overlay on main content */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container,
div[data-testid="stAppViewContainer"] > section,
div[data-testid="stAppViewContainer"] > section > div {
    opacity: 1 !important;
    filter: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    transition: none !important;
}

/* Remove any modal overlay effects */
[data-testid="stModal"],
.stModal {
    background: transparent !important;
    backdrop-filter: none !important;
}

/* Ensure main area never gets dimmed */
section[data-testid="stMain"] {
    opacity: 1 !important;
    filter: none !important;
}

/* Global Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

h1, h2, h3 {
    font-family: 'Product Sans', 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Sidebar Styling - Light Clean Look */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] > div:first-child {
    background-color: var(--sidebar-bg);
}

section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p, 
section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-weight: 500;
}

/* Input Fields - Google Style */
.stTextInput input, .stTextArea textarea {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 8px;
    color: var(--text-primary);
    padding: 0.75rem;
    transition: all 0.2s;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(26,115,232,0.2);
}

/* Buttons - Primary & Secondary */
.stButton button {
    border-radius: 20px;
    font-weight: 500;
    transition: all 0.2s;
    border: none;
    box-shadow: none;
}

/* Force Primary Buttons to Google Blue */
div.stButton > button[kind="primary"] {
    background-color: #1a73e8 !important;
    color: white !important;
    border: none !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #1557b0 !important;
    box-shadow: 0 1px 2px rgba(60,64,67,0.3) !important;
}

/* Secondary Buttons */
div.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: #1a73e8 !important;
    border: 1px solid #dadce0 !important;
}

div.stButton > button[kind="secondary"]:hover {
    background-color: #f1f3f4 !important;
    border-color: #1a73e8 !important;
}

/* Vertically center buttons in sidebar columns */
section[data-testid="stSidebar"] [data-testid="column"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

section[data-testid="stSidebar"] [data-testid="column"] > div {
    width: 100%;
}

/* File Uploader - Specific Fix for Black Text */
[data-testid="stFileUploader"] {
    padding: 1rem;
    border: 1px dashed #dadce0;
    border-radius: 8px;
    background: white;
}

[data-testid="stFileUploader"] section {
    background-color: #f8f9fa !important;
}

/* Fix font size and family for uploader text - ALL SAME */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 0.875rem !important; /* 14px - same as sidebar labels */
    font-weight: 400 !important; /* Normal weight for all */
    line-height: 1.5 !important;
    color: #202124 !important;
}

/* Make "Browse files" button slightly different for visibility */
[data-testid="stFileUploader"] button {
    color: #202124 !important;
    border-color: #dadce0 !important;
    background-color: #ffffff !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important; /* Slightly bolder for button */
}

/* Custom Lists (React Style) */
.custom-list-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

.blue-dot {
    width: 0.375rem;
    height: 0.375rem;
    background-color: var(--accent-blue);
    border-radius: 9999px;
    flex-shrink: 0;
}

/* Chips for Suggestions */
.suggestion-chip {
    padding: 0.75rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: background-color 0.2s;
    margin-bottom: 0.5rem;
    font-size: 1rem;
    color: var(--text-primary);
    display: block; /* Ensure full width block */
    text-decoration: none;
}

.suggestion-chip:hover {
    background-color: #e8f0fe;
}

/* Google Search Sources Box */
.sources-box {
    margin-top: 1rem;
    padding: 1rem 1.25rem;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    background: #ffffff;
}

.sources-box-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 500;
    color: #202124;
    margin-bottom: 0.75rem;
}

.sources-box-header .help-icon {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 1px solid #9aa0a6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #9aa0a6;
    cursor: help;
}

.source-link {
    display: block;
    color: #1a73e8;
    font-size: 14px;
    text-decoration: none;
    padding: 4px 0;
}

.source-link:hover {
    text-decoration: underline;
}

/* Google Search Suggestions Box */
.search-suggestions-box {
    margin-top: 1rem;
    padding: 1rem 1.25rem;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    background: #ffffff;
}

.search-suggestions-header {
    font-size: 14px;
    font-weight: 500;
    color: #202124;
    margin-bottom: 4px;
}

.search-suggestions-subheader {
    font-size: 12px;
    color: #5f6368;
    margin-bottom: 1rem;
}

.search-suggestions-subheader a {
    color: #1a73e8;
    text-decoration: none;
}

.search-suggestions-subheader a:hover {
    text-decoration: underline;
}

.search-chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.search-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 20px;
    font-size: 14px;
    color: #202124;
    cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
}

.search-chip:hover {
    background: #e8f0fe;
    border-color: #1a73e8;
}

.search-chip .google-icon {
    width: 18px;
    height: 18px;
    flex-shrink: 0;
}

/* Project Cards - Clean & Minimal */
.project-card {
    background-color: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 8px 0;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.project-card:hover {
    background-color: #f8f9fa;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transform: translateY(-1px);
}

.project-card.active {
    background-color: #e8f0fe; /* Light blue selection */
    border-color: var(--accent-blue);
    color: var(--accent-blue);
}

/* Chat Messages */
.chat-message {
    padding: 1rem 0;
}

.chat-bubble {
    padding: 16px 20px;
    border-radius: 18px;
    line-height: 1.5;
    font-size: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.chat-bubble.user {
    background-color: #e8f0fe; /* Google Blue Tint */
    color: #1a73e8;
    border-bottom-right-radius: 4px;
}

.chat-bubble.assistant {
    background-color: #ffffff;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
}

/* Sidebar Section Headers */
.sidebar-section {
    font-size: 13px; /* Slightly larger for readability */
    font-weight: 600;
    color: var(--text-primary) !important; /* Force dark color */
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Capabilities & Tips Boxes - Google Style Cards */
.big-box {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #dadce0;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #bdc1c6;
}

/* Modal/Overlay Fixes */
div[data-baseweb="modal"], div[class*="backdrop"] {
    display: none !important;
}

/* Stop Button Style */
.stop-button {
    background-color: #dc3545 !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background-color 0.2s !important;
}

.stop-button:hover {
    background-color: #c82333 !important;
}

/* Edit Button on User Messages */
.edit-btn {
    background: transparent;
    border: none;
    color: #5f6368;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    opacity: 0.6;
    transition: all 0.2s;
}

.edit-btn:hover {
    background: #f1f3f4;
    opacity: 1;
    color: #1a73e8;
}

.user-message-wrapper {
    position: relative;
}

.user-message-wrapper:hover .edit-btn {
    opacity: 1;
}

</style>
""", unsafe_allow_html=True)

# Constants
MAX_PROJECTS = 10

# Initialize session state
def init_session_state():
    if 'projects' not in st.session_state:
        st.session_state.projects = [{
            'id': str(uuid.uuid4()),
            'name': 'Default Project',
            'messages': [],
            'documents': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'cross_memory': False
        }]
    
    if 'current_project_id' not in st.session_state:
        st.session_state.current_project_id = st.session_state.projects[0]['id']
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if 'knowledge_base_loaded' not in st.session_state:
        st.session_state.knowledge_base_loaded = False
        st.session_state.kb_count = 0
        st.session_state.kb_categories = []
    
    if 'active_citation' not in st.session_state:
        st.session_state.active_citation = None
    
    if 'input_value' not in st.session_state:
        st.session_state.input_value = ''
    
    if 'renaming_project_id' not in st.session_state:
        st.session_state.renaming_project_id = None
    
    if 'rag_indexing' not in st.session_state:
        st.session_state.rag_indexing = False
    
    if 'rag_stats' not in st.session_state:
        st.session_state.rag_stats = None
    
    if 'rag_indexed' not in st.session_state:
        st.session_state.rag_indexed = False
    
    if 'rag_chunk_count' not in st.session_state:
        st.session_state.rag_chunk_count = 0
    
    if 'auto_index_triggered' not in st.session_state:
        st.session_state.auto_index_triggered = False
    
    if 'stop_streaming' not in st.session_state:
        st.session_state.stop_streaming = False
    
    if 'editing_message_id' not in st.session_state:
        st.session_state.editing_message_id = None
    
    if 'edit_text' not in st.session_state:
        st.session_state.edit_text = ''
    
    if 'pending_edit_prompt' not in st.session_state:
        st.session_state.pending_edit_prompt = None
    
    if 'show_rag_debug' not in st.session_state:
        st.session_state.show_rag_debug = False
    
    if 'last_rag_context' not in st.session_state:
        # Always keep a string so the debug panel can render even when empty.
        st.session_state.last_rag_context = ""
    elif st.session_state.last_rag_context is None:
        # Backward-compat: older sessions may have stored None.
        st.session_state.last_rag_context = ""

    if 'last_citation_allowlist' not in st.session_state:
        st.session_state.last_citation_allowlist = []
    if 'last_citation_violations' not in st.session_state:
        st.session_state.last_citation_violations = []

def get_current_project() -> Optional[Dict]:
    """Get the current project"""
    for p in st.session_state.projects:
        if p['id'] == st.session_state.current_project_id:
            return p
    return None

def create_new_project(name: str = None) -> Dict:
    """Create a new project"""
    return {
        'id': str(uuid.uuid4()),
        'name': name or f"Project {datetime.now().strftime('%Y-%m-%d')}",
        'messages': [],
        'documents': [],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'cross_memory': False
    }

def get_conversation_history(current_project: Dict, include_current_message: bool = False) -> List[Dict]:
    """
    Get conversation history for AI context.
    
    This function builds a complete conversation history that enables:
    1. Within-session memory - AI remembers all Q&A in current project
    2. Cross-project memory - When enabled, AI can access history from other linked projects
    
    Args:
        current_project: The current project dictionary
        include_current_message: Whether to include the last message (usually False when calling AI)
    
    Returns:
        List of message dicts with 'role' and 'text' keys for AI context
    """
    history = []
    
    # Check if cross-project memory is enabled for current project
    cross_memory_enabled = current_project.get('cross_memory', False)
    
    if cross_memory_enabled:
        # Collect history from ALL projects with cross_memory enabled
        # This allows the AI to reference prior conversations across projects
        for project in st.session_state.projects:
            # Include messages from projects that have cross_memory enabled
            if project.get('cross_memory', False) and project['id'] != current_project['id']:
                project_messages = project.get('messages', [])
                if project_messages:
                    # Add project context marker
                    history.append({
                        'role': 'user',
                        'text': f"[Context from project '{project['name']}']:"
                    })
                    # Add messages from this project (limit to last 10 to avoid token overflow)
                    for msg in project_messages[-10:]:
                        history.append({
                            'role': msg.get('role', 'user'),
                            'text': msg.get('text', '')
                        })
    
    # Add current project's messages (this is the main conversation history)
    current_messages = current_project.get('messages', [])
    
    # Determine how many messages to include
    messages_to_include = current_messages if include_current_message else current_messages[:-1] if current_messages else []
    
    for msg in messages_to_include:
        # Only include messages with actual text content
        msg_text = msg.get('text', '')
        if msg_text and msg_text.strip():
            history.append({
                'role': msg.get('role', 'user'),
                'text': msg_text
            })
    
    return history

def parse_citations(text: str) -> str:
    """Parse citation JSON and convert to HTML buttons"""
    pattern = r'\[\[\{.*?\}\]\]'
    
    def replace_citation(match):
        try:
            json_str = match.group(0)[2:-2]  # Remove [[ and ]]
            citation = json.loads(json_str)
            ref = citation.get('ref', 'Citation')
            # Format in proper OSCOLA style - just the reference in brackets
            return f'({ref})'
        except:
            return match.group(0)
    
    return re.sub(pattern, replace_citation, text)

def render_message(message: Dict, is_user: bool, message_id: str = None, show_edit: bool = True):
    """Render a chat message"""
    import html
    import urllib.parse
    
    # Clean text (remove ** and * markdown)
    text = message.get('text', '')
    text = text.replace('**', '').replace('*', '')
    
    # Parse citations
    text_with_citations = parse_citations(text)
    
    # CRITICAL: Convert newlines to HTML line breaks for paragraph gaps to display
    # Double newlines (\n\n) become paragraph breaks (<br><br>)
    # Single newlines (\n) become line breaks (<br>)
    text_with_citations = text_with_citations.replace('\n\n', '<br><br>')
    text_with_citations = text_with_citations.replace('\n', '<br>')
    
    if is_user:
        # User message with label - rendered as HTML only (no interactive edit here)
        st.markdown(f"""
        <div class="chat-message user user-message-wrapper">
            <div class="chat-bubble user">
                <div class="chat-role user">You</div>
                <div class="chat-text">{text_with_citations}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Get grounding data
        grounding_sources = message.get('grounding_sources', [])
        search_suggestions = message.get('search_suggestions', [])
        
        # Render the main message text
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="chat-bubble assistant">
                <div class="chat-text">{text_with_citations}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Render Sources box if available (as separate component)
        if grounding_sources:
            # Remove duplicates based on title
            seen_titles = set()
            unique_sources = []
            for source in grounding_sources:
                title = source.get('title', '')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_sources.append(source)
            
            if unique_sources:
                sources_links = ""
                for i, source in enumerate(unique_sources, 1):
                    url = source.get('url', '#')
                    title = html.escape(source.get('title', 'Source'))
                    sources_links += f'<a href="{url}" target="_blank" class="source-link">{i}. {title}</a>'
                
                st.markdown(f"""
                <div class="sources-box">
                    <div class="sources-box-header">
                        Sources <span class="help-icon" title="These sources were used by Google Search to provide this answer">?</span>
                    </div>
                    {sources_links}
                </div>
                """, unsafe_allow_html=True)
        
        # Render Search Suggestions box if available (as separate component)
        if search_suggestions:
            chips_html = ""
            for suggestion in search_suggestions:
                safe_suggestion = html.escape(suggestion)
                search_url = f"https://www.google.com/search?q={urllib.parse.quote(suggestion)}"
                chips_html += f'''<a href="{search_url}" target="_blank" class="search-chip"><svg class="google-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>{safe_suggestion}</a>'''
            
            st.markdown(f"""
            <div class="search-suggestions-box">
                <div class="search-suggestions-header">Search Suggestions</div>
                <div class="search-chip-container">
                    {chips_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Render RAG Debug info if enabled (show even when empty so "nothing shown" is actionable)
        # Backward-compat: older saved messages may have rag_context=None.
        rag_context = message.get('rag_context') or ""
        if st.session_state.get('show_rag_debug'):
            with st.expander("📚 RAG Retrieved Content (Debug)", expanded=False):
                st.markdown(f"**Context Length:** {len(rag_context)} characters")
                st.markdown("---")
                allow = message.get('citation_allowlist') or []
                removed = message.get('citation_violations') or []
                if removed:
                    st.warning(f"Removed {len(removed)} non-retrieved authority mention(s) from the saved answer.")
                    st.code("\n".join(removed[:20]) + ("..." if len(removed) > 20 else ""), language=None)
                    st.markdown("---")
                # Thin context warning
                ctx_len = len(rag_context)
                if 0 < ctx_len < 15000:
                    st.warning(f"⚠️ Low retrieval: only {ctx_len:,} characters retrieved. The knowledge base may lack materials for this legal area. Consider adding relevant PDFs (statutes, cases, textbooks) to improve answer quality.")
                if allow:
                    has_primary = any((" act " in a.lower() and any(ch.isdigit() for ch in a)) or (" v " in a.lower() and "[" in a) for a in allow)
                    if not has_primary:
                        st.warning("No obvious primary authorities (Acts/cases) detected in retrieved sources for this answer; consider adding statute/judgment PDFs to the index for 90+ work.")
                    st.markdown("**Allowed Authorities (preview):**")
                    st.code("\n".join(allow[:20]) + ("..." if len(allow) > 20 else ""), language=None)
                    st.markdown("---")
                if rag_context:
                    # Display the context in a scrollable code block (first 8000 chars)
                    st.code(rag_context[:8000] + ("..." if len(rag_context) > 8000 else ""), language=None)
                else:
                    st.code("(No RAG context returned for this message.)", language=None)

def main():
    init_session_state()
    
    # Load knowledge base on startup
    if not st.session_state.knowledge_base_loaded:
        index = load_law_resource_index()
        if index:
            st.session_state.knowledge_base_loaded = True
            st.session_state.kb_count = index.totalFiles
            st.session_state.kb_categories = index.categories
            initialize_knowledge_base()
    
    # ===== SIDEBAR =====
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <span style="color: #1a73e8; font-size: 1.25rem;">⚖️</span>
            <h1 style="color: #202124; font-family: 'Product Sans', sans-serif;">Legal AI</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration Section
        st.markdown('<div class="sidebar-section">⚙️ Configuration</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="Enter Key or use Default...",
            help="Leave empty to use the default system key."
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        st.markdown("---")
        
        # Projects Section - Header and New button on same line
        col_header, col_new = st.columns([3, 1])
        with col_header:
            st.markdown(f'<div class="sidebar-section" style="display: flex; align-items: center; height: 38px; margin: 0;">📁 Projects ({len(st.session_state.projects)}/{MAX_PROJECTS})</div>', unsafe_allow_html=True)
        with col_new:
            if st.button("New", disabled=len(st.session_state.projects) >= MAX_PROJECTS, key="new_project_btn", use_container_width=True):
                new_project = create_new_project()
                st.session_state.projects.insert(0, new_project)
                st.session_state.current_project_id = new_project['id']
                st.rerun()
        
        # Project list with rename functionality
        for project in st.session_state.projects:
            is_active = project['id'] == st.session_state.current_project_id
            is_renaming = st.session_state.renaming_project_id == project['id']
            
            col1, col2, col3, col4 = st.columns([5, 1, 1, 1])
            
            with col1:
                if is_renaming:
                    # Show text input when renaming
                    new_name = st.text_input(
                        "New name",
                        value=project['name'],
                        key=f"rename_{project['id']}",
                        label_visibility="collapsed"
                    )
                else:
                    # Show project button
                    if st.button(
                        project['name'],
                        key=f"proj_{project['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.current_project_id = project['id']
                        st.rerun()
            
            with col2:
                if is_renaming:
                    # Save button when renaming
                    if st.button("✓", key=f"save_{project['id']}", help="Save"):
                        new_name = st.session_state.get(f"rename_{project['id']}", project['name'])
                        if new_name.strip():
                            project['name'] = new_name.strip()
                        st.session_state.renaming_project_id = None
                        st.rerun()
                else:
                    # Rename button (pencil icon)
                    if st.button("✎", key=f"rename_btn_{project['id']}", help="Rename"):
                        st.session_state.renaming_project_id = project['id']
                        st.rerun()
            
            with col3:
                # Cross memory toggle
                icon = "🔗" if project.get('cross_memory') else "⛓️"
                if st.button(icon, key=f"mem_{project['id']}", help="Toggle cross-memory"):
                    project['cross_memory'] = not project.get('cross_memory', False)
                    st.rerun()
            
            with col4:
                # Delete button
                if len(st.session_state.projects) > 1:
                    if st.button("✕", key=f"del_{project['id']}", help="Delete project"):
                        st.session_state.projects = [p for p in st.session_state.projects if p['id'] != project['id']]
                        if st.session_state.current_project_id == project['id']:
                            st.session_state.current_project_id = st.session_state.projects[0]['id']
                        reset_session(project['id'])
                        st.rerun()
        
        st.caption("Double-click project name or click ✎ to rename. 🔗 = share memory across projects.")
        
        st.markdown("---")
        
        # Research Materials Section
        st.markdown('<div class="sidebar-section">📚 Research Materials</div>', unsafe_allow_html=True)
        
        # Link input
        link_url = st.text_input("Add Web Reference (URL)", placeholder="https://...")
        if st.button("Add URL", use_container_width=True):
            if link_url:
                current_project = get_current_project()
                if current_project:
                    url = link_url if link_url.startswith('http') else f'https://{link_url}'
                    current_project['documents'].append({
                        'id': str(uuid.uuid4()),
                        'type': 'link',
                        'name': url,
                        'mimeType': 'text/uri-list',
                        'data': url,
                        'size': 0
                    })
                    st.rerun()
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Files",
            type=['pdf', 'txt', 'md', 'csv'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            current_project = get_current_project()
            if current_project:
                files_added = False
                for file in uploaded_files:
                    # Check if file already added
                    existing_names = [d['name'] for d in current_project['documents']]
                    if file.name not in existing_names:
                        content = file.read()
                        current_project['documents'].append({
                            'id': str(uuid.uuid4()),
                            'type': 'file',
                            'name': file.name,
                            'mimeType': file.type or 'application/octet-stream',
                            'data': encode_file_to_base64(content),
                            'size': len(content)
                        })
                        files_added = True
                # Only rerun after processing all files
                if files_added:
                    st.rerun()
        
        st.markdown("---")
        
        # ===== KNOWLEDGE BASE ACTIVE SECTION =====
        # This section handles auto-indexing for Streamlit Cloud deployment
        if RAG_AVAILABLE:
            try:
                rag_service = get_rag_service()
                stats = rag_service.get_stats()
                
                # Check if we need to auto-index (first deployment or empty database)
                resources_path = os.path.join(os.path.dirname(__file__), 'Law resouces  copy 2')
                
                # Auto-index on first startup if database is empty
                if stats['total_chunks'] == 0 and not st.session_state.auto_index_triggered and os.path.exists(resources_path):
                    st.session_state.auto_index_triggered = True
                    st.session_state.rag_indexing = True
                    st.rerun()
                
                # Show indexing progress if currently indexing
                if st.session_state.rag_indexing:
                    st.markdown('<div class="sidebar-section">📚 Knowledge Base</div>', unsafe_allow_html=True)
                    st.info("⏳ Auto-indexing law documents... Please wait.")
                    
                    with st.spinner("Indexing documents... This may take a few minutes on first startup."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def progress_callback(count, filename):
                            progress_bar.progress(min(count / 500, 1.0))  # Estimate ~500 files
                            status_text.text(f"Processing: {filename[:40]}...")
                        
                        try:
                            result = rag_service.index_documents(resources_path, progress_callback)
                            st.session_state.rag_stats = result
                            st.session_state.rag_indexed = True
                            st.session_state.rag_chunk_count = result['chunks']
                            st.session_state.rag_indexing = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Indexing error: {str(e)}")
                            st.session_state.rag_indexing = False
                else:
                    # Show Knowledge Base Active status
                    st.markdown('<div class="sidebar-section">📚 Knowledge Base Active</div>', unsafe_allow_html=True)
                    
                    if stats['total_chunks'] > 0:
                        st.caption("The AI can now search inside your law documents!")
                    else:
                        st.caption("No documents added. AI will use knowledge base and Google Search.")
                        
                        # Show manual index button if no documents indexed
                        if os.path.exists(resources_path):
                            if st.button("🔄 Index Law Documents", use_container_width=True, help="Extract and index text from all law resources"):
                                st.session_state.rag_indexing = True
                                st.rerun()
                
            except Exception as e:
                print(f"RAG service error: {e}")
                st.markdown('<div class="sidebar-section">📚 Knowledge Base Active</div>', unsafe_allow_html=True)
                st.caption("AI will use knowledge base and Google Search.")
        else:
            # RAG not available - show basic Knowledge Base status  
            st.markdown('<div class="sidebar-section">📚 Knowledge Base Active</div>', unsafe_allow_html=True)
            if st.session_state.knowledge_base_loaded:
                st.caption("AI will use knowledge base and Google Search.")
            else:
                st.caption("No documents added. AI will use knowledge base and Google Search.")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0;">
            <div class="ai-badge">AI</div>
            <div>
                <div style="font-size: 0.875rem; font-weight: 500; color: #202124;">Gemini 3 Pro</div>
                <div style="font-size: 0.75rem; color: #5f6368;">""" + ("Custom Key Active" if st.session_state.api_key else "Default Key Active") + """</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== MAIN AREA =====
    current_project = get_current_project()

    # Disclaimer
    st.caption("AI may make mistakes. All output is for reference only.")

    # Header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("### 📖 Legal Research Workspace")
    with col2:
        if st.button("Clear", type="secondary"):
            if current_project:
                current_project['messages'] = []
                reset_session(current_project['id'])
                st.rerun()
    
    st.markdown("---")
    
    # Chat area
    if current_project:
        messages = current_project.get('messages', [])
        
        # Check if there are any messages - if yes, show chat only
        if len(messages) > 0:
            # Display existing messages with edit functionality
            for idx, msg in enumerate(messages):
                is_user = msg.get('role') == 'user'
                msg_id = msg.get('id', str(idx))
                
                # Check if this message is being edited
                if is_user and st.session_state.editing_message_id == msg_id:
                    # Show edit interface
                    st.markdown("""
                    <div class="chat-message user">
                        <div class="chat-bubble user" style="padding: 8px;">
                            <div class="chat-role user">You (Editing)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Text area for editing
                    edited_text = st.text_area(
                        "Edit your question:",
                        value=msg.get('text', ''),
                        key=f"edit_area_{msg_id}",
                        height=100,
                        label_visibility="collapsed"
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("✓ Submit", key=f"submit_edit_{msg_id}", type="primary", use_container_width=True):
                            if edited_text.strip():
                                # Find index of the message being edited
                                edit_idx = next((i for i, m in enumerate(messages) if m.get('id') == msg_id), None)
                                if edit_idx is not None:
                                    # Update the message text
                                    messages[edit_idx]['text'] = edited_text.strip()
                                    # Remove all messages after this one (including AI response)
                                    current_project['messages'] = messages[:edit_idx + 1]
                                    # Clear editing state
                                    st.session_state.editing_message_id = None
                                    # This will trigger a new AI response
                                    st.session_state.pending_edit_prompt = edited_text.strip()
                                    st.rerun()
                    with col2:
                        if st.button("✕ Cancel", key=f"cancel_edit_{msg_id}", use_container_width=True):
                            st.session_state.editing_message_id = None
                            st.rerun()
                else:
                    # Normal message display
                    if is_user:
                        # User message with Edit button
                        col_msg, col_btn = st.columns([20, 1])
                        with col_msg:
                            render_message(msg, is_user=True, message_id=msg_id)
                        with col_btn:
                            if st.button("✎", key=f"edit_btn_{msg_id}", help="Edit this question"):
                                st.session_state.editing_message_id = msg_id
                                st.rerun()
                    else:
                        # Assistant message - no edit button
                        render_message(msg, is_user=False)
        else:
            # EMPTY STATE - Show welcome screen with boxes
            # Use a placeholder so we can clear it immediately when user types
            welcome_placeholder = st.empty()
            with welcome_placeholder.container():
                st.markdown("""
                <div style="text-align: center; max-width: 40rem; margin: 3rem auto; padding: 2rem;">
                    <div style="font-size: 4rem; color: #dadce0; margin-bottom: 1rem;">📚</div>
                    <h2 style="font-family: 'Product Sans', sans-serif; font-size: 2rem; color: #202124; margin-bottom: 0.5rem;">Legal AI</h2>
                    <p style="color: #5f6368; font-size: 1rem; margin-bottom: 2rem;">AI-powered legal research assistant</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Knowledge Base Status
                col1, col2, col3 = st.columns([1, 2, 1])
                if st.session_state.knowledge_base_loaded:
                    with col2:
                        st.success("✅ Knowledge Base Active")
                
                # Centered content - BIGGER BOXES with DARKER TEXT
                with col2:
                    st.markdown('<p style="color: #202124; font-size: 1.25rem; font-weight: 500; text-align: center; margin: 2rem 0;">Just ask your question</p>', unsafe_allow_html=True)
                    
                    # Capabilities box - React Style (Blue Dots)
                    st.markdown("""
                    <div style="background: white; border: 1px solid #dadce0; border-radius: 0.75rem; padding: 2rem; margin: 1.5rem 0; text-align: left; box-shadow: 0 1px 2px rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);">
                        <h4 style="font-size: 0.75rem; font-weight: 700; color: #5f6368; text-transform: uppercase; margin-bottom: 1rem; letter-spacing: 0.5px;">Capabilities</h4>
                        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                            <div class="custom-list-item"><div class="blue-dot"></div>Essay Writing</div>
                            <div class="custom-list-item"><div class="blue-dot"></div>Problem Questions</div>
                            <div class="custom-list-item"><div class="blue-dot"></div>Legal Advice & Strategy</div>
                            <div class="custom-list-item"><div class="blue-dot"></div>General Queries</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tips box - React Style (Chips)
                    st.markdown("""
                    <div style="background: white; border: 1px solid #dadce0; border-radius: 0.75rem; padding: 2rem; margin: 1.5rem 0; text-align: left; box-shadow: 0 1px 2px rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);">
                        <h4 style="font-size: 0.75rem; font-weight: 700; color: #5f6368; margin-bottom: 1rem; letter-spacing: 0.5px; text-transform: uppercase; display: flex; align-items: center; gap: 0.5rem;">
                            <span style="color: #eab308; font-size: 1rem;">✨</span> Try Asking
                        </h4>
                        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                            <div class="suggestion-chip">"What are the key elements of a valid contract under English law?"</div>
                            <div class="suggestion-chip">"Explain the duty of care in negligence under UK tort law"</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Check for pending edit prompt (from editing a previous message)
    pending_prompt = st.session_state.pending_edit_prompt
    if pending_prompt:
        st.session_state.pending_edit_prompt = None  # Clear it
        prompt = pending_prompt
    else:
        # Input area - Docked at bottom (st.chat_input)
        prompt = st.chat_input("Ask for an Essay, Case Analysis, or Client Advice...")
    
    if prompt:
        # Clear welcome screen immediately if it exists
        if 'welcome_placeholder' in locals():
            welcome_placeholder.empty()
            
        if current_project:
            prompt_lower = prompt.strip().lower()
            is_starting_pending_long = bool(st.session_state.pending_long_prompt) and prompt_lower in {"proceed now", "part 1", "continue"}
            prompt_for_model = st.session_state.pending_long_prompt if is_starting_pending_long else prompt

            # Only add user message if this is a NEW prompt (not an edited one)
            # Edited prompts already have the message in place
            if not pending_prompt:
                user_message = {
                    'id': str(uuid.uuid4()),
                    'role': 'user',
                    'text': prompt,
                    'timestamp': datetime.now().isoformat()
                }
                current_project['messages'].append(user_message)
                
                # Display user message immediately (message loop already ran before this point)
                render_message(user_message, is_user=True)

            
            # Get API key
            api_key = st.session_state.api_key or os.environ.get('GEMINI_API_KEY', '')
            
            if not api_key:
                st.error("Please enter a Gemini API key in the sidebar configuration.")
            else:
                # Check for long essay and show suggestion
                # If the user is starting a pending long response, do not re-run the "await choice" gate.
                # Render the long-essay gate inside a placeholder so we can clear it immediately on the next run.
                long_essay_gate = st.empty()
                long_essay_info = detect_long_essay(prompt_for_model) if not is_starting_pending_long else {'is_long_essay': False}
                if long_essay_info.get('is_long_essay'):
                    with long_essay_gate.container():
                        st.info(long_essay_info['suggestion_message'])
                        st.markdown("---")

                        # If await_user_choice is True, STOP here and don't show "Thinking..." yet
                        # Wait for user to respond with their choice (proceed now or use parts approach)
                        if long_essay_info.get('await_user_choice'):
                            st.session_state.pending_long_prompt = prompt
                            st.info("💡 **Please respond** with either:\n- \"Proceed now\" - I'll write 2500 words MAX\n- \"Part 1\" or your specific request - To start with the parts approach")
                            # Stop execution here - wait for user's next message
                            st.stop()
                else:
                    # Ensure any previously rendered gate UI is removed before streaming begins.
                    long_essay_gate.empty()

                # In-chat "Thinking..." bubble (visible in the thread while retrieval/generation runs)
                assistant_chat = st.chat_message("assistant")
                with assistant_chat:
                    thinking_placeholder = st.empty()
                    thinking_placeholder.markdown(
                        """
                        <div style="
                            border: 1px solid #e0e0e0;
                            background: #f8f9fa;
                            border-radius: 12px;
                            padding: 14px 16px;
                            display: flex;
                            align-items: center;
                            gap: 10px;
                            max-width: 900px;
                        ">
                            <div style="display:flex; gap:6px; align-items:center;">
                                <span style="width:8px; height:8px; background:#5f6368; border-radius:50%; opacity:.35; animation: dotPulse 1.2s infinite;"></span>
                                <span style="width:8px; height:8px; background:#5f6368; border-radius:50%; opacity:.35; animation: dotPulse 1.2s infinite .15s;"></span>
                                <span style="width:8px; height:8px; background:#5f6368; border-radius:50%; opacity:.35; animation: dotPulse 1.2s infinite .30s;"></span>
                            </div>
                            <div style="color:#5f6368; font-style: italic;">Thinking...</div>
                        </div>
                        <style>
                        @keyframes dotPulse {
                            0%, 100% { opacity: .25; transform: translateY(0px); }
                            50% { opacity: 1; transform: translateY(-1px); }
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    status_placeholder = st.empty()
                    status_placeholder.markdown(
                        "<div style='color:#5f6368; font-size: 0.85rem; margin-top: 6px;'>Retrieving sources…</div>",
                        unsafe_allow_html=True,
                    )

                    response_placeholder = st.empty()
                    stop_button_placeholder = st.empty()
                full_response = ""
                was_stopped = False
                
                try:
                    # Build conversation history for context
                    # This enables the AI to remember prior Q&A and provide follow-up responses
                    conversation_history = get_conversation_history(current_project, include_current_message=False)
                    
                    # Use streaming for faster response  
                    # Pass history to enable conversation memory
                    # NOTE: Retrieval happens inside send_message_with_docs; we surface a status line above.
                    stream, rag_context = send_message_with_docs(
                        api_key,
                        prompt_for_model,
                        current_project.get('documents', []),
                        current_project['id'],
                        history=conversation_history,  # Enable conversation memory
                        stream=True
                    )
                    status_placeholder.markdown(
                        "<div style='color:#5f6368; font-size: 0.85rem; margin-top: 6px;'>Generating answer…</div>",
                        unsafe_allow_html=True,
                    )
                    
                    # DEBUG: Keep RAG context (even if empty) so the panel can render.
                    st.session_state.last_rag_context = rag_context or ""
                    if rag_context:
                        print(f"\n[DEBUG RAG CONTEXT] Retrieved content for query: '{prompt[:50]}...'")
                        print(f"[DEBUG RAG CONTEXT] Context length: {len(rag_context)} characters")
                    
                    # Clear thinking indicator once we start getting response
                    first_chunk = True
                    grounding_sources = []
                    search_suggestions = []
                    last_chunk = None

                    def _extract_stream_text(chunk_obj) -> str:
                        """
                        Extract text from both new `google.genai` stream chunks and legacy shapes.
                        Some chunks may not expose `.text` even though they contain text in candidates.
                        """
                        if chunk_obj is None:
                            return ""
                        try:
                            txt = getattr(chunk_obj, "text", None)
                            if isinstance(txt, str) and txt:
                                return txt
                        except Exception:
                            pass
                        try:
                            candidates = getattr(chunk_obj, "candidates", None)
                            if candidates:
                                cand = candidates[0]
                                content = getattr(cand, "content", None)
                                parts = getattr(content, "parts", None)
                                if parts:
                                    out_parts = []
                                    for p in parts:
                                        t = getattr(p, "text", None)
                                        if isinstance(t, str) and t:
                                            out_parts.append(t)
                                    joined = "".join(out_parts).strip()
                                    if joined:
                                        return joined
                        except Exception:
                            pass
                        if isinstance(chunk_obj, dict):
                            try:
                                txt = chunk_obj.get("text")
                                if isinstance(txt, str) and txt:
                                    return txt
                            except Exception:
                                pass
                        return ""
                    
                    # Stream the response chunks
                    # Keep "Thinking..." box visible during streaming; show final answer only when complete
                    for chunk in stream:
                        # Check if stop was requested
                        if st.session_state.stop_streaming:
                            was_stopped = True
                            st.session_state.stop_streaming = False
                            break

                        last_chunk = chunk  # Keep track of final chunk for metadata
                        chunk_text = _extract_stream_text(chunk)
                        if chunk_text:
                            if first_chunk:
                                # Update thinking box to show generation progress
                                thinking_placeholder.markdown(
                                    """
                                    <div style="
                                        border: 1px solid #e0e0e0;
                                        background: #f8f9fa;
                                        border-radius: 12px;
                                        padding: 14px 16px;
                                        display: flex;
                                        align-items: center;
                                        gap: 10px;
                                        max-width: 900px;
                                    ">
                                        <div style="display:flex; gap:6px; align-items:center;">
                                            <span style="width:8px; height:8px; background:#1a73e8; border-radius:50%; opacity:.35; animation: dotPulse 1.2s infinite;"></span>
                                            <span style="width:8px; height:8px; background:#1a73e8; border-radius:50%; opacity:.35; animation: dotPulse 1.2s infinite .15s;"></span>
                                            <span style="width:8px; height:8px; background:#1a73e8; border-radius:50%; opacity:.35; animation: dotPulse 1.2s infinite .30s;"></span>
                                        </div>
                                        <div style="color:#1a73e8; font-style: italic;">Generating final answer — please wait…</div>
                                    </div>
                                    <style>
                                    @keyframes dotPulse {
                                        0%, 100% { opacity: .25; transform: translateY(0px); }
                                        50% { opacity: 1; transform: translateY(-1px); }
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                status_placeholder.empty()
                                # Show Stop button
                                stop_button_placeholder.button("⏹ Stop", key="stop_streaming_btn", type="secondary", on_click=lambda: setattr(st.session_state, 'stop_streaming', True))
                                first_chunk = False

                            full_response += chunk_text

                    # Streaming complete - strip any internal reasoning and show the final answer
                    full_response = strip_internal_reasoning(full_response)
                    thinking_placeholder.empty()
                    status_placeholder.empty()
                    if full_response.strip():
                        response_placeholder.markdown(full_response)

                    # Clear stop button
                    stop_button_placeholder.empty()
                    
                    # Extract grounding metadata from the final response
                    if last_chunk is not None:
                        try:
                            # Try to get grounding metadata from candidates
                            if hasattr(last_chunk, 'candidates') and last_chunk.candidates:
                                candidate = last_chunk.candidates[0]
                                
                                # Access grounding_metadata - it's a Pydantic model in new library
                                gm = getattr(candidate, 'grounding_metadata', None)
                                if gm is not None:
                                    print(f"DEBUG: grounding_metadata found!")
                                    
                                    # Extract grounding chunks (source URLs)
                                    chunks = getattr(gm, 'grounding_chunks', None)
                                    if chunks:
                                        print(f"DEBUG: Found {len(chunks)} grounding_chunks")
                                        for gc in chunks:
                                            web = getattr(gc, 'web', None)
                                            if web:
                                                url = getattr(web, 'uri', '') or ''
                                                title = getattr(web, 'title', '') or ''
                                                print(f"DEBUG: Source - {title}: {url}")
                                                grounding_sources.append({
                                                    'url': url,
                                                    'title': title
                                                })
                                    else:
                                        print("DEBUG: No grounding_chunks found")
                                    
                                    # Extract web_search_queries for suggestions
                                    queries = getattr(gm, 'web_search_queries', None)
                                    if queries:
                                        search_suggestions = list(queries)
                                        print(f"DEBUG: web_search_queries: {search_suggestions}")
                                    else:
                                        print("DEBUG: No web_search_queries found")
                                    
                                    # Try search_entry_point for rendered search widget
                                    sep = getattr(gm, 'search_entry_point', None)
                                    if sep:
                                        rendered = getattr(sep, 'rendered_content', None)
                                        if rendered:
                                            print(f"DEBUG: search_entry_point rendered_content (first 200 chars): {rendered[:200]}")
                                else:
                                    print("DEBUG: No grounding_metadata on candidate")
                                    
                        except Exception as meta_e:
                            print(f"Could not extract grounding metadata: {meta_e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Debug: Print what we collected
                    print(f"DEBUG: Final grounding_sources count: {len(grounding_sources)}")
                    print(f"DEBUG: Final search_suggestions count: {len(search_suggestions)}")
                    
                    # Fallback: some streaming backends only provide text on the final chunk.
                    # If we haven't displayed anything yet, keep the Thinking UI visible until we can.
                    if not full_response and last_chunk is not None:
                        full_response = _extract_stream_text(last_chunk)
                    if full_response and first_chunk:
                        thinking_placeholder.empty()
                        status_placeholder.empty()
                        response_placeholder.markdown(full_response)
                        first_chunk = False
                    
                    # Add assistant message with grounding data
                    # If stopped, add indicator to the response
                    final_response = full_response
                    if was_stopped and full_response:
                        final_response = full_response + "\n\n[Response stopped by user]"
                    elif final_response.strip():
                        # Optional second-pass tightening for explicit word-count prompts (slow; OFF by default)
                        fix_instruction = _needs_wordcount_fix(prompt_for_model, final_response) if st.session_state.enable_wordcount_adjust else None
                        if st.session_state.enable_wordcount_adjust and fix_instruction:
                            response_placeholder = st.empty()
                            response_placeholder.markdown("""
                            <div class="chat-message assistant">
                                <div class="chat-bubble assistant" style="color: #5f6368; font-style: italic;">
                                    Adjusting to match requested word count…
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            try:
                                rewrite_prompt = (
                                    f"{fix_instruction}\n\n"
                                    "Return ONLY the revised answer.\n"
                                    "Keep the same question order and section headers.\n"
                                    "Use case name + full OSCOLA citation in parentheses.\n"
                                    "If (and only if) this is the final answer, end with exactly one (End of Answer). If it is an intermediate part (it ends with a 'Will Continue...' line), do NOT include (End of Answer).\n\n"
                                    "ORIGINAL USER PROMPT:\n"
                                    f"{prompt_for_model}\n\n"
                                    "DRAFT ANSWER TO REWRITE:\n"
                                    f"{final_response}\n"
                                )
                                (rewrite_text, _), _ = send_message_with_docs(
                                    api_key,
                                    rewrite_prompt,
                                    current_project.get('documents', []),
                                    current_project['id'],
                                    history=conversation_history,
                                    stream=False
                                )
                                if isinstance(rewrite_text, str) and rewrite_text.strip():
                                    final_response = rewrite_text
                            except Exception as fix_e:
                                print(f"Word-count fix skipped due to error: {fix_e}")
                            response_placeholder.empty()

                    if final_response.strip():
                        # Always run strip_internal_reasoning as a final pass —
                        # rewrites (word-count fix, citation-fix) may reintroduce
                        # Law Trove labels, Source N references, or other artifacts.
                        final_response = strip_internal_reasoning(final_response)
                        final_response = _enforce_end_of_answer(final_response)
                        if is_starting_pending_long:
                            st.session_state.pending_long_prompt = None

                        # Post-output citation enforcement: strip any authority mentions not present in retrieved RAG.
                        # Pass rag_context length so the sanitizer can skip when retrieval is thin.
                        allow = get_allowed_authorities_from_rag(rag_context or "", limit=70)
                        rag_ctx_len = len(rag_context or "")
                        sanitized, removed = sanitize_output_against_allowlist(final_response, allow, rag_context_len=rag_ctx_len)
                        st.session_state.last_citation_allowlist = allow
                        st.session_state.last_citation_violations = removed
                        if removed:
                            # Attempt a single rewrite pass to remove/replace non-retrieved authorities,
                            # then re-sanitize as a final safety net.
                            try:
                                allow_lines = "\n".join([f"- {a}" for a in allow]) if allow else "(none)"
                                rewrite_prompt = (
                                    "[STRICT NO-HALLUCINATION REWRITE]\n"
                                    "Rewrite the answer to remove ANY mention/citation of authorities not in the ALLOWED list.\n"
                                    "If a point depends on a missing authority, rewrite the point generically without naming it.\n"
                                    "Do NOT introduce any new cases/statutes/articles.\n\n"
                                    "ALLOWED AUTHORITIES (verbatim):\n"
                                    f"{allow_lines}\n\n"
                                    "ORIGINAL ANSWER:\n"
                                    f"{sanitized}\n"
                                )
                                (rewrite_text, _), _ = send_message_with_docs(
                                    api_key,
                                    rewrite_prompt,
                                    current_project.get('documents', []),
                                    current_project['id'],
                                    history=conversation_history,
                                    stream=False
                                )
                                if isinstance(rewrite_text, str) and rewrite_text.strip():
                                    sanitized, removed2 = sanitize_output_against_allowlist(rewrite_text, allow, rag_context_len=rag_ctx_len)
                                    st.session_state.last_citation_violations = removed2
                            except Exception as cite_fix_e:
                                print(f"Citation-fix rewrite skipped due to error: {cite_fix_e}")
                            final_response = sanitized
                        else:
                            final_response = sanitized

                    # Ensure we never persist visible removal markers to the chat history.
                    if "[REMOVED: authority not in retrieved sources]" in (final_response or ""):
                        final_response = final_response.replace("[REMOVED: authority not in retrieved sources]", "")
                        # Clean up empty parentheses and broken text left by citation stripping:
                        # "()" or "( )" or "(  )" → remove entirely
                        final_response = re.sub(r'\s*\(\s*\)', '', final_response)
                        # "( ," or "in (," → just the comma
                        final_response = re.sub(r'\s*\(\s*,', ',', final_response)
                        # Dangling open paren before lowercase: "( the judicial" → "the judicial"
                        final_response = re.sub(r'\(\s+([a-z])', r'\1', final_response)
                        # Dangling "see " or "in " before nothing: "see ." → "."
                        final_response = re.sub(r'\b(?:see|in|per|cf)\s+([.;,])', r'\1', final_response)
                        # Repeated commas/semicolons: ",," or ", ," → ","
                        final_response = re.sub(r',\s*,', ',', final_response)
                        final_response = re.sub(r';\s*;', ';', final_response)
                        # Period after comma: ",." → "."
                        final_response = re.sub(r',\s*\.', '.', final_response)
                        # Double spaces left by removals
                        final_response = re.sub(r'  +', ' ', final_response)
                        # Triple+ newlines
                        final_response = re.sub(r"\n{3,}", "\n\n", final_response).strip()
                    
                    # Only add message if there's content
                    if final_response.strip():
                        assistant_message = {
                            'id': str(uuid.uuid4()),
                            'role': 'assistant',
                            'text': final_response,
                            'timestamp': datetime.now().isoformat(),
                            'grounding_sources': grounding_sources if not was_stopped else [],
                            'search_suggestions': search_suggestions if not was_stopped else [],
                            'was_stopped': was_stopped,
                            # Store RAG for per-message debug display (keep string even if empty)
                            'rag_context': rag_context or "",
                            'citation_allowlist': st.session_state.get('last_citation_allowlist', []),
                            'citation_violations': st.session_state.get('last_citation_violations', [])
                        }
                        current_project['messages'].append(assistant_message)
                        
                        # Display RAG Debug info if enabled
                        if st.session_state.show_rag_debug:
                            with st.expander("📚 RAG Retrieved Content (Debug)", expanded=False):
                                last_ctx = st.session_state.last_rag_context or ""
                                st.markdown(f"**Context Length:** {len(last_ctx)} characters")
                                if 0 < len(last_ctx) < 15000:
                                    st.warning(f"⚠️ Low retrieval: only {len(last_ctx):,} characters. Consider adding more materials for this legal area.")
                                st.markdown("---")
                                if last_ctx:
                                    st.code(last_ctx[:5000] + ("..." if len(last_ctx) > 5000 else ""), language=None)
                                else:
                                    st.code("(No RAG context returned for this message.)", language=None)
                    else:
                        # If we got no text at all, surface an actionable error instead of silently adding nothing.
                        error_message = {
                            'id': str(uuid.uuid4()),
                            'role': 'assistant',
                            'text': "I didn’t receive any text back from the model (empty streamed response). Please try again; if it repeats, check the terminal logs for Gemini/API errors or quota/timeouts.",
                            'timestamp': datetime.now().isoformat(),
                            'is_error': True
                        }
                        current_project['messages'].append(error_message)
                    
                except Exception as e:
                    thinking_placeholder.empty()
                    try:
                        status_placeholder.empty()
                    except Exception:
                        pass
                    response_placeholder.empty()
                    # Add error message
                    error_message = {
                        'id': str(uuid.uuid4()),
                        'role': 'assistant',
                        'text': f"I encountered an error: {str(e)}",
                        'timestamp': datetime.now().isoformat(),
                        'is_error': True
                    }
                    current_project['messages'].append(error_message)
            
            st.rerun()

if __name__ == "__main__":
    main()
