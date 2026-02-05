"""
RAG Service for Legal AI
========================
Advanced Retrieval-Augmented Generation with:
- Hybrid Search (ChromaDB + BM25)
- Relevance Threshold Filtering
- Document Diversity Limiting
- Category-Weighted Retrieval
- Semantic Document Chunking
"""

import os
import re
import math
import pickle
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import hashlib
import chromadb
from chromadb.config import Settings

# For document parsing
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyMuPDF not available. PDF parsing disabled.")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx not available. DOCX parsing disabled.")

# Upgraded embedding model (Improvement 6)
# BAAI/bge-large-en-v1.5 produces 1024-dim embeddings (vs 384 for all-MiniLM-L6-v2)
# and scores significantly better on legal/domain-specific retrieval benchmarks.
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("ℹ️ sentence-transformers not installed. Using default ChromaDB embeddings.")

# Embedding model configuration
UPGRADED_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
UPGRADED_COLLECTION_NAME = "law_resources_bge"
DEFAULT_COLLECTION_NAME = "law_resources"

# Hugging Face dataset for pre-indexed ChromaDB
HUGGINGFACE_DATASET_ID = "Agnes999/legalbot6-db"


def download_chroma_from_huggingface(persist_directory: str) -> bool:
    """
    Download pre-indexed ChromaDB from Hugging Face if it doesn't exist locally.

    Returns True if download was successful or DB already exists.
    """
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # Check if chroma.sqlite3 exists (main DB file)
        if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            print(f"ChromaDB already exists at {persist_directory}")
            return True

    print(f"Downloading ChromaDB from Hugging Face: {HUGGINGFACE_DATASET_ID}")

    try:
        from huggingface_hub import snapshot_download

        # Create directory if needed
        os.makedirs(persist_directory, exist_ok=True)

        # Download the dataset to chroma_db folder
        snapshot_download(
            repo_id=HUGGINGFACE_DATASET_ID,
            repo_type="dataset",
            local_dir=persist_directory,
            local_dir_use_symlinks=False
        )

        print(f"Successfully downloaded ChromaDB to {persist_directory}")
        return True

    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"Failed to download ChromaDB from Hugging Face: {e}")
        return False


class BGEEmbeddingFunction:
    """
    ChromaDB-compatible embedding function using BAAI/bge-large-en-v1.5.
    This model produces higher-quality embeddings for legal text retrieval
    compared to the default all-MiniLM-L6-v2.
    """

    def __init__(self, model_name: str = UPGRADED_EMBEDDING_MODEL):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for BGE embeddings. "
                              "Install with: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        # BGE models recommend prepending "Represent this sentence: " for retrieval
        self._query_prefix = "Represent this sentence: "
        print(f"🧠 Loaded embedding model: {model_name}")

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of texts (used by ChromaDB for document embedding)."""
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query with the query prefix for better retrieval."""
        embedding = self.model.encode(
            [self._query_prefix + query],
            normalize_embeddings=True
        )
        return embedding[0].tolist()


# ================================================================================
# CONFIGURATION
# ================================================================================

# Semantic chunking - break at logical boundaries based on document type
CHUNK_STRATEGIES = {
    "case_law": 2000,        # Cases need more context for holdings & reasoning
    "statutes": 1000,        # Statutes are denser, shorter sections work better
    "textbooks": 1500,       # Standard academic text
    "articles": 1800,        # Journal articles - longer for academic context
    "default": 1500          # Default fallback
}

# Category-specific chunk size overrides: {category_lower: {doc_type: size}}
# These fine-tune chunk sizes for legal areas where the default doc-type sizes
# aren't ideal. For example, EU/international law cases often have very long
# reasoned judgments that need larger chunks, while equity & trusts statutes
# are more nuanced and benefit from slightly larger chunks.
CATEGORY_CHUNK_OVERRIDES = {
    "eu law": {
        "case_law": 2400,       # EU judgments are lengthy with multi-part reasoning
        "statutes": 1200,       # EU regulations/directives are more verbose
    },
    "public law": {
        "case_law": 2200,       # Judicial review cases have extended reasoning
    },
    "human rights": {
        "case_law": 2400,       # ECtHR judgments are very long
        "articles": 2000,       # HRA scholarship tends to be detailed
    },
    "equity & trusts": {
        "statutes": 1200,       # Trusts legislation is nuanced
        "case_law": 2200,       # Trust cases involve detailed factual analysis
    },
    "land law": {
        "statutes": 1200,       # Land Registration Act sections need more context
    },
    "company law": {
        "statutes": 1200,       # Companies Act 2006 sections are detailed
        "case_law": 2200,       # Company law cases involve complex facts
    },
    "criminal law": {
        "case_law": 2200,       # Criminal appeal judgments are lengthy
    },
    "intellectual property": {
        "case_law": 2200,       # IP cases involve detailed technical analysis
    },
    "immigration": {
        "case_law": 2200,       # Immigration tribunal decisions are lengthy
        "statutes": 1200,       # Immigration rules are verbose
    },
    "tax": {
        "statutes": 1200,       # Tax legislation is highly technical
        "case_law": 2200,       # Tax cases involve detailed statutory interpretation
    },
}

# Document type detection patterns (for semantic chunking)
DOCUMENT_TYPE_PATTERNS = {
    "case_law": [
        r'\[?\d{4}\]?\s*(UKSC|UKHL|EWCA|EWHC|AC|WLR|QB|KB|Ch|Fam)',  # UK case citations
        r'v\s+\w+\s+\[?\d{4}\]?',  # "v" in case names
        r'judgment|held|ratio|obiter|per\s+(Lord|Lady|LJ)',  # Case terminology
        r'appellant|respondent|claimant|defendant',
        r'judgment of the court|reasons for judgment'
    ],
    "statutes": [
        r'Act\s+\d{4}',  # "Act 2020"
        r'section\s+\d+|s\s*\.\s*\d+|s\s+\d+',  # Section references
        r'regulation\s+\d+|reg\s*\.\s*\d+',  # Regulations
        r'Directive\s+\d+/\d+',  # EU Directives
        r'article\s+\d+|art\s*\.\s*\d+',  # Articles
        r'schedule\s+\d+|sch\s*\.\s*\d+'  # Schedules
    ],
    "articles": [
        r'abstract|introduction|methodology|conclusion',
        r'journal|review|quarterly',
        r'\(\d{4}\)\s+\d+\s+\w+',  # Journal citations (2020) 15 Journal
        r'cambridge|oxford|harvard|yale|stanford',  # University journals
        r'law\s+review|law\s+journal|legal\s+studies'
    ],
    "textbooks": [
        r'chapter\s+\d+|ch\s*\.\s*\d+',
        r'edition|edn|ed\.',
        r'oxford\s+university\s+press|cambridge\s+university\s+press',
        r'sweet\s+&\s+maxwell|hart\s+publishing',
        r'preface|index|bibliography'
    ]
}

# Category weighting for legal domains
CATEGORY_WEIGHTS = {
    "Pensions Law": 1.5,           # High priority for pensions questions
    "Competition Law": 1.5,        # High priority for competition questions
    "Private international law": 1.4,
    "Trusts law": 1.3,
    "Criminal law": 1.2,
    "Contract law": 1.2,
    "Tort law": 1.2,
    "EU law": 1.2,
    "Law and medicine materials": 1.2,
    "Business law": 1.1,
    "default": 1.0
}

# Relevance threshold - chunks below this score are filtered out
RELEVANCE_THRESHOLD = 0.38  # Base threshold (increased from 0.35)

# Document diversity - max chunks from a single document
MAX_CHUNKS_PER_DOCUMENT = 5

# BM25 parameters
BM25_K1 = 1.2  # Term frequency saturation
BM25_B = 0.75  # Length normalization

# Category matching: ignore generic tokens that otherwise over-boost irrelevant domains.
CATEGORY_MATCH_STOPWORDS = {
    "law", "laws", "legal", "materials", "material", "copy", "copies",
    "and", "of", "the", "for", "to", "in", "on", "at", "by", "with"
}

# BM25: stop common prompt/meta words and structural tokens that cause noisy matches.
BM25_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "this", "that", "these", "those",
    "it", "its", "with", "as", "by", "from", "into", "about", "which",
    # Prompt meta / writing tasks
    "essay", "problem", "question", "pq", "answer", "write", "writing", "words", "word",
    "discuss", "analyse", "analyze", "evaluate", "critically", "explain", "outline",
    "describe", "compare", "contrast", "advise", "advice", "apply", "application", "conclusion",
    # Structural headings
    "part", "chapter", "section", "subsection", "paragraph", "para", "paras", "schedule",
    "appendix", "page", "pages", "table", "figure",
    # Generic legal drafting words (high-noise across statutes/bills)
    "act", "acts", "bill", "bills",
    # High-noise domain labels (prefer specific doctrinal tokens)
    "law", "laws", "legal", "public", "private", "international",
    "criminal", "evidence", "family", "intellectual", "property",
}

_RAG_FOOTER_PATTERNS = [
    r"^printed\s+from\s+oxford\s+law\s+trove\.$",
    r"^printed\s+from\s+oxford\s+law\s+trove.*$",
    r"^subscriber:\s+.*$",
    r"^under\s+the\s+terms\s+of\s+the\s+licence\s+agreement.*$",
    r"^for\s+personal\s+use\s*\(for\s+details\s+see\s+privacy\s+policy\s+and\s+legal\s+notice\)\.?$",
    r"^page\s+\d+\s+of\s+\d+\s*$",
    r"^\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2}\s*$",
    r"^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*$",
    r"^\d{1,2},\s*\d{1,2}:\d{2}\s*$",
    r"^\d+\s*/\s*\d+\s*$",
    r"^\d+\s*/\s*\d+\s*\.\.\..*$",
    r"^copyright\s+.*$",
    r"^created\s+from\s+.*$",
    r"^published\s+online\s+by\s+.*$",
]

def _clean_text_for_rag(text: str) -> str:
    """
    Clean extracted PDF text for retrieval context display/prompting:
    - Remove common platform boilerplate/footers.
    - Drop lines containing CJK characters (often OCR/header artefacts) to avoid mixed-language noise.
    """
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if any(re.match(pat, line, flags=re.IGNORECASE) for pat in _RAG_FOOTER_PATTERNS):
            continue

        # Drop common Oxford/LawTrove download boilerplate.
        if re.search(r"^this\s+content\s+downloaded\s+from\s+.+$", line, flags=re.IGNORECASE):
            continue

        # Drop bare URLs / DOI lines and Oxford Law Trove navigation crumbs.
        if "http://" in line.lower() or "https://" in line.lower():
            continue
        if "doi.org" in line.lower() or line.lower().startswith("doi:"):
            continue

        # Drop lines with CJK characters (Chinese/Japanese/Korean ranges), which are usually artefacts here.
        if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

# Hybrid search weights (base values - can be overridden by query type)
SEMANTIC_WEIGHT = 0.60  # Weight for ChromaDB semantic search (reduced from 0.7)
BM25_WEIGHT = 0.40      # Weight for BM25 keyword search (increased from 0.3)

# ================================================================================
# QUERY-TYPE-SPECIFIC CONFIGURATIONS
# ================================================================================
# These settings optimize retrieval for different question types:
# - PB (Problem Questions): Higher BM25 for precise legal term matching
# - Essays: Balanced semantic/BM25 for broader conceptual retrieval
# - SQE: Higher thresholds for exam-focused accuracy

QUERY_TYPE_RETRIEVAL_CONFIG = {
    # Problem-based questions - keyword precision is critical
    # Legal terms like "breach of contract", "negligent misstatement" must match
    # IMPROVED: Lowered thresholds and increased chunks for better retrieval coverage
    "pb": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.36,  # Lowered from 0.42 for broader retrieval
        "max_per_document": 10  # Increased from 5
    },
    # Word-count-derived PB types (produced by query detection for combined prompts)
    "pb_1500": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.36,  # Lowered from 0.42
        "max_per_document": 10  # Increased from 5
    },
    "pb_2000": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.36,  # Lowered from 0.42
        "max_per_document": 10  # Increased from 5
    },
    "pb_2500": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.35,  # Lowered from 0.44
        "max_per_document": 10  # Increased from 5
    },
    "pb_complex": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.34,  # Lowered from 0.40
        "max_per_document": 11  # Increased from 6
    },

    # Essay questions - need broader conceptual coverage
    # IMPROVED: Lowered thresholds and increased chunks
    "essay": {
        "semantic_weight": 0.60,
        "bm25_weight": 0.40,
        "relevance_threshold": 0.35,  # Lowered from 0.40
        "max_per_document": 10  # Increased from 5
    },
    "essay_1500": {
        "semantic_weight": 0.60,
        "bm25_weight": 0.40,
        "relevance_threshold": 0.36,  # Lowered from 0.42
        "max_per_document": 10  # Increased from 5
    },
    "essay_2000": {
        "semantic_weight": 0.60,
        "bm25_weight": 0.40,
        "relevance_threshold": 0.35,  # Lowered from 0.40
        "max_per_document": 10  # Increased from 5
    },
    "essay_2000_complex": {
        "semantic_weight": 0.58,
        "bm25_weight": 0.42,
        "relevance_threshold": 0.33,  # Lowered from 0.38
        "max_per_document": 11  # Increased from 6
    },
    "essay_3000": {
        "semantic_weight": 0.58,
        "bm25_weight": 0.42,
        "relevance_threshold": 0.33,  # Lowered from 0.38
        "max_per_document": 11  # Increased from 6
    },
    "essay_3000_complex": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.31,  # Lowered from 0.36
        "max_per_document": 12  # Increased from 7
    },
    "essay_4000": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.31,  # Lowered from 0.36
        "max_per_document": 12  # Increased from 7
    },
    "essay_4000_complex": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.30,  # Lowered from 0.35
        "max_per_document": 13  # Increased from 8
    },
    "essay_5000": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.30,  # Lowered from 0.35
        "max_per_document": 13  # Increased from 8
    },
    "essay_5000_complex": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.28,  # Lowered from 0.33
        "max_per_document": 14  # Increased from 9
    },
    "essay_10000": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.27,  # Lowered from 0.32
        "max_per_document": 15  # Increased from 10
    },
    "essay_10000_complex": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.25,  # Lowered from 0.30
        "max_per_document": 17  # Increased from 12
    },

    # SQE notes - need high accuracy
    # IMPROVED: Lowered thresholds and increased chunks
    "sqe1_notes": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.35,  # Lowered from 0.40
        "max_per_document": 13  # Increased from 8
    },
    "sqe2_notes": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.35,  # Lowered from 0.40
        "max_per_document": 13  # Increased from 8
    },
    "sqe_topic": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.36,  # Lowered from 0.42
        "max_per_document": 11  # Increased from 6
    },

    # Advice/Mode C - balance of precision and coverage
    "advice_mode_c": {
        "semantic_weight": 0.55,
        "bm25_weight": 0.45,
        "relevance_threshold": 0.35,  # Lowered from 0.40
        "max_per_document": 10  # Increased from 5
    },

    # General/default
    "general": {
        "semantic_weight": 0.60,
        "bm25_weight": 0.40,
        "relevance_threshold": 0.35,  # Lowered from 0.40
        "max_per_document": 10  # Increased from 5
    },
    "non_legal": {
        "semantic_weight": 0.70,
        "bm25_weight": 0.30,
        "relevance_threshold": 0.30,  # Lowered from 0.35
        "max_per_document": 8  # Increased from 3
    }
}

def get_retrieval_config(query_type: str) -> dict:
    """Get retrieval configuration for a specific query type."""
    return QUERY_TYPE_RETRIEVAL_CONFIG.get(query_type, {
        "semantic_weight": SEMANTIC_WEIGHT,
        "bm25_weight": BM25_WEIGHT,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "max_per_document": MAX_CHUNKS_PER_DOCUMENT
    })


# ================================================================================
# DATA CLASSES
# ================================================================================

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk"""
    document_id: str
    document_name: str
    category: str
    subcategory: str
    chunk_index: int
    total_chunks: int
    document_type: str  # case_law, statutes, articles, textbooks
    page_number: Optional[int] = None
    section_title: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from retrieval with scoring information"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: float  # ChromaDB similarity score
    bm25_score: float     # BM25 keyword score
    category_weight: float  # Category relevance weight
    final_score: float    # Combined weighted score


# ================================================================================
# BM25 IMPLEMENTATION
# ================================================================================

class BM25:
    """
    BM25 (Best Matching 25) implementation for keyword-based retrieval.
    
    BM25 is a ranking function that considers:
    - Term frequency (TF): How often a term appears in a document
    - Inverse document frequency (IDF): How rare a term is across all documents
    - Document length normalization: Accounts for longer documents having more term occurrences
    """
    
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter. Higher = less saturation.
                Controls how TF scaling works. At k1=0, TF doesn't matter.
                At k1→∞, TF scales linearly.
            b: Length normalization parameter (0-1). 
                b=0: No length normalization
                b=1: Full length normalization
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.term_freqs: List[Dict[str, int]] = []  # doc_index -> {term: count}
        self.idf: Dict[str, float] = {}
    
    def fit(self, corpus: List[str]):
        """
        Fit BM25 to a corpus of documents.
        
        Args:
            corpus: List of document texts
        """
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)
        
        # Tokenize and count
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))
            
            # Count document frequencies (how many docs contain each term)
            for term in set(tokens):
                self.doc_freqs[term] += 1
        
        # Calculate average document length
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate IDF for each term
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 scoring."""
        text = text.lower()

        # Collapse common legal reference patterns into single tokens to reduce numeric noise.
        text = re.sub(r"\b(s|ss)\.?\s*(\d+[a-z]?)\b", r"s_\2", text)
        text = re.sub(r"\bsection\s*(\d+[a-z]?)\b", r"s_\1", text)
        text = re.sub(r"\b(art|article)\s*(\d+[a-z]?)\b", r"art_\2", text)
        text = re.sub(r"\b(sch|schedule)\s*(\d+[a-z]?)\b", r"sch_\2", text)
        text = re.sub(r"\b(uksc|ukhl)\s*(\d+)\b", r"\1_\2", text)
        text = re.sub(r"\b(ewca|ewhc)\s*(civ|crim)?\s*(\d+)\b", lambda m: f"{m.group(1)}_{(m.group(2) or '').strip()}_{m.group(3)}".replace("__", "_").strip("_"), text)
        text = re.sub(r"\b(\d+)\s+(ac|wlr|qb|kb|ch|fam)\s+(\d+)\b", r"\2_\3", text)

        # Extract tokens (keep joined tokens like "uksc_11", "s_75", "ewca_civ_123").
        tokens = re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-z]+\b", text)

        return [t for t in tokens if len(t) > 1 and t not in BM25_STOPWORDS]
    
    def score(self, query: str, doc_index: int) -> float:
        """
        Calculate BM25 score for a query against a specific document.
        
        Args:
            query: The search query
            doc_index: Index of the document in the corpus
            
        Returns:
            BM25 score (higher = more relevant)
        """
        query_tokens = self._tokenize(query)
        doc_tf = self.term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        
        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue
            
            tf = doc_tf[term]
            idf = self.idf.get(term, 0)
            
            # BM25 scoring formula
            # Numerator: tf * (k1 + 1)
            # Denominator: tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for a query against all documents.
        
        Args:
            query: The search query
            
        Returns:
            List of scores, one per document
        """
        return [self.score(query, i) for i in range(self.corpus_size)]


# ================================================================================
# RAG SERVICE
# ================================================================================

class RAGService:
    """
    Advanced RAG Service with hybrid search, diversity limiting, and category weighting.
    """
    
    def __init__(self, persist_directory: str = None, use_upgraded_embeddings: bool = False):
        """
        Initialize the RAG service with ChromaDB and BM25.

        Args:
            persist_directory: Path to ChromaDB storage. Defaults to ./chroma_db
            use_upgraded_embeddings: If True and sentence-transformers is installed,
                use BAAI/bge-large-en-v1.5 for higher-quality embeddings.
                Requires a separate collection (law_resources_bge) that must be
                built via migrate_to_bge_embeddings().
        """
        if persist_directory is None:
            persist_directory = os.path.join(os.path.dirname(__file__), 'chroma_db')

        self.persist_directory = persist_directory
        self.use_upgraded_embeddings = use_upgraded_embeddings

        # Auto-download from Hugging Face if chroma_db doesn't exist
        download_chroma_from_huggingface(persist_directory)

        # ChromaDB persistence migration (compatibility):
        # Some older/newer Chroma builds stored `index_metadata.pickle` as a dict.
        # Newer versions expect a `PersistentData` object, otherwise queries crash with:
        #   AttributeError: 'dict' object has no attribute 'dimensionality'
        self._migrate_legacy_index_metadata(self.persist_directory)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Determine which collection + embedding function to use
        self._embedding_fn = None
        collection_name = DEFAULT_COLLECTION_NAME

        if use_upgraded_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_fn = BGEEmbeddingFunction()
                collection_name = UPGRADED_COLLECTION_NAME
                print(f"🧠 Using upgraded BGE embeddings (collection: {collection_name})")
            except Exception as e:
                print(f"⚠️ Failed to load BGE model, falling back to default: {e}")
                self._embedding_fn = None
                collection_name = DEFAULT_COLLECTION_NAME

        # Get or create the collection
        try:
            if self._embedding_fn:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self._embedding_fn
                )
            else:
                self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            # Fallback to create if doesn't exist
            create_kwargs = {"name": collection_name, "metadata": {"hnsw:space": "cosine"}}
            if self._embedding_fn:
                create_kwargs["embedding_function"] = self._embedding_fn
            self.collection = self.client.get_or_create_collection(**create_kwargs)

        # Initialize BM25 (will be populated when needed)
        self.bm25: Optional[BM25] = None
        self.bm25_chunk_ids: List[str] = []  # Maps BM25 index to chunk ID

        print(f"📚 RAG Service initialized with {self.collection.count()} chunks")

    def _migrate_legacy_index_metadata(self, persist_directory: str) -> None:
        """
        Migrate legacy `index_metadata.pickle` payloads stored as dicts into the
        an attribute-bearing object expected by current Chroma versions.

        Newer Chroma code does `self._persist_data = pickle.load(...)` and then accesses
        attributes like `self._persist_data.dimensionality`. Older persisted files may
        contain a plain dict, which breaks with:
          AttributeError: 'dict' object has no attribute 'dimensionality'
        """
        from types import SimpleNamespace

        try:
            for root, _dirs, files in os.walk(persist_directory):
                if "index_metadata.pickle" not in files:
                    continue
                path = os.path.join(root, "index_metadata.pickle")
                try:
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                except Exception:
                    continue

                if not isinstance(obj, dict):
                    continue

                # Convert dict -> object with attributes (pickle-safe, stdlib type).
                # Ensure `max_seq_id` exists for Chroma's internal migration path.
                if obj.get("max_seq_id") is None:
                    obj["max_seq_id"] = 0
                pd = SimpleNamespace(**obj)

                tmp_path = path + ".tmp"
                try:
                    with open(tmp_path, "wb") as f:
                        pickle.dump(pd, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, path)
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            print(f"RAG metadata migration warning: {e}")
    
    # ============================================================================
    # DOCUMENT TYPE DETECTION
    # ============================================================================
    
    def detect_document_type(self, text: str, filename: str = "") -> str:
        """
        Detect the type of legal document for semantic chunking.
        
        This is important because different document types have different
        optimal chunk sizes:
        - Cases (2000 chars): Need more context to capture holdings and reasoning
        - Statutes (1000 chars): Dense legal text, shorter chunks work better
        - Articles (1800 chars): Academic context benefits from longer chunks
        - Textbooks (1500 chars): Standard academic text
        
        Args:
            text: Document text content
            filename: Original filename for additional hints
            
        Returns:
            Document type: "case_law", "statutes", "articles", "textbooks", or "default"
        """
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Score each document type
        scores = {doc_type: 0 for doc_type in DOCUMENT_TYPE_PATTERNS}
        
        for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                scores[doc_type] += matches
        
        # Also check filename for hints
        if 'case' in filename_lower or 'judgment' in filename_lower:
            scores['case_law'] += 10
        if 'act' in filename_lower or 'statute' in filename_lower or 'regulation' in filename_lower:
            scores['statutes'] += 10
        if 'journal' in filename_lower or 'article' in filename_lower or 'review' in filename_lower:
            scores['articles'] += 10
        if 'textbook' in filename_lower or 'chapter' in filename_lower or 'edition' in filename_lower:
            scores['textbooks'] += 10
        
        # Return the highest scoring type
        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type
        return "default"
    
    def get_chunk_size(self, document_type: str, category: str = "") -> int:
        """
        Get the optimal chunk size for a document type, optionally tuned by legal category.

        Smarter chunk sizing explained:
        --------------------------------
        Fixed chunk sizes (like 1500 chars) don't account for document structure.
        Different legal documents have different information densities:

        1. CASE LAW (2000 chars):
           - Cases contain holdings, ratio decidendi, and obiter dicta
           - Judicial reasoning needs context to be understood
           - Cutting mid-reasoning loses important connections
           - Example: Montgomery v Lanarkshire needs surrounding context

        2. STATUTES (1000 chars):
           - Statutes are very dense - each word is carefully chosen
           - Sections are often self-contained
           - Shorter chunks = more precise retrieval
           - Example: Mental Capacity Act s.1 is self-explanatory

        3. JOURNAL ARTICLES (1800 chars):
           - Academic arguments build over paragraphs
           - Authors make connected points
           - Need enough context for the argument flow

        4. TEXTBOOKS (1500 chars):
           - Standard academic prose
           - Explanatory text with examples
           - Good balance of context and specificity

        Category overrides allow fine-tuning per legal area (e.g. EU law cases
        get larger chunks because EU judgments are longer).

        Args:
            document_type: Type of document
            category: Legal category (e.g. "EU Law", "Criminal Law") for fine-tuning

        Returns:
            Optimal chunk size in characters
        """
        # Check category-specific override first
        if category:
            cat_lower = category.strip().lower()
            cat_overrides = CATEGORY_CHUNK_OVERRIDES.get(cat_lower, {})
            if document_type in cat_overrides:
                return cat_overrides[document_type]
        return CHUNK_STRATEGIES.get(document_type, CHUNK_STRATEGIES["default"])
    
    # ============================================================================
    # DOCUMENT CHUNKING
    # ============================================================================
    
    def chunk_document(self, text: str, document_type: str = "default", category: str = "") -> List[str]:
        """
        Split a document into semantic chunks based on document type.

        Uses intelligent boundaries:
        - Paragraph breaks (double newlines)
        - Section markers (Part I, Section 1, etc.)
        - Case structure (Holdings, Ratio, etc.)

        Args:
            text: Full document text
            document_type: Type of document for chunk sizing
            category: Legal category for fine-tuned chunk sizes

        Returns:
            List of text chunks
        """
        chunk_size = self.get_chunk_size(document_type, category)
        overlap = chunk_size // 5  # 20% overlap
        
        # First, try to split by semantic boundaries
        chunks = []
        
        # Define semantic break points
        break_patterns = [
            r'\n\n+',  # Paragraph breaks
            r'\n(?=Part\s+[IVX]+:)',  # Part headings
            r'\n(?=Section\s+\d+)',  # Section headings
            r'\n(?=Article\s+\d+)',  # Article headings
            r'\n(?=\d+\.\s+)',  # Numbered sections
            r'\n(?=[A-Z][A-Z\s]+:)',  # ALL CAPS headings
        ]
        
        # Try semantic splitting first
        segments = [text]
        for pattern in break_patterns:
            new_segments = []
            for segment in segments:
                parts = re.split(pattern, segment)
                new_segments.extend([p.strip() for p in parts if p.strip()])
            segments = new_segments
        
        # Now combine segments into chunks of appropriate size
        current_chunk = ""
        for segment in segments:
            if len(current_chunk) + len(segment) <= chunk_size:
                current_chunk += "\n\n" + segment if current_chunk else segment
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If segment itself is too long, split it further
                if len(segment) > chunk_size:
                    # Split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', segment)
                    sub_chunk = ""
                    for sentence in sentences:
                        if len(sub_chunk) + len(sentence) <= chunk_size:
                            sub_chunk += " " + sentence if sub_chunk else sentence
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = sentence
                    if sub_chunk:
                        current_chunk = sub_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = segment
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap for context continuity
        if len(chunks) > 1 and overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Add end of previous chunk as context
                    prev_overlap = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                    chunk = prev_overlap + " ... " + chunk
                overlapped_chunks.append(chunk)
            chunks = overlapped_chunks
        
        return chunks
    
    # ============================================================================
    # DOCUMENT PARSING
    # ============================================================================
    
    def parse_pdf(self, file_path: str) -> str:
        """Parse text from a PDF file."""
        if not PDF_AVAILABLE:
            return ""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return ""
    
    def parse_docx(self, file_path: str) -> str:
        """Parse text from a DOCX file."""
        if not DOCX_AVAILABLE:
            return ""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            return text.strip()
        except Exception as e:
            print(f"Error parsing DOCX {file_path}: {e}")
            return ""
    
    def parse_txt(self, file_path: str) -> str:
        """Parse text from a TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error parsing TXT {file_path}: {e}")
            return ""
    
    def parse_document(self, file_path: str) -> str:
        """Parse text from any supported document type."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self.parse_pdf(file_path)
        elif ext == '.docx':
            return self.parse_docx(file_path)
        elif ext == '.txt':
            return self.parse_txt(file_path)
        return ""
    
    # ============================================================================
    # INDEXING
    # ============================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        try:
            total_chunks = self.collection.count()
            return {
                "total_chunks": total_chunks,
                "status": "active" if total_chunks > 0 else "empty"
            }
        except Exception as e:
            # We don't want to crash the whole app if stats fail
            print(f"Error getting RAG stats: {e}")
            return {"total_chunks": 0, "status": "error", "error": str(e)}
            
    def index_documents(
        self, 
        directory: str, 
        progress_callback: Callable[[int, str], None] = None,
        rebuild_bm25: bool = True
    ) -> Dict[str, int]:
        """
        Index all documents in a directory with semantic chunking.
        
        Args:
            directory: Path to directory containing documents
            progress_callback: Optional callback for progress updates
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'processed': 0,
            'chunks': 0,
            'errors': 0,
            'skipped': 0,
            'type_stats': defaultdict(int)
        }
        
        # Walk through directory
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.pdf', '.docx', '.txt')):
                    all_files.append(os.path.join(root, file))
        
        for i, file_path in enumerate(all_files):
            try:
                abs_path = os.path.abspath(file_path)

                # Avoid re-indexing the same file across runs.
                # NOTE: Older versions used a non-stable `hash(file_path)` doc_id, which could duplicate chunks.
                try:
                    existing = self.collection.get(where={"file_path": abs_path}, limit=1, include=["ids"])
                    if existing and existing.get("ids"):
                        stats["skipped"] += 1
                        continue
                except Exception:
                    # If the backend doesn't support where/limit in this environment, fall back to indexing.
                    pass

                # Parse document
                text = self.parse_document(abs_path)
                if not text or len(text) < 100:
                    stats['skipped'] += 1
                    continue
                
                # Detect document type
                filename = os.path.basename(file_path)
                doc_type = self.detect_document_type(text, filename)
                stats['type_stats'][doc_type] += 1

                # Get category/subcategory from path BEFORE chunking so we can
                # use category-specific chunk sizes.
                # Prefer classifying relative to the global law resources root so that indexing a single
                # subfolder (via add_to_index.py) still assigns the correct top-level category.
                resources_root = os.path.join(os.path.dirname(__file__), "Law resouces  copy 2")
                try:
                    abs_resources_root = os.path.abspath(resources_root)
                    if abs_path.startswith(abs_resources_root + os.sep) or abs_path == abs_resources_root:
                        rel_path = os.path.relpath(abs_path, abs_resources_root)
                    else:
                        rel_path = os.path.relpath(abs_path, directory)
                except Exception:
                    rel_path = os.path.relpath(abs_path, directory)

                parts = [p for p in rel_path.split(os.sep) if p]
                # Expected shapes:
                # - From resources root: ["Category", "file.pdf"] or ["Category", "Subcategory", "file.pdf"]
                # - From a standalone folder: ["file.pdf"] or ["Subfolder", "file.pdf"]
                category = parts[0] if len(parts) >= 2 else os.path.basename(os.path.abspath(directory))
                subcategory = parts[1] if len(parts) > 2 else ""

                # Some users store additional topic folders inside "Law resources Extra".
                # Treat the immediate child folder as the real category to avoid burying useful materials
                # under a generic "Law resources Extra" label (which hurts category weighting).
                if category.strip().lower() == "law resources extra" and len(parts) > 2:
                    category = parts[1]
                    subcategory = parts[2] if len(parts) > 3 else ""

                # Chunk document (with category-aware sizing)
                chunks = self.chunk_document(text, doc_type, category)
                
                # Generate document ID
                doc_id = "doc_" + hashlib.sha1(abs_path.encode("utf-8", errors="ignore")).hexdigest()[:16]
                
                # Add chunks to ChromaDB
                for j, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{j}"
                    metadata = {
                        'document_id': doc_id,
                        'document_name': filename,
                        'category': category,
                        'subcategory': subcategory,
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'document_type': doc_type,
                        '_type': doc_type,
                        'file_path': abs_path
                    }
                    
                    # Upsert to avoid duplicates
                    self.collection.upsert(
                        ids=[chunk_id],
                        documents=[chunk],
                        metadatas=[metadata]
                    )
                    stats['chunks'] += 1
                
                stats['processed'] += 1
                
                if progress_callback:
                    progress_callback(stats['processed'], filename)
                    
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
                stats['errors'] += 1
        
        # Rebuild BM25 index after adding documents (optional; full rebuild can be slow on large DBs)
        if rebuild_bm25:
            self._rebuild_bm25_index()
        else:
            # Mark BM25 as stale; it will be rebuilt lazily on the next BM25 query.
            self.bm25 = None
            self.bm25_chunk_ids = []
        
        # Rebuild citation graph after indexing
        try:
            self.build_citation_graph()
            print(f"🔗 Citation graph built with {len(getattr(self, '_citation_graph', {}))} entries")
        except Exception as e:
            print(f"⚠️ Citation graph rebuild skipped: {e}")

        return dict(stats)

    def migrate_to_bge_embeddings(self, progress_callback: Callable = None) -> Dict[str, int]:
        """
        Migrate the existing law_resources collection to the upgraded BGE
        embedding model. Creates a new collection (law_resources_bge) and
        re-embeds all existing chunks.

        This is a one-time operation. After migration, set use_upgraded_embeddings=True
        when constructing RAGService to use the new collection.

        Returns:
            Stats dict with 'total', 'migrated', 'errors' counts
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for BGE migration. "
                "Install with: pip install sentence-transformers"
            )

        bge_fn = BGEEmbeddingFunction()

        # Get or create the BGE collection
        bge_collection = self.client.get_or_create_collection(
            name=UPGRADED_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=bge_fn
        )

        # Read all data from the current (default) collection
        try:
            source = self.client.get_collection(name=DEFAULT_COLLECTION_NAME)
        except Exception:
            print("⚠️ Source collection not found. Nothing to migrate.")
            return {"total": 0, "migrated": 0, "errors": 0}

        total = source.count()
        stats = {"total": total, "migrated": 0, "errors": 0}

        if total == 0:
            print("ℹ️ Source collection is empty. Nothing to migrate.")
            return stats

        batch_size = 100
        for offset in range(0, total, batch_size):
            try:
                batch = source.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                if not batch["ids"]:
                    break

                bge_collection.upsert(
                    ids=batch["ids"],
                    documents=batch["documents"],
                    metadatas=batch["metadatas"]
                )
                stats["migrated"] += len(batch["ids"])

                if progress_callback:
                    progress_callback(stats["migrated"], total)

            except Exception as e:
                print(f"⚠️ Migration batch error at offset {offset}: {e}")
                stats["errors"] += 1

        print(f"✅ BGE migration complete: {stats['migrated']}/{stats['total']} chunks migrated")
        return stats

    # ============================================================================
    # BM25 INDEX MANAGEMENT
    # ============================================================================
    
    def _rebuild_bm25_index(self):
        """Rebuild the BM25 index from ChromaDB data."""
        print("🔄 Rebuilding BM25 index...")
        
        # Get all documents from ChromaDB
        result = self.collection.get(include=['documents'])
        
        if not result['documents']:
            print("⚠️ No documents in ChromaDB to build BM25 index")
            return
        
        # Build BM25 index
        self.bm25 = BM25()
        self.bm25.fit(result['documents'])
        self.bm25_chunk_ids = result['ids']
        
        print(f"✅ BM25 index built with {len(self.bm25_chunk_ids)} chunks")
    
    def _ensure_bm25_index(self):
        """Ensure BM25 index is built."""
        if self.bm25 is None or not self.bm25_chunk_ids:
            self._rebuild_bm25_index()
    
    # ============================================================================
    # HYBRID RETRIEVAL
    # ============================================================================
    
    def _get_semantic_results(
        self, 
        query: str, 
        n_results: int = 50
    ) -> Dict[str, Tuple[float, Dict]]:
        """
        Get semantic search results from ChromaDB.
        
        Returns:
            Dict mapping chunk_id to (score, metadata)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        semantic_results = {}
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # ChromaDB returns distance, convert to similarity
                # For cosine distance, similarity = 1 - distance
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                metadata = results['metadatas'][0][i]
                content = results['documents'][0][i]
                
                semantic_results[chunk_id] = {
                    'score': similarity,
                    'content': content,
                    'metadata': metadata
                }
        
        return semantic_results

    def _doc_type_weight(self, query_type: Optional[str], metadata: Dict[str, Any]) -> float:
        """
        Weight sources by document type.

        Goal:
        - Problem questions: prefer primary authority (cases/statutes) and regulator guidance.
        - Essays: keep broader mix (do not heavily penalize journals/textbooks).
        """
        qt = (query_type or "").lower()
        doc_type = (metadata or {}).get("document_type") or (metadata or {}).get("_type") or ""
        doc_type = str(doc_type).lower()

        doc_name = str((metadata or {}).get("document_name") or "").lower()
        category = str((metadata or {}).get("category") or "").lower()

        # Heuristic for "guidance/regulator" docs (often not tagged as statutes).
        is_guidance = any(k in doc_name for k in ["code", "guidance", "practice", "protocol", "policy", "cma", "gmc", "bma"]) or \
            any(k in category for k in ["regulator", "guidance", "code"])

        is_problem = qt.startswith("pb") or qt in {"pb", "problem", "problem_question"}
        if is_problem:
            # Heuristic for academic/journal-like materials that may not be tagged as "articles".
            is_academic = any(k in doc_name for k in ["journal", "law review", "law rev", "lqr", "clj", "cmlr", "eu law live"]) or \
                any(k in category for k in ["journal", "journals", "review", "articles"])

            # Heuristic for quality textbooks (Law Trove, OUP, Cambridge, Hart, etc.)
            is_quality_textbook = any(k in doc_name for k in ["law trove", "oxford", "cambridge", "hart", "sweet & maxwell", "sweet and maxwell", "lexisnexis"]) or \
                any(k in category for k in ["textbook", "textbooks"])

            if doc_type in ("case_law", "case", "cases"):
                return 1.40
            if doc_type in ("statutes", "statute", "legislation"):
                return 1.35
            if is_guidance:
                return 1.25
            if is_academic:
                return 0.80  # Increased from 0.70 - journals can provide good analysis
            # IMPROVED: Textbooks now get higher weight (0.95) as they often contain
            # case extracts and are valuable when primary cases aren't indexed
            if is_quality_textbook:
                return 0.98  # Quality textbooks get near-parity with primary sources
            if doc_type in ("textbooks", "default"):
                return 0.95  # Increased from 0.85 - textbooks are valuable for PB
            if doc_type in ("articles", "journal", "journals"):
                return 0.80  # Increased from 0.70
            return 0.90  # Increased from 0.85

        # Essays: broad coverage; slight preference for primary sources but keep journals usable.
        if doc_type in ("case_law", "case", "cases"):
            return 1.05
        if doc_type in ("statutes", "statute", "legislation"):
            return 1.05
        if is_guidance:
            return 1.00
        return 1.00

    def _chunk_quality_multiplier(self, content: str) -> float:
        """
        Penalize obviously truncated/garbled chunks so they don't waste retrieval slots.
        """
        cleaned = _clean_text_for_rag(content or "")
        if not cleaned:
            return 0.0

        # Very short chunks are usually headers/artefacts after cleaning.
        if len(cleaned) < 120:
            return 0.05
        if len(cleaned) < 220:
            return 0.35

        # Truncation markers / mid-sentence clipping.
        if cleaned.rstrip().endswith("..."):
            return 0.60
        if re.search(r"\b(?:legal r\.{0,2}|the legal r)\b", cleaned.lower()):
            return 0.50

        # If it's mostly non-letters, it's likely OCR noise.
        letters = sum(ch.isalpha() for ch in cleaned)
        ratio = letters / max(1, len(cleaned))
        if ratio < 0.35:
            return 0.65
        return 1.0
    
    def _get_bm25_results(
        self, 
        query: str, 
        n_results: int = 50
    ) -> Dict[str, float]:
        """
        Get BM25 keyword search results.
        
        Returns:
            Dict mapping chunk_id to BM25 score
        """
        self._ensure_bm25_index()
        
        if self.bm25 is None:
            return {}
        
        # Get all BM25 scores
        scores = self.bm25.get_scores(query)
        
        # Create (chunk_id, score) pairs and sort by score
        scored = [(self.bm25_chunk_ids[i], score) for i, score in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores to 0-1 range
        max_score = max(s for _, s in scored) if scored else 1
        if max_score == 0:
            max_score = 1
        
        return {chunk_id: score / max_score for chunk_id, score in scored[:n_results]}
    
    def _get_category_weight(self, query: str, category: str) -> float:
        """
        Get category weight based on query relevance.
        
        Args:
            query: The search query
            category: Document category
            
        Returns:
            Weight multiplier (default 1.0)
        """
        query_lower = query.lower()
        category_lower = category.lower()

        query_tokens = {t for t in re.findall(r"[a-z]+", query_lower) if t not in CATEGORY_MATCH_STOPWORDS}
        category_tokens = {t for t in re.findall(r"[a-z]+", category_lower) if t not in CATEGORY_MATCH_STOPWORDS}

        weight = 1.0

        # Direct category token overlap (only on meaningful tokens, not generic words like "law").
        if category_tokens and (query_tokens & category_tokens):
            weight = max(weight, 1.5)

        # Domain gating: when the query clearly belongs to a domain, downweight unrelated categories.
        # NOTE: Domain detection uses keyword hits. For very short keywords ("ai", "ip", "cma"),
        # we require word-boundary matches to avoid accidental substring hits.
        domain_keywords = {
            "medical": [
                "medical", "medicine", "clinical",
                "hospital", "nhs", "gmc", "patient", "doctor",
                "anaesthetist", "anesthetist", "surgeon", "operation", "surgery",
                "treatment", "malpractice",
            ],
            "criminal": [
                "criminal", "cps", "prosecution", "prosecut", "charge",
                "manslaughter", "gross negligence", "gross negligence manslaughter",
                "adomako", "sellu", "misra", "rose", "rudling",
                "offences against the person act", "oapa", "cunningham", "recklessness",
                "actus reus", "mens rea",
            ],
            "defamation": [
                "defamation", "libel", "slander",
                "defamation act 2013", "serious harm",
                "section 1 defamation act", "s 1 defamation act",
                "truth defence", "honest opinion",
                "publication on matter of public interest",
                "operators of website", "section 5", "s 5",
                "privilege", "qualified privilege", "absolute privilege",
                "single publication rule",
                "lachaux", "jameel", "thornton", "economou",
                "reynolds", "responsible journalism",
            ],
            "merger_control_uk": [
                "enterprise act 2002", "ea 2002", "merger control", "relevant merger situation",
                "phase 1", "phase i", "phase 2", "phase ii",
                "substantial lessening of competition", "slc",
                "undertakings in lieu", "uil", "uils",
                "share of supply", "turnover test",
                "initial enforcement order", "ieo",
                "reference", "call in", "call-in", "killer acquisition", "killer acquisitions",
                "remedy", "remedies", "divestiture", "structural remedy", "behavioural remedy", "behavioral remedy",
                "competition and markets authority", "cma",
            ],
            "employment_discrimination": [
                "equality act 2010", "eqa 2010",
                "direct discrimination", "indirect discrimination",
                "pcp", "provision criterion or practice",
                "objective justification", "legitimate aim", "proportionate means",
                "harassment", "victimisation", "victimization",
                "burden of proof", "section 136", "s 136",
                "reasonable adjustments", "disability",
                "equal pay",
            ],
            "private_international": [
                # Core concepts
                "private international law", "conflict of laws", "conflicts of law", "pil",
                "dicey", "dicey morris collins", "dicey morris and collins",
                "jurisdiction", "choice of law", "proper law", "applicable law",
                "connecting factor", "connecting factors",

                # Domicile
                "domicile", "domicile of origin", "domicile of choice", "domicile of dependence",
                "domicil", "lex domicilii", "personal law",
                "animus manendi", "factum", "intention", "permanent residence",
                "revival", "revival doctrine", "tenacious", "sticky domicile",
                "udny v udny", "udny", "barlow clowes", "barlow clowes v henwood",
                "winans v attorney general", "winans", "fuld", "in re fuld",
                "irc v bullock", "bullock", "plummer v irc",
                "non-dom", "non dom", "non-domiciled", "remittance basis",

                # Habitual Residence
                "habitual residence", "habitually resident", "centre of interests",
                "factual residence", "integration", "settled purpose",
                "mark v mark", "marinos v marinos", "swaddling",
                "a v a", "re j", "re m", "re lc",

                # Jurisdiction (Civil)
                "rome i", "rome ii", "brussels", "brussels i", "brussels ia", "brussels i bis",
                "brussels regulation", "recast brussels regulation",
                "lugano", "lugano convention", "hague convention",
                "service out", "service out of jurisdiction", "cpr 6.36", "cpr pd 6b",
                "gateway", "gateways", "permission to serve out",
                "forum conveniens", "forum non conveniens", "natural forum",
                "spiliada", "spiliada maritime", "appropriate forum",
                "lis pendens", "related actions", "article 29", "article 30",
                "exclusive jurisdiction", "jurisdiction clause", "jurisdiction agreement",
                "prorogation", "derogation", "asymmetric jurisdiction clause",
                "anti-suit injunction", "anti suit injunction", "anti-anti-suit", "anti anti-suit",
                "comity", "restraining foreign proceedings",
                "owusu v jackson", "owusu", "turner v grovit",

                # Recognition and Enforcement of Foreign Judgments
                "recognition", "enforcement", "recognition and enforcement",
                "foreign judgment", "foreign judgments", "overseas judgment",
                "common law enforcement", "enforcement at common law",
                "final and conclusive", "fixed sum", "debt on judgment",
                "international jurisdiction", "indirect jurisdiction",
                "presence", "submission", "residence basis", "tag jurisdiction",
                "adams v cape industries", "adams v cape", "cape industries",
                "schibsby v westenholz", "emanuel v symon",
                "section 32", "s 32", "cjja 1982", "civil jurisdiction and judgments act",
                "arbitration clause defence", "breach of arbitration agreement",

                # Fraud defence
                "fraud defence", "fraud defense", "abouloff", "abouloff v oppenheimer",
                "owens bank v bracco", "owens bank", "bracco",
                "res judicata", "issue estoppel", "cause of action estoppel",

                # Choice of Law - Contract
                "rome i regulation", "regulation 593/2008",
                "characteristic performance", "article 4 rome i",
                "overriding mandatory provisions", "article 9", "public policy",
                "party autonomy", "freedom of choice", "express choice", "implied choice",
                "closest connection",

                # Choice of Law - Tort
                "rome ii regulation", "regulation 864/2007",
                "lex loci delicti", "place of damage", "article 4 rome ii",
                "product liability", "unfair competition", "environmental damage",
                "culpa in contrahendo",

                # Family PIL
                "matrimonial domicile", "divorce jurisdiction", "brussels ii", "brussels iia",
                "maintenance", "maintenance regulation", "child abduction",
                "hague abduction convention", "hague 1980",
                "wrongful removal", "wrongful retention", "habitual residence child",

                # Succession PIL
                "succession regulation", "eu succession regulation", "regulation 650/2012",
                "last habitual residence", "professio juris", "choice of law succession",
                "forced heirship", "testamentary freedom",
            ],
            "ai_techlaw": [
                "artificial intelligence", "ai", "machine learning", "ml",
                "large language model", "llm", "generative ai", "foundation model",
                "training data", "model training",
                "robotics", "autonomous", "algorithmic", "algorithm",
                "copyright", "text and data mining", "tdm",
                "openai", "anthropic", "stability ai", "cohere",
                "bias", "fairness", "accountability",
                "ai act", "ai regulation",
            ],
            "cyber_cma": [
                "computer misuse act", "computer misuse act 1990", "cma 1990",
                "unauthorised access", "unauthorized access", "unauthorised", "unauthorized",
                "authorisation", "authorization", "authorised", "authorized",
                "section 1", "s 1", "s.1", "section 3", "s 3", "s.3",
                "section 3a", "s 3a", "s.3a", "tools offence", "article for use",
                "section 3za", "s 3za", "s.3za", "critical national infrastructure", "cni",
                "section 4", "s 4", "s.4", "significant link", "jurisdiction",
                "ddos", "denial of service", "distributed denial of service",
                "scraping", "web scraping", "terms of service", "tos",
                "ethical hacking", "security research", "penetration testing", "bug bounty",
                "credential stuffing", "brute force", "password guessing",
                "gold and schifreen", "schifreen", "lennon",
            ],
            "consumer_cra": [
                # Core labels
                "consumer rights act", "cra 2015", "consumer rights act 2015",
                "consumer", "consumer law", "trader", "consumer contract",
                # Digital content
                "digital content", "data produced and supplied in digital form",
                "s 34", "section 34", "s.34", "satisfactory quality",
                "s 42", "section 42", "s.42",
                "s 43", "section 43", "s.43", "repair or replacement",
                "s 44", "section 44", "s.44", "price reduction",
                "s 45", "section 45", "s.45", "right to supply digital content",
                "s 46", "section 46", "s.46", "damage to device", "damage to other digital content",
                "paid for either directly or indirectly", "paid directly or indirectly",
                # Goods remedies ladder
                "short-term right to reject", "right to reject", "final right to reject",
                "s 20", "section 20", "s.20",
                "s 22", "section 22", "s.22",
                "s 23", "section 23", "s.23",
                "s 24", "section 24", "s.24",
                # Goods with digital elements
                "update", "software update", "digital elements", "smart", "connected",
                # Regulator guidance
                "cma 37", "competition and markets authority", "unfair contract terms guidance",
            ],
            "media_privacy": [
                # Core labels
                "media", "press", "newspaper", "journalist", "publication", "injunction",
                "super-injunction", "super injunction",
                # Tort name / evolution
                "misuse of private information", "mpi", "breach of confidence", "confidence",
                "invasion of privacy",
                # Human rights framing
                "human rights act", "hra", "section 12", "s 12",
                "article 8", "art 8", "private life", "family life",
                "article 10", "art 10", "freedom of expression",
                "public interest", "reasonable expectation of privacy",
                # Key authorities
                "wainwright", "coco v", "coco v a n clark", "coco v clark",
                "campbell v mgn", "campbell", "mgn",
                "re s", "in re s",
                "cream holdings", "american cyanamid", "cyanamid",
                "pjs", "pjs v", "mosley", "douglas v hello", "murray v express", "vidal-hall",
                "hypocrisy", "celebrity", "anonymity",
            ],
            "competition": [
                # Core concepts
                "competition", "antitrust", "cartel", "cartels", "dominance", "dominant", "abuse",
                "market definition", "relevant market", "ssnip", "ssndq", "hypothetical monopolist",
                "concerted practice", "agreement", "information exchange", "price fixing", "price-fixing",
                "refusal to supply", "refusal to deal", "essential facility", "essential facilities",
                "market power", "monopoly", "monopolisation", "monopolization", "oligopoly",
                "horizontal agreement", "vertical agreement", "hub and spoke", "hub-and-spoke",
                "bid rigging", "bid-rigging", "market allocation", "output restriction",

                # US Antitrust
                "sherman act", "section 1", "section 2", "sherman act section 1", "sherman act section 2",
                "clayton act", "ftc act", "federal trade commission", "ftc", "doj antitrust",
                "rule of reason", "per se", "per se rule", "quick look", "ancillary restraints",
                "consumer welfare", "consumer welfare standard", "allocative efficiency",
                "chicago school", "harvard school", "structuralism", "structuralist",
                "robert bork", "bork", "antitrust paradox", "posner", "hovenkamp", "areeda", "turner",
                "standard oil", "alcoa", "northern pacific", "topco", "continental tv", "sylvania",
                "leegin", "ohio v american express", "amex", "ncaa v alston", "apple v pepper",
                "brooke group", "predatory pricing", "recoupment", "matsushita", "weyerhaeuser",
                "trinko", "linkline", "price squeeze", "monopoly leveraging",
                "horizontal merger guidelines", "vertical merger guidelines", "hhi",
                "herfindahl-hirschman", "herfindahl hirschman", "concentration ratio",
                "unilateral effects", "coordinated effects", "upward pricing pressure", "guppi",
                "neo-brandeisian", "neo brandeisian", "new brandeis", "lina khan", "tim wu",
                "hipster antitrust", "anticompetitive harm", "procompetitive justification",

                # EU Competition Law
                "article 101", "article 102", "art 101", "art 102", "tfeu",
                "european commission", "dg comp", "general court", "court of justice",
                "object or effect", "restriction by object", "restriction by effect",
                "appreciable effect", "de minimis", "effect on trade", "inter-state trade",
                "abuse of dominance", "exploitative abuse", "exclusionary abuse",
                "margin squeeze", "telia", "teliasonera", "predatory pricing",
                "united brands", "hoffmann-la roche", "akzo", "intel", "post danmark",
                "google shopping", "google android", "google adsense", "microsoft",
                "bronner", "commercial solvents", "magill", "ims health", "oscar bronner",
                "collective dominance", "airtours", "impala",
                "vertical block exemption", "vber", "vertical restraints", "vertical guidelines",
                "selective distribution", "exclusive distribution", "exclusive dealing",
                "resale price maintenance", "rpm", "territorial restrictions", "geo-blocking",
                "horizontal block exemption", "hber", "specialisation", "r&d agreements",
                "state aid", "article 107", "article 108", "general block exemption", "gber",
                "eu merger regulation", "eumr", "significally impede effective competition", "siec",
                "merger control", "notification threshold", "concentrative joint venture",

                # UK Competition Law
                "chapter i", "chapter ii", "competition act 1998", "ca 1998",
                "enterprise act 2002", "ea 2002", "market investigation", "market study",
                "substantial lessening of competition", "slc", "slc test",
                "cma", "competition and markets authority", "ofcom", "ofgem", "ofwat", "fca",
                "merger control uk", "share of supply", "turnover test", "public interest",
                "dmcc", "digital markets competition and consumers act", "dmcca",
                "digital markets unit", "dmu", "strategic market status", "sms",
                "conduct requirements", "pro-competition interventions", "pci",
                "concurrent powers", "sectoral regulators", "primacy",

                # Digital Markets & Platforms
                "digital markets", "digital markets act", "dma", "digital services act", "dsa",
                "gatekeeper", "gatekeepers", "core platform service", "cps",
                "platform", "platforms", "ecosystem", "digital ecosystem", "multi-sided market",
                "two-sided market", "network effects", "indirect network effects", "direct network effects",
                "switching costs", "lock-in", "data portability", "interoperability",
                "zero-price", "zero price", "attention market", "data as currency",
                "self-preferencing", "self preferencing", "ranking", "search bias",
                "killer acquisition", "killer acquisitions", "nascent competition", "nascent competitor",
                "conglomerate merger", "conglomerate effects", "portfolio effects",
                "tying", "bundling", "technical tying", "contractual tying",
                "most favoured nation", "mfn", "parity clause", "parity clauses",
                "amazon", "apple", "google", "meta", "facebook", "microsoft", "alphabet",
                "big tech", "tech giants", "faang", "gafa", "digital conglomerates",
                "app store", "play store", "default", "pre-installation",

                # Comparative Competition Law
                "comparative competition", "comparative antitrust", "transatlantic",
                "convergence", "divergence", "regulatory competition", "extraterritoriality",
                "effects doctrine", "effects-based", "form-based", "formalism",
                "more economic approach", "mea", "consumer harm", "total welfare",
                "efficiency", "efficiencies", "efficiency defence", "efficiency defense",
                "fairness", "ordo-liberalism", "ordoliberalism", "freiburg school",
                "small and medium enterprises", "sme", "competitors", "trading partners",
                "contestable markets", "barriers to entry", "minimum efficient scale",

                # Scholars & Commentary
                "orbach", "chirita", "hovenkamp", "wils", "whish", "bailey", "jones", "sufrin",
                "easterbrook", "kaplow", "shapiro", "carl shapiro", "hal varian",
                "tirole", "rochet", "wright", "first", "crane", "kovacic",
                "antitrust law", "competition policy", "industrial organization",

                # SEP / FRAND interface keywords (competition framing)
                "standard essential", "standard-essential", "sep", "seps", "frand",
                "standard setting", "standard-setting", "sso", "willing licensee",
                "hold-up", "hold up", "hold-out", "hold out", "royalty stacking",
                "huawei v zte", "unwired planet", "injunction", "injunctions",
            ],
            "public_international": [
                "public international law", "international law", "un charter", "united nations charter",
                "article 2(4)", "art 2(4)", "use of force", "threat or use of force", "article 51", "self-defence",
                "chapter vii", "security council", "p5", "veto", "collective security",
                "icj", "international court of justice", "nicaragua", "kosovo", "iraq",
                "humanitarian intervention", "responsibility to protect", "r2p",
                "vienna convention on diplomatic relations", "vcdr", "diplomatic immunity", "persona non grata",
                "diplomatic bag", "article 22", "article 27", "article 29", "article 31",
                "state immunity", "sovereign immunity",
            ],
            "ip": [
                "intellectual property", "ip", "patent", "patents", "patents act 1977", "pa 1977",
                "epc", "european patent convention", "article 56", "inventive step", "non-obvious", "non obvious",
                "obviousness", "windsurfing", "pozzoli", "skilled person", "person skilled in the art",
                "common general knowledge", "obvious to try",
                # SEP / FRAND licensing (IP framing)
                "standard essential", "standard-essential", "standard essential patent", "standard-essential patent",
                "sep", "seps", "frand", "fair reasonable and non-discriminatory", "fair, reasonable and non-discriminatory",
                "standard setting", "standard-setting", "sso", "licensing", "licence", "portfolio",
                "unwired planet", "huawei v zte", "injunction", "injunctions",
                # Trade marks / functions / dilution / comparative advertising
                "trade mark", "trademark", "trade marks", "tma 1994", "trade marks act 1994",
                "section 10", "s 10", "section 10(2)", "s 10(2)", "section 10(3)", "s 10(3)",
                "likelihood of confusion", "unfair advantage", "detriment", "detriment to distinctive character",
                "detriment to repute", "tarnishment", "dilution",
                "functions of a trade mark", "origin function", "advertising function", "investment function", "communication function",
                "l'oreal", "bellure", "google france", "intel", "specsavers", "o2", "comparative advertising",
                "trade mark", "trademark", "trade marks act 1994", "tma 1994",
                "section 10", "s 10", "section 11", "s 11",
                "likelihood of confusion", "unfair advantage", "detriment", "reputation", "descriptive use",
            ],
            "family": [
                "family law", "divorce", "no-fault", "no fault", "divorce dissolution and separation act",
                "ddsa 2020", "matrimonial causes act", "mca 1973", "section 25", "s 25", "section 25a", "s 25a",
                "financial remedy", "ancillary relief", "clean break",
                "sharing principle", "needs", "compensation",
                "white v white", "miller", "mcfarlane", "sharp v sharp",
                "matrimonial property", "non-matrimonial", "non matrimonial", "inheritance",
                "post-separation", "post separation", "lottery", "rossi", "s v ag",
                # Private children / welfare
                "children act", "children act 1989", "ca 1989", "section 8", "s 8",
                "child arrangements order", "cafcass", "welfare checklist", "paramountcy", "paramount",
                "parental involvement", "contact", "residence",
            ],
            "evidence": [
                "evidence", "criminal evidence", "hearsay", "res gestae",
                "criminal justice act 2003", "cja 2003", "s 114", "section 114", "s 116", "section 116",
                "s 117", "section 117", "s 118", "section 118", "s 125", "section 125",
                "article 6", "horncastle", "al-khawaja", "al khawaja",
                "pace", "section 78", "s 78",
                "criminal justice and public order act 1994", "cjpo 1994", "s 34", "section 34",
                "bad character", "s 101", "section 101", "s 103", "section 103",
                "adverse inference",
            ],
            "data_privacy": [
                "gdpr", "uk gdpr", "data protection", "privacy", "dpia", "controller", "processor",
                "personal data", "special category", "lawful basis", "consent", "legitimate interests",
                "dpa 2018", "data protection act", "ico", "pecr", "cookies",
                "ai", "automated decision", "profiling", "machine learning",
                "cyber", "breach notification", "information security",
            ],
            "commercial": [
                "commercial", "sale of goods", "sga", "agency", "bills of exchange", "carriage",
                "cif", "fob", "incoterms", "letters of credit", "l/c", "guarantee", "indemnity",
                "secured transactions", "charge", "mortgage", "insolvency set-off",
            ],
            "adr": [
                "mediation", "arbitration", "adr", "settlement", "without prejudice",
                "commercial mediation",
            ],
            "medical": [
                "medical", "medicine", "clinical", "nhs", "doctor", "patient", "consent", "informed consent",
                "montgomery", "bolam", "bolitho", "sidaway", "therapeutic privilege",
                "material risk", "materiality", "disclosure", "risk disclosure",
                "bio", "bioethics", "biolaw", "medical law",
            ],
            "employment": [
                "employment", "employer", "employee", "contract of employment",
                "restrictive covenant", "restrictive covenants", "restraint of trade",
                "non-compete", "non compete", "non-solicitation", "non solicitation",
                "garden leave", "confidentiality", "trade secret", "trade secrets",
                "injunction", "severance", "blue pencil", "post-termination", "post termination",
                "dismissal", "wrongful dismissal", "constructive dismissal",
            ],
            "company": [
                "company", "companies", "corporate", "director", "directors", "shareholder", "shareholders",
                "derivative", "fiduciary", "companies act", "ca 2006", "s 172", "section 172", "s172",
            ],
            "trusts": [
                # Core trust concepts
                "trust", "trusts", "trustee", "trustees", "beneficiary", "beneficiaries",
                "settlor", "settler", "trust property", "trust fund",
                "express trust", "implied trust", "resulting trust", "constructive trust",
                "equity", "equitable", "equitable interest", "equitable title",
                "legal title", "bare trust", "fixed trust", "discretionary trust",

                # Three certainties
                "three certainties", "certainty of intention", "certainty of subject matter",
                "certainty of objects", "conceptual certainty", "evidential certainty",
                "knight v knight", "re adams and kensington", "re kayford",
                "paul v constance", "paul", "the money is as much yours as mine",
                "re london wine", "london wine", "tangible ascertainability",
                "re goldcorp", "goldcorp", "unascertained goods",
                "hunter v moss", "hunter", "intangible property",
                "re hay's settlement", "hay", "is or is not test",
                "mcphail v doulton", "mcphail", "given postulant",
                "re baden (no 2)", "baden", "sachs megaw stamp",
                "irc v broadway cottages", "broadway cottages", "fixed trust complete list",

                # Constitution of trusts
                "constitution", "constituted trust", "completely constituted",
                "imperfectly constituted", "equity will not assist a volunteer",
                "milroy v lord", "milroy", "three modes",
                "re rose", "rose", "done everything necessary",
                "mascall v mascall", "mascall",
                "pennington v waine", "pennington", "unconscionable",
                "t choithram v pagarani", "choithram", "one of several trustees",
                "strong v bird", "strong", "donatio mortis causa", "dmc",
                "sen v headley", "sen", "dominion",

                # Secret trusts
                "secret trust", "half-secret trust", "fully secret trust",
                "communication", "acceptance", "reliance",
                "ottaway v norman", "ottaway", "fully secret",
                "re boyes", "boyes", "timing",
                "re keen", "keen", "sealed envelope",
                "re bateman", "bateman", "dehors the will",
                "blackwell v blackwell", "blackwell",

                # Resulting trusts
                "resulting trust", "automatic resulting trust", "presumed resulting trust",
                "quistclose trust", "quistclose", "barclays bank v quistclose",
                "re vandervell (no 2)", "vandervell", "air jamaica",
                "presumption of advancement", "equality act 2010", "section 199",
                "purchase money resulting trust", "gratuitous transfer",

                # Constructive trusts
                "constructive trust", "institutional constructive trust",
                "remedial constructive trust", "common intention constructive trust",
                "citct", "beneficial interest", "family home",
                "lloyds bank v rosset", "rosset", "express discussions", "direct contributions",
                "stack v dowden", "stack", "jones v kernott", "kernott",
                "sole name", "joint names", "quantification",
                "gissing v gissing", "gissing", "common intention",
                "grant v edwards", "grant", "eves v eves", "eves", "excuse",
                "oxley v hiscock", "oxley", "fairness",
                "abbott v abbott", "abbott", "privy council",
                "capehorn v harris", "capehorn",

                # Proprietary estoppel
                "proprietary estoppel", "estoppel", "assurance", "reliance", "detriment",
                "representation", "expectation", "minimum equity",
                "thorner v major", "thorner", "oblique assurance",
                "guest v guest", "guest", "prima facie expectation",
                "jennings v rice", "jennings", "proportionality",
                "gillett v holt", "gillett", "irrevocable assurance",
                "crabb v arun", "crabb", "equity of access",
                "cobbe v yeoman's row", "cobbe", "commercial context",
                "herbert v doyle", "herbert", "constructive trust overlap",

                # TOLATA
                "tolata", "trusts of land and appointment of trustees act 1996",
                "section 14", "s 14", "section 15", "s 15",
                "application to court", "sale", "occupation rights",
                "re citro", "citro", "bankruptcy",
                "mortgage corporation v shaire", "shaire",

                # Fiduciary duties
                "fiduciary", "fiduciary duty", "fiduciary relationship",
                "no conflict rule", "no profit rule", "duty of loyalty",
                "keech v sandford", "keech", "renewal of lease",
                "boardman v phipps", "boardman", "information",
                "regal (hastings) v gulliver", "regal hastings",
                "bray v ford", "bray", "unless authorised",
                "self-dealing", "fair dealing",

                # Breach of trust
                "breach of trust", "personal liability", "proprietary liability",
                "target holdings v redferns", "target holdings", "but for",
                "aib v mark redler", "aib", "equitable compensation",
                "contribution", "indemnity", "limitation act 1980",

                # Tracing and following
                "tracing", "following", "claiming", "proprietary remedy",
                "common law tracing", "equitable tracing",
                "taylor v plumer", "taylor", "unmixed fund",
                "re hallett's estate", "hallett", "mixed fund", "innocent mixing",
                "re oatway", "oatway", "wrongdoer", "lowest intermediate balance",
                "foskett v mckeown", "foskett", "insurance policy",
                "re diplock", "diplock", "backwards tracing",
                "boscawen v bajwa", "boscawen",
                "shalson v russo", "shalson", "backwards tracing",

                # Third party liability
                "knowing receipt", "recipient liability",
                "bank of credit and commerce v akindele", "akindele", "unconscionability",
                "dishonest assistance", "accessory liability",
                "barnes v addy", "barnes", "two limbs",
                "royal brunei airlines v tan", "royal brunei", "dishonesty",
                "twinsectra v yardley", "twinsectra", "combined test",
                "barlow clowes v eurotrust", "barlow clowes", "privy council",
                "ivey v genting", "ivey", "objective dishonesty",
                "group seven v notable", "group seven", "strict liability",
                "manifest shipping v uni-polaris", "manifest", "blind eye knowledge",

                # Wills and probate
                "wills", "will", "testator", "testatrix", "probate", "intestacy",
                "inheritance", "executor", "administrator", "personal representative",
                "wills act 1837", "section 9", "s 9", "formalities",
                "attestation", "witnesses", "testamentary capacity",
                "banks v goodfellow", "banks", "capacity test",
                "knowledge and approval", "suspicious circumstances",
                "undue influence", "fraudulent calumny",
                "mutual wills", "secret trust", "inheritance act 1975",
                "codicil", "revocation", "dependent relative revocation",
            ],
            "tort": [
                # Core tort concepts
                "tort", "torts", "tortious", "tortfeasor", "tortious liability",

                # NEGLIGENCE - Core elements
                "negligence", "negligent", "negligently", "carelessness", "careless",
                "duty of care", "duty of care owed", "owed a duty", "legal duty",
                "breach of duty", "breach", "breached", "standard of care",
                "causation", "cause", "caused", "causal", "chain of causation",
                "remoteness", "remoteness of damage", "too remote", "foreseeable",
                "foreseeability", "reasonably foreseeable", "type of damage",

                # Duty of care - key cases & principles
                "donoghue v stevenson", "donoghue", "stevenson", "neighbour principle",
                "neighbour test", "reasonable foresight", "proximity", "just and reasonable",
                "caparo v dickman", "caparo", "caparo test", "three-stage test", "incremental",
                "robinson v chief constable", "robinson", "established duty", "novel duty",
                "anns v merton", "anns", "two-stage test",
                "murphy v brentwood", "murphy", "pure economic loss", "economic loss",
                "hedley byrne", "hedley byrne v heller", "special relationship", "assumption of responsibility",
                "spring v guardian", "white v jones", "wills solicitor",

                # Breach of duty - standard of care
                "reasonable person", "reasonable man", "objective standard", "ordinarily competent",
                "blyth v birmingham", "blyth", "omission to do something",
                "nettleship v weston", "nettleship", "learner driver",
                "bolam v friern", "bolam", "bolam test", "professional standard", "responsible body",
                "bolitho v hackney", "bolitho", "logical basis",
                "montgomery v lanarkshire", "montgomery", "informed consent", "material risk",

                # Breach - risk calculus (Latimer factors)
                "latimer v aec", "latimer", "risk calculus", "cost-benefit", "cost benefit",
                "magnitude of risk", "likelihood of harm", "severity of harm", "gravity of harm",
                "practicability", "practicability of precautions", "cost of precautions",
                "social utility", "social value", "utility of conduct",
                "bolton v stone", "bolton", "cricket ball", "probability of injury",
                "paris v stepney", "paris", "one-eyed welder", "serious consequences",
                "haley v london electricity", "haley", "blind pedestrian",
                "watt v hertfordshire", "watt", "emergency services", "fire brigade",
                "scout association v barnes", "scout", "social benefit",
                "tomlinson v congleton", "tomlinson", "obvious risk", "free will",
                "compensation act 2006", "section 1 compensation act", "desirable activity",

                # Causation - factual
                "but for test", "but for", "but-for", "factual causation",
                "barnett v chelsea", "barnett", "hospital", "arsenic poisoning",
                "mcghee v national coal board", "mcghee", "material contribution",
                "wilsher v essex", "wilsher", "multiple causes", "possible causes",
                "fairchild v glenhaven", "fairchild", "mesothelioma", "materially increased risk",
                "sienkiewicz v greif", "sienkiewicz", "doubling of risk",
                "bailey v ministry of defence", "bailey", "cumulative cause",

                # Causation - legal (novus actus)
                "novus actus interveniens", "novus actus", "intervening act", "break in chain",
                "chain broken", "independent cause", "superseding cause",
                "knightley v johns", "knightley", "unreasonable act",
                "lamb v camden", "lamb", "squatters", "third party act",
                "mckew v holland", "mckew", "claimant's own act", "unreasonable conduct",
                "corr v ibc", "corr", "suicide", "depression",
                "spencer v wincanton", "spencer", "second accident",

                # Remoteness
                "wagon mound", "wagon mound no 1", "wagon mound (no 1)", "overseas tankship",
                "type of harm", "kind of damage", "manner of occurrence",
                "hughes v lord advocate", "hughes", "paraffin lamp", "explosion",
                "doughty v turner", "doughty", "eruption", "splash",
                "smith v leech brain", "smith", "eggshell skull", "thin skull",
                "egg-shell skull", "take victim as found", "pre-existing condition",
                "robinson v post office", "robinson", "allergic reaction",

                # Defences to negligence
                "contributory negligence", "contributory", "contrib neg",
                "law reform contributory negligence act 1945", "just and equitable",
                "reduction", "apportionment", "share of blame",
                "sayers v harlow", "sayers", "toilet cubicle",
                "froom v butcher", "froom", "seatbelt",
                "volenti non fit injuria", "volenti", "consent", "voluntary assumption",
                "ici v shatwell", "shatwell", "shotfirer",
                "morris v murray", "morris", "drunk pilot",
                "illegality", "ex turpi causa", "illegal activity",
                "gray v thames trains", "gray", "manslaughter",
                "patel v mirza", "patel", "trio of considerations",

                # Psychiatric injury / nervous shock
                "psychiatric injury", "nervous shock", "mental injury", "psychological harm",
                "primary victim", "secondary victim", "bystander",
                "page v smith", "page", "zone of danger", "physical impact",
                "dulieu v white", "dulieu", "fear of immediate injury",
                "alcock v chief constable", "alcock", "hillsborough", "proximity requirements",
                "mcloughlin v o'brian", "mcloughlin", "immediate aftermath",
                "white v chief constable", "white", "rescuer", "professional rescuer",
                "paul v royal wolverhampton", "paul", "secondary victim",
                "ronayne v liverpool", "ronayne", "horrifying event",
                "liverpool women's hospital", "walters v north glamorgan",
                "close tie of love and affection", "temporal proximity", "spatial proximity",
                "own unaided senses", "sudden shock", "gradual realisation",

                # Occupiers' liability
                "occupiers liability", "occupier", "occupiers' liability act 1957",
                "occupiers' liability act 1984", "ola 1957", "ola 1984",
                "visitor", "lawful visitor", "trespasser", "child trespasser",
                "common duty of care", "section 2(2)", "s 2(2)",
                "warning", "adequate warning", "discharge duty",
                "wheat v lacon", "wheat", "multiple occupiers",
                "roles v nathan", "roles", "chimney sweep", "obvious risk",
                "phipps v rochester", "phipps", "child", "parental supervision",
                "british railway board v herrington", "herrington", "humanitarian duty",
                "tomlinson v congleton", "tomlinson", "diving", "obvious danger",

                # Nuisance
                "nuisance", "private nuisance", "public nuisance", "nuisance claim",
                "unreasonable interference", "use and enjoyment", "land use",
                "sturges v bridgman", "sturges", "coming to nuisance",
                "hunter v canary wharf", "hunter", "television reception", "proprietary interest",
                "cambridge water v eastern counties", "cambridge water", "foreseeability",
                "coventry v lawrence", "coventry", "planning permission",
                "miller v jackson", "miller", "cricket ground",
                "kennaway v thompson", "kennaway", "motorboat racing",
                "locality", "character of neighbourhood", "duration", "sensitivity",

                # Rylands v Fletcher / strict liability
                "rylands v fletcher", "rylands", "strict liability", "escape",
                "non-natural use", "accumulation", "dangerous thing",
                "transco v stockport", "transco", "modern application",
                "cambridge water", "foreseeability requirement",
                "read v lyons", "read", "munitions factory", "escape requirement",

                # Vicarious liability
                "vicarious liability", "vicariously liable", "employer liability",
                "course of employment", "scope of employment", "field of activities",
                "close connection", "close connection test", "sufficiently connected",
                "lister v hesley hall", "lister", "sexual abuse", "warden",
                "mohamud v wm morrison", "mohamud", "petrol station", "assault",
                "cox v ministry of justice", "cox", "prisoners", "catering",
                "various claimants v catholic child welfare", "various claimants", "christian brothers",
                "barclays bank v various claimants", "barclays", "akin to employment",
                "wm morrison supermarkets v various claimants", "morrisons", "data breach", "frolic",
                "uber v aslam", "uber", "gig economy", "worker status",
                "independent contractor", "control test", "integration test",
                "ready mixed concrete", "multiple factors",

                # Product liability
                "product liability", "defective product", "consumer protection act 1987",
                "cpa 1987", "strict product liability", "defect", "producer",
                "a v national blood authority", "national blood", "blood products",
                "development risks defence", "state of the art",

                # Professional negligence
                "professional negligence", "professional liability", "solicitor negligence",
                "medical negligence", "clinical negligence",
                "accountant negligence", "surveyor negligence",
                "white v jones", "solicitor will", "disappointed beneficiary",

                # Employers' liability
                "employers' liability", "employer's duty", "safe system of work",
                "competent staff", "safe equipment", "safe premises",
                "wilsons and clyde coal", "personal non-delegable duty",
                "employers' liability act 1969", "employers' liability defective equipment act",

                # Act of God / force majeure in tort
                "act of god", "vis major", "inevitable accident", "natural event",
                "greenock corporation v caledonian railway", "greenock", "extraordinary rainfall",
                "nichols v marsland", "nichols", "ornamental lakes", "unprecedented storm",

                # Damages in tort
                "damages", "compensatory damages", "general damages", "special damages",
                "pecuniary loss", "non-pecuniary loss", "pain and suffering",
                "loss of amenity", "future loss", "multiplier", "multiplicand",
                "mitigation", "duty to mitigate", "failure to mitigate",
                "lump sum", "periodical payments", "provisional damages",
            ],
            "contract": [
                # Core contract formation
                "contract", "contracts", "contractual", "agreement", "legally binding",
                "offer", "offeror", "offeree", "invitation to treat", "advertisement",
                "partridge v crittenden", "carlill v carbolic", "carlill", "smoke ball",
                "pharmaceutical society v boots", "boots", "self-service",
                "acceptance", "acceptance by conduct", "acceptance by post", "postal rule",
                "adams v lindsell", "adams", "instantaneous communication",
                "entores v miles far east", "entores", "telex",
                "brinkibon v stahag stahl", "brinkibon",
                "consideration", "past consideration", "executory consideration", "executed consideration",
                "sufficiency of consideration", "adequacy", "peppercorn",
                "currie v misa", "currie", "valuable consideration",
                "re mcardle", "mcardle", "past consideration invalid",
                "lampleigh v braithwait", "lampleigh", "implied promise",
                "williams v roffey", "roffey", "practical benefit", "factual benefit",
                "stilk v myrick", "stilk", "pre-existing duty",
                "hartley v ponsonby", "hartley", "extra duty",
                "foakes v beer", "foakes", "part payment", "pinnel's case",
                "mwb business exchange v rock advertising", "mwb", "rock advertising", "oral variation",
                "promissory estoppel", "estoppel", "equitable estoppel",
                "central london property v high trees", "high trees", "denning",
                "combe v combe", "combe", "shield not sword",
                "intention to create legal relations", "intention", "domestic agreements",
                "balfour v balfour", "balfour", "husband wife",
                "merritt v merritt", "merritt", "separated spouses",
                "jones v padavatton", "jones", "mother daughter",
                "simpkins v pays", "simpkins", "competition",
                "commercial agreements", "presumed intention",
                "rose and frank v crompton", "rose and frank", "honour clause",
                "esso petroleum v commissioners", "esso", "world cup coins",

                # Privity of contract
                "privity", "privity of contract", "third party rights",
                "contracts (rights of third parties) act 1999", "1999 act",
                "tweddle v atkinson", "tweddle",
                "dunlop v selfridge", "dunlop",
                "jackson v horizon holidays", "jackson", "family holiday",
                "collateral contract", "shanklin pier v detel", "shanklin",

                # Terms of contract
                "terms", "express terms", "implied terms", "conditions", "warranties",
                "innominate terms", "intermediate terms",
                "hong kong fir v kawasaki", "hong kong fir", "seaworthiness",
                "poussard v spiers", "poussard", "condition",
                "bettini v gye", "bettini", "warranty",
                "parol evidence rule", "entire agreement clause",
                "incorporation", "reasonable notice", "parker v south eastern railway", "parker",
                "thornton v shoe lane parking", "thornton", "red hand",
                "interfoto v stiletto", "interfoto", "onerous clause",
                "implied by statute", "sga 1979", "sale of goods act",
                "satisfactory quality", "fitness for purpose", "description",
                "consumer rights act", "cra 2015", "consumer rights act 2015",
                "consumer", "digital content", "goods", "services",

                # Misrepresentation
                "misrepresentation", "misrep", "misrepresent", "representation", "representations",
                "misrepresentation act", "misrepresentation act 1967", "ma 1967",
                "section 2", "s 2", "section 2(1)", "s 2(1)", "section 2(2)", "s 2(2)",
                "section 3", "s 3", "reasonable grounds",
                "fraud", "fraudulent", "fraudulent misrepresentation",
                "innocent misrepresentation", "negligent misrepresentation",
                "fiction of fraud", "statutory sledgehammer",
                "derry v peek", "derry", "royscot", "royscot trust", "royscot v rogerson", "rogerson",
                "howard marine", "hedley byrne", "esso petroleum v mardon", "mardon",
                "heilbut", "heilbut symons", "dick bentley", "oscar chess", "term or representation",
                "bisset v wilkinson", "bisset", "statement of opinion",
                "smith v land and house property", "smith", "implied fact",
                "edgington v fitzmaurice", "edgington", "statement of intention",
                "with v o'flanagan", "with", "change of circumstances",
                "spice girls v aprilia", "spice girls", "conduct",
                "attwood v small", "attwood", "no inducement",
                "redgrave v hurd", "redgrave", "opportunity to verify",
                "collateral warranty", "inducement", "rescission", "bars to rescission",
                "leaf v international galleries", "leaf", "lapse of time",
                "long v lloyd", "long", "affirmation",

                # Remedies / remoteness
                "hadley v baxendale", "hadley", "remoteness in contract", "contemplation of parties",
                "victoria laundry v newman", "victoria laundry", "special circumstances",
                "the heron ii", "heron", "koufos", "serious possibility",
                "transfield shipping v mercator", "transfield", "achilleas", "assumption of responsibility",
                "doyle v olby", "doyle", "smith new court", "measure of damages",
                "expectation interest", "reliance interest", "restitution interest",

                # Exclusion clauses
                "exclusion clause", "exemption clause", "limitation clause",
                "ucta", "unfair contract terms act 1977", "unfair contract terms",
                "section 2", "s 2", "negligence liability",
                "section 3", "s 3", "written standard terms",
                "section 11", "schedule 2", "reasonableness test",
                "cra 2015", "unfair terms", "consumer protection",
                "photo production v securicor", "photo production", "fundamental breach",
                "entire agreement", "non-reliance", "no reliance",
                "contra proferentem", "ambiguity",
                "canada steamship", "clear words", "negligence exclusion",

                # Frustration
                "frustration", "frustrated", "frustrating event",
                "taylor v caldwell", "taylor", "music hall", "destruction",
                "krell v henry", "krell", "coronation", "foundation of contract",
                "herne bay steamboat v hutton", "herne bay", "naval review",
                "davis contractors v fareham", "davis contractors", "delay",
                "national carriers v panalpina", "panalpina", "lease",
                "super servant two", "super servant", "self-induced",
                "maritime national fish v ocean trawlers", "ocean trawlers", "election",
                "force majeure", "force majeure clause",
                "law reform (frustrated contracts) act 1943", "lr(fca) 1943", "1943 act",
                "fibrosa", "fibrosa v fairbairn", "total failure of consideration",
                "chandler v webster", "chandler",
                "gamerco v icm", "gamerco", "bp exploration v hunt", "bp exploration",
                "canary wharf v european medicines agency", "canary wharf", "brexit",

                # Breach and termination
                "breach", "breach of contract", "repudiatory breach", "anticipatory breach",
                "termination", "terminate", "affirmation", "election",
                "hochster v de la tour", "hochster", "anticipatory",
                "white and carter v mcgregor", "white and carter", "legitimate interest",
                "specific performance", "injunction", "equitable remedies",

                # Duress and undue influence
                "duress", "economic duress", "illegitimate pressure",
                "dsnd subsea v petroleum geo-services", "dsnd",
                "universe tankships v itf", "universe tankships",
                "pao on v lau yiu long", "pao on", "commercial pressure",
                "times travel v pakistan airlines", "times travel", "lawful act duress",
                "undue influence", "presumed undue influence", "actual undue influence",
                "class 1", "class 2a", "class 2b",
                "royal bank of scotland v etridge", "etridge", "independent advice",
                "barclays bank v o'brien", "o'brien", "notice", "constructive notice",
                "cibc mortgages v pitt", "pitt",
            ],
            "criminal": [
                # Core criminal law concepts
                "criminal", "criminal law", "crime", "offence", "offense", "criminal liability",
                "mens rea", "guilty mind", "mental element", "fault element",
                "actus reus", "guilty act", "conduct element", "physical element",
                "coincidence", "contemporaneity", "transaction principle",
                "thabo meli", "fagan v mpc", "fagan", "continuing act",
                "r v church", "church", "series of acts",

                # Intention
                "intention", "intent", "direct intention", "oblique intention",
                "virtual certainty", "foresight", "woollin", "r v woollin",
                "nedrick", "r v nedrick", "moloney", "r v moloney",
                "hancock and shankland", "hancock",

                # Recklessness
                "recklessness", "reckless", "subjective recklessness", "cunningham recklessness",
                "r v cunningham", "cunningham", "caldwell", "objective recklessness",
                "r v g", "r v g and another", "overruled caldwell",

                # Murder and manslaughter
                "murder", "unlawful killing", "malice aforethought", "year and a day",
                "manslaughter", "voluntary manslaughter", "involuntary manslaughter",
                "constructive manslaughter", "unlawful act manslaughter",
                "r v church", "church", "dangerous act", "sober and reasonable",
                "r v newbury and jones", "newbury", "objective dangerousness",
                "gross negligence manslaughter", "gross negligence",
                "r v adomako", "adomako", "obvious risk of death", "grossly negligent",
                "r v bateman", "bateman", "r v misra", "misra",
                "r v sellu", "sellu", "r v rose", "rose",
                "reckless manslaughter", "subjective recklessness death",

                # Partial defences to murder
                "diminished responsibility", "abnormality of mental functioning",
                "coroners and justice act 2009", "section 52", "s 52",
                "r v byrne", "byrne", "irresistible impulse",
                "r v dietschmann", "dietschmann", "intoxication",
                "loss of control", "loss of self-control", "qualifying trigger",
                "fear trigger", "anger trigger", "section 54", "s 54", "section 55", "s 55",
                "r v clinton", "clinton", "sexual infidelity",
                "r v dawes", "dawes", "self-induced",
                "provocation", "r v duffy", "duffy", "sudden and temporary",
                "suicide pact", "section 4", "homicide act 1957",

                # Non-fatal offences
                "assault", "battery", "common assault", "assault by beating",
                "abh", "actual bodily harm", "section 47", "s 47", "oapa",
                "offences against the person act 1861", "oapa 1861",
                "r v miller", "miller", "bodily harm",
                "gbh", "grievous bodily harm", "section 18", "s 18", "section 20", "s 20",
                "maliciously", "wounding", "wound",
                "r v bollom", "bollom", "characteristics of victim",
                "r v burstow", "burstow", "r v ireland", "ireland", "psychiatric harm",
                "r v savage", "savage", "r v parmenter", "parmenter",
                "r v mowatt", "mowatt", "foresight of some harm",
                "consent", "r v brown", "brown", "sado-masochism",
                "r v wilson", "wilson", "branding",
                "r v barnes", "barnes", "sport",

                # Theft and property offences
                "theft", "theft act 1968", "ta 1968", "appropriation", "property",
                "belonging to another", "dishonesty", "intention to permanently deprive",
                "s 1", "section 1", "s 2", "section 2", "s 3", "section 3",
                "s 4", "section 4", "s 5", "section 5", "s 6", "section 6",
                "r v gomez", "gomez", "consent ineffective",
                "r v hinks", "hinks", "valid gift",
                "r v morris", "morris", "label switching",
                "ivey v genting", "ivey", "ghosh test overruled",
                "r v ghosh", "ghosh", "dishonesty test",
                "r v lloyd", "lloyd", "borrowing", "all the virtue",
                "robbery", "section 8", "s 8", "force", "immediately before or at time",
                "r v hale", "hale", "continuing appropriation",
                "r v clouden", "clouden", "force on property",
                "burglary", "section 9", "s 9", "s 9(1)(a)", "s 9(1)(b)",
                "building", "part of building", "entry", "trespass",
                "r v collins", "collins", "effective and substantial entry",
                "r v ryan", "ryan", "partial entry",
                "fraud", "fraud act 2006", "false representation", "failing to disclose",
                "abuse of position", "section 2", "s 2", "section 3", "s 3", "section 4", "s 4",
                "r v ivey", "barton and booth", "barton", "booth",

                # Inchoate offences
                "inchoate", "attempt", "criminal attempts act 1981",
                "more than merely preparatory", "section 1", "r v gullefer", "gullefer",
                "r v geddes", "geddes", "r v jones", "jones",
                "conspiracy", "statutory conspiracy", "common law conspiracy",
                "criminal law act 1977", "section 1(1)",
                "r v anderson", "anderson", "play some part",
                "r v saik", "saik", "knowledge of facts",
                "encouraging or assisting", "serious crime act 2007",
                "sections 44-46", "s 44", "s 45", "s 46",

                # Participation in crime
                "accessory", "secondary party", "joint enterprise", "common purpose",
                "principal", "aiding", "abetting", "counselling", "procuring",
                "accessories and abettors act 1861",
                "r v jogee", "jogee", "parasitic accessory liability overruled",
                "r v powell", "powell", "r v english", "english",
                "r v gnango", "gnango", "transferred malice",
                "withdrawal", "r v becerra", "becerra", "timely communication",

                # Defences
                "defence", "defences", "justification", "excuse",
                "self-defence", "self defence", "reasonable force", "section 76",
                "criminal justice and immigration act 2008",
                "r v martin", "martin", "householder", "grossly disproportionate",
                "r v clegg", "clegg", "excessive force",
                "duress", "duress by threats", "duress of circumstances",
                "r v hasan", "hasan", "graham test", "reasonable belief",
                "r v howe", "howe", "no defence to murder",
                "r v gotts", "gotts", "no defence to attempted murder",
                "r v conway", "conway", "r v martin", "duress of circumstances",
                "necessity", "r v dudley and stephens", "dudley", "cannibalism",
                "r v re a", "re a", "conjoined twins",
                "intoxication", "voluntary intoxication", "involuntary intoxication",
                "specific intent", "basic intent", "r v majewski", "majewski",
                "r v heard", "heard", "r v lipman", "lipman",
                "insanity", "m'naghten rules", "m'naghten", "mcnaughten",
                "disease of the mind", "defect of reason", "nature and quality",
                "automatism", "insane automatism", "non-insane automatism",
                "r v quick", "quick", "r v hennessy", "hennessy",
                "r v burgess", "burgess", "sleepwalking",
                "mistake", "mistaken belief", "r v williams", "gladstone williams",
            ],
            "public_law": [
                # Core constitutional principles
                "public law", "constitutional", "constitution", "constitutional law",
                "parliamentary sovereignty", "parliament", "sovereignty", "supremacy",
                "dicey", "a v dicey", "law of the constitution",
                "rule of law", "legality", "legal certainty", "equality before law",
                "separation of powers", "executive", "legislature", "judiciary",
                "constitutional conventions", "convention", "unwritten constitution",

                # Parliamentary sovereignty key cases
                "factortame", "r v secretary of state ex p factortame",
                "jackson v attorney general", "jackson", "hunting act",
                "thoburn v sunderland", "thoburn", "metric martyrs", "constitutional statutes",
                "r (miller) v secretary of state", "miller", "miller 1", "miller 2",
                "prorogation", "article 50", "brexit",
                "r (buckinghamshire) v secretary of state", "hs2",

                # Ouster clauses
                "ouster clause", "privative clause", "exclusion of review",
                "anisminic v foreign compensation commission", "anisminic",
                "r (privacy international) v investigatory powers tribunal", "privacy international",
                "r (cart) v upper tribunal", "cart",

                # Judicial review - standing
                "judicial review", "jr", "administrative law",
                "standing", "locus standi", "sufficient interest", "victim",
                "r v inland revenue ex p national federation", "national federation", "floodgates",
                "r v secretary of state ex p world development movement", "world development movement",
                "r v somerset ex p dixon", "dixon",

                # Judicial review - grounds
                "illegality", "irrationality", "procedural impropriety",
                "gchq", "council of civil service unions", "ccsu", "lord diplock",
                "wednesbury unreasonableness", "wednesbury", "unreasonable",
                "associated provincial picture houses", "so unreasonable",
                "proportionality", "proportionate", "pressing social need",
                "de smith", "wade", "administrative law",

                # Illegality
                "ultra vires", "acting outside powers", "excess of jurisdiction",
                "r v secretary of state ex p fire brigades union", "fire brigades union",
                "padfield v minister of agriculture", "padfield", "improper purpose",
                "r v secretary of state ex p venables", "venables", "fettering discretion",
                "british oxygen v board of trade", "british oxygen", "rigid policy",
                "r v port of london ex p kynoch", "kynoch",

                # Irrationality / unreasonableness
                "manifestly unreasonable", "perverse", "absurd",
                "r v ministry of defence ex p smith", "ex p smith", "anxious scrutiny",
                "r (daly) v secretary of state", "daly", "heightened scrutiny",
                "kennedy v charity commission", "kennedy", "common law rights",

                # Procedural impropriety
                "natural justice", "procedural fairness", "fair hearing",
                "audi alteram partem", "hear both sides", "right to be heard",
                "ridge v baldwin", "ridge", "dismissal", "office holder",
                "r v secretary of state ex p doody", "doody", "reasons",
                "r v army board ex p anderson", "anderson", "disclosure",
                "nemo iudex in causa sua", "rule against bias", "bias",
                "porter v magill", "porter", "apparent bias",
                "r v bow street ex p pinochet", "pinochet", "automatic disqualification",
                "legitimate expectation", "substantive legitimate expectation",
                "r v north and east devon ex p coughlan", "coughlan",
                "r (nadarajah) v secretary of state", "nadarajah", "abuse of power",

                # Human Rights Act
                "human rights act 1998", "hra 1998", "hra", "human rights",
                "convention rights", "echr", "european convention",
                "section 2", "s 2", "take into account", "strasbourg",
                "section 3", "s 3", "interpretive obligation", "so far as possible",
                "section 4", "s 4", "declaration of incompatibility",
                "section 6", "s 6", "public authority", "unlawful",
                "section 7", "s 7", "victim", "standing",
                "horizontal effect", "vertical effect",
                "r v a", "r v a (no 2)", "rape shield", "ghaidan",
                "ghaidan v godin-mendoza", "ghaidan", "reading in",
                "bellinger v bellinger", "bellinger", "declaration",

                # Key Convention rights
                "article 2", "right to life", "positive obligation",
                "osman v uk", "osman", "operational duty",
                "article 3", "torture", "inhuman treatment", "degrading treatment",
                "absolute right", "non-derogable",
                "article 5", "liberty", "security", "detention", "habeas corpus",
                "article 6", "fair trial", "fair hearing", "independent tribunal",
                "article 8", "private life", "family life", "home", "correspondence",
                "qualified right", "necessary in democratic society",
                "article 10", "freedom of expression", "press freedom",
                "article 11", "freedom of assembly", "association",
                "article 14", "discrimination", "prohibited grounds",

                # Remedies in judicial review
                "quashing order", "certiorari", "mandatory order", "mandamus",
                "prohibiting order", "prohibition", "declaration", "injunction",
                "damages", "just satisfaction",
            ],
            "public_law_uk": [
                # Combine with above but UK-specific focus
                "public law", "constitutional", "constitution", "judicial review",
                "parliamentary sovereignty", "dicey", "rule of law", "ouster clause",
                "anisminic", "privacy international", "factortame", "jackson",
                "ultra vires", "legality", "ex p simms",
                "human rights act", "hra", "article 6", "article 8", "article 10",
                "uk constitution", "uncodified", "unwritten",
                "devolution", "scotland act", "wales act", "northern ireland act",
                "sewel convention", "legislative consent",
                "prerogative powers", "royal prerogative", "casu", "gchq",
                "r (miller) v prime minister", "prorogation", "justiciability",
            ],
            "public_law_us": [
                "judicial review", "administrative law", "administrative procedure act", "apa",
                "arbitrary and capricious", "reasoned decisionmaking", "reasoned decision-making",
                "chevron", "auer", "skidmore",
                "standing", "ripeness", "mootness",
                "first amendment", "fourth amendment", "due process",
                "supreme court", "u.s.", "united states",
            ],
            "land": [
                # Core land law concepts
                "land law", "land", "property", "real property", "estate", "interest in land",
                "freehold", "leasehold", "fee simple", "fee simple absolute",
                "legal estate", "equitable interest", "overreaching", "overriding interest",
                "lpa 1925", "law of property act 1925",
                "lra 2002", "land registration act 2002",
                "registered land", "unregistered land", "title register",
                "proprietorship register", "charges register", "property register",

                # Registration and priority
                "registration", "registrable disposition", "priority",
                "first registration", "compulsory registration",
                "schedule 1", "sch 1", "schedule 3", "sch 3",
                "overriding interest", "actual occupation",
                "williams and glyn's bank v boland", "boland",
                "link lending v bustard", "bustard", "fleeting presence",
                "thompson v foy", "thompson", "enquiry",
                "notice", "actual notice", "constructive notice", "imputed notice",
                "hunt v luck", "hunt", "inspection",

                # Co-ownership
                "co-ownership", "joint tenancy", "tenancy in common",
                "four unities", "unity of possession", "unity of interest",
                "unity of title", "unity of time",
                "right of survivorship", "jus accrescendi", "severance",
                "williams v hensman", "williams", "three modes of severance",
                "acting on own share", "mutual agreement", "course of dealing",
                "goodman v gallant", "goodman", "express declaration",
                "stack v dowden", "stack", "jones v kernott", "kernott",
                "section 36(2)", "s 36(2)", "written notice",
                "re draper's conveyance", "draper", "unilateral severance",
                "harris v goddard", "harris", "course of dealing",
                "burgess v rawnsley", "burgess",

                # Trusts of land (TOLATA)
                "tolata", "trusts of land", "trusts of land and appointment of trustees act 1996",
                "section 6", "s 6", "powers of trustees",
                "section 11", "s 11", "consultation",
                "section 12", "s 12", "occupation rights",
                "section 13", "s 13", "exclusion and restriction",
                "section 14", "s 14", "application to court",
                "section 15", "s 15", "matters relevant",
                "re citro", "citro", "bankruptcy trustee",
                "bank of ireland v bell", "bell",

                # Leases
                "lease", "leasehold", "tenancy", "landlord", "tenant",
                "term of years absolute", "exclusive possession",
                "street v mountford", "street", "three hallmarks",
                "rent", "certainty of term", "periodic tenancy",
                "tenancy at will", "licence", "lodger",
                "ag securities v vaughan", "ag securities",
                "bruton v london and quadrant", "bruton", "non-estate lease",
                "lease formalities", "section 52", "s 52", "deed",
                "section 54(2)", "s 54(2)", "short lease exception",
                "walsh v lonsdale", "walsh", "equity looks on as done",
                "covenants in leases", "privity of contract", "privity of estate",
                "landlord and tenant (covenants) act 1995", "ltca 1995",
                "authorised guarantee agreement", "aga",
                "assignment", "subletting", "forfeiture", "relief from forfeiture",
                "section 146", "s 146", "notice", "remedy breach",

                # Easements
                "easement", "easements", "dominant tenement", "servient tenement",
                "right of way", "right to light", "parking",
                "re ellenborough park", "ellenborough", "four characteristics",
                "accommodate", "capable of grant", "ouster principle",
                "copeland v greenhalf", "copeland", "joint user",
                "moncrieff v jamieson", "moncrieff", "ancillary right",
                "regency villas v diamond resorts", "regency villas", "recreational",
                "express grant", "express reservation",
                "implied grant", "implied reservation",
                "necessity", "common intention", "wheeldon v burrows", "wheeldon",
                "section 62", "s 62", "lpa 1925", "conveyance",
                "wood v waddington", "wood", "diversity of occupation",
                "prescription", "lost modern grant", "prescription act 1832",
                "20 years", "40 years", "continuous user", "as of right",
                "nec vi nec clam nec precario",

                # Covenants
                "covenant", "covenants", "restrictive covenant", "positive covenant",
                "freehold covenant", "burden", "benefit",
                "tulk v moxhay", "tulk", "equity running burden",
                "austerberry v oldham", "austerberry", "burden at law",
                "rhone v stephens", "rhone", "positive burden not run",
                "halsall v brizell", "halsall", "benefit and burden",
                "thamesmead v allotey", "thamesmead", "mutual benefit and burden",
                "annexation", "express annexation", "statutory annexation",
                "section 78", "s 78", "federated homes",
                "federated homes v mill lodge", "federated homes",
                "assignment", "building scheme", "scheme of development",
                "elliston v reacher", "elliston", "four requirements",
                "section 84", "s 84", "modification", "discharge",
                "upper tribunal", "lands chamber",

                # Mortgages
                "mortgage", "mortgagor", "mortgagee", "charge", "legal charge",
                "equity of redemption", "right to redeem",
                "clogs on equity", "oppressive terms", "collateral advantage",
                "multiservice bookbinding v marden", "multiservice",
                "cityland v dibble", "cityland",
                "repossession", "possession", "power of sale",
                "section 101", "s 101", "lpa 1925",
                "cuckmere brick v mutual finance", "cuckmere", "duty of care",
                "silven properties v royal bank", "silven",
                "palk v mortgage services", "palk",
                "quennell v maltby", "quennell",

                # Adverse possession
                "adverse possession", "squatter", "squatters", "possession",
                "animus possidendi", "factual possession",
                "powell v mcfarlane", "powell", "intention to possess",
                "ja pye v graham", "pye", "echr",
                "limitation act 1980", "la 1980", "section 15", "s 15",
                "twelve years", "section 17", "s 17", "extinguishment",
                "land registration act 2002", "lra 2002",
                "schedule 6", "sch 6", "ten years", "two-year window",
                "paragraph 1", "para 1", "paragraph 5", "para 5",
                "counter-notice", "objection",
                "paragraph 6", "para 6", "adverse possession in registered land",
                "best v chief land registrar", "best",
                "zarb v parry", "zarb", "boundary",
                "jourdan v scott", "jourdan",
            ],
            "pensions": ["pension", "pensions", "scheme", "trust deed", "ppf", "section 75"],
            "eu": ["eu", "tfeu", "directive", "regulation", "retained eu law"],
            "insolvency": ["insolvency", "administration", "liquidation", "wrongful trading"],
            "insurance": [
                "insurance", "insurer", "insured", "underwriter", "underwriting",
                "subrogation", "utmost good faith", "uberrima fides", "indemnity",
                "lloyd's", "lloyds", "reinsurance", "marine insurance",
                "marine insurance act 1906", "mia 1906",
                "non-disclosure", "material fact", "duty of disclosure",
                "insurance act 2015", "ia 2015",
            ],
            "banking_finance": [
                "banking", "bank", "fca", "pra", "financial conduct authority",
                "prudential regulation authority", "basel", "capital requirements",
                "ring-fencing", "ring fencing", "financial services", "fsma", "fsma 2000",
                "money laundering", "aml", "kyc", "know your customer",
                "payment services", "psd2", "fintech", "cryptocurrency",
                "consumer credit", "consumer credit act 1974", "cca 1974",
            ],
            "construction": [
                "construction", "adjudication", "hgcra",
                "housing grants construction and regeneration act",
                "payment notice", "pay less notice",
                "smash and grab", "true value adjudication",
                "tcc", "technology and construction court",
                "defects", "practical completion", "retention",
                "jct", "nec", "fidic", "design and build",
            ],
            "planning": [
                "planning", "planning permission", "permitted development",
                "tcpa 1990", "town and country planning act",
                "section 106", "s 106", "s106", "planning obligation",
                "community infrastructure levy", "cil",
                "local plan", "national planning policy framework", "nppf",
                "development plan", "material consideration",
                "listed building", "conservation area", "green belt",
                "enforcement notice", "breach of condition",
            ],
            "shipping_admiralty": [
                "shipping", "admiralty", "maritime", "marine",
                "carriage of goods by sea", "cogsa",
                "hague rules", "hague-visby", "hague visby", "hamburg rules", "rotterdam rules",
                "charterparty", "charter party", "time charter", "voyage charter",
                "demurrage", "laytime", "bill of lading",
                "general average", "york-antwerp", "york antwerp",
                "salvage", "collision", "limitation of liability",
            ],
            "tax": [
                "tax", "taxation", "hmrc", "revenue", "inland revenue",
                "gaar", "general anti-abuse rule",
                "sdlt", "stamp duty land tax", "stamp duty",
                "capital gains", "cgt", "capital gains tax",
                "inheritance tax", "iht", "potentially exempt transfer",
                "taar", "targeted anti-avoidance rule",
                "income tax", "corporation tax", "vat", "value added tax",
                "tax avoidance", "tax evasion", "ramsay", "furniss v dawson",
            ],
            "immigration": [
                "immigration", "asylum", "refugee", "deportation",
                "human rights claim", "fresh claim", "further submissions",
                "siac", "special immigration appeals commission",
                "immigration act", "nationality", "british citizenship",
                "right to remain", "leave to remain", "indefinite leave",
                "article 3", "article 8", "removal", "return",
                "hostile environment", "windrush",
                "upper tribunal", "first-tier tribunal",
            ],
            "environmental": [
                "environment", "environmental", "pollution", "contamination",
                "epa 1990", "environmental protection act",
                "environmental permit", "waste", "hazardous waste",
                "contaminated land", "part 2a",
                "aarhus convention", "environmental impact assessment", "eia",
                "nuisance", "statutory nuisance", "clean air",
                "water resources", "climate change act 2008", "net zero",
            ],
            "wills_probate": [
                "will", "wills", "probate", "intestacy", "intestate",
                "inheritance act 1975",
                "testamentary capacity", "banks v goodfellow",
                "testamentary freedom", "freedom of testation",
                "undue influence", "knowledge and approval",
                "grant of probate", "letters of administration",
                "executor", "administrator", "personal representative",
                "codicil", "attestation", "wills act 1837",
                "mutual wills", "secret trust", "half-secret trust",
            ],
            "sports": [
                "sport", "sports", "sports law",
                "cas", "court of arbitration for sport",
                "anti-doping", "wada", "world anti-doping",
                "player transfer", "transfer fee", "transfer window",
                "ffp", "financial fair play", "uefa",
                "bosman", "bosman ruling",
                "match fixing", "corruption in sport",
                "broadcasting rights", "image rights",
            ],

            # ================================================================================
            # INTERNATIONAL INVESTMENT LAW - Comprehensive coverage
            # ================================================================================
            "international_investment": [
                # Core concepts
                "international investment law", "investment law", "foreign investment",
                "investment arbitration", "investor-state", "investor state",
                "isds", "investor-state dispute settlement",
                "bilateral investment treaty", "bit", "bits",
                "investment treaty", "investment protection",
                "foreign direct investment", "fdi",

                # ICSID and jurisdiction
                "icsid", "icsid convention", "washington convention",
                "article 25", "icsid jurisdiction", "investment dispute",
                "icsid arbitration", "icsid tribunal", "icsid award",
                "salini", "salini v morocco", "salini test", "salini criteria",
                "contribution", "duration", "risk", "host state development",
                "nationality", "corporate nationality", "nationality planning",
                "consent", "consent to arbitration", "arbitration clause",
                "uncitral", "uncitral rules", "ad hoc arbitration",
                "pca", "permanent court of arbitration",
                "scc", "stockholm chamber of commerce",

                # Expropriation
                "expropriation", "expropriate", "indirect expropriation",
                "direct expropriation", "creeping expropriation", "regulatory expropriation",
                "nationalization", "nationalisation", "taking", "takings",
                "sole effects", "sole effects doctrine", "effects doctrine",
                "police powers", "police powers doctrine", "regulatory powers",
                "deprivation", "substantial deprivation", "total deprivation",
                "economic impact", "destruction of value",
                "metalclad", "metalclad v mexico",
                "tecmed", "tecmed v mexico", "tecnicas medioambientales",
                "santa elena", "compania del desarrollo",
                "philip morris", "philip morris v australia", "philip morris v uruguay",
                "methanex", "methanex v usa",
                "ethyl corporation", "ethyl v canada",
                "vattenfall", "vattenfall v germany",
                "lone pine", "eli lilly",

                # Fair and Equitable Treatment (FET)
                "fair and equitable treatment", "fet", "fet standard",
                "minimum standard of treatment", "mst", "customary international law minimum",
                "legitimate expectations", "legitimate expectation",
                "regulatory stability", "stable legal framework",
                "predictability", "consistency", "transparency",
                "due process", "denial of justice", "procedural fairness",
                "arbitrary", "arbitrariness", "arbitrary conduct",
                "discriminatory", "discrimination", "non-discrimination",
                "bad faith", "good faith",
                "waste management", "waste management ii", "waste management v mexico",
                "thunderbird", "thunderbird v mexico",
                "glamis gold", "glamis v usa",
                "bilcon", "bilcon v canada",
                "neer", "neer claim", "neer standard",

                # Full Protection and Security
                "full protection and security", "fps", "physical security",
                "legal security", "constant protection",
                "amco", "amco v indonesia",
                "aapl", "aapl v sri lanka",
                "wena hotels", "wena v egypt",

                # National Treatment / MFN
                "national treatment", "most favoured nation", "mfn",
                "like circumstances", "comparable circumstances",
                "maffezini", "maffezini v spain", "mfn clause",

                # Regulatory Chill and Right to Regulate
                "regulatory chill", "chilling effect",
                "right to regulate", "regulatory autonomy", "regulatory space",
                "police powers exception", "public welfare",
                "public health", "environmental protection", "public interest",
                "proportionality", "proportionality test", "balancing",
                "necessity", "necessary measures",
                "margin of appreciation",

                # Compensation and Damages
                "compensation", "prompt adequate effective", "hull formula",
                "fair market value", "fmv", "going concern value",
                "dcf", "discounted cash flow", "valuation",
                "lost profits", "future profits", "lucrum cessans",
                "damnum emergens", "actual loss",
                "interest", "compound interest", "pre-award interest", "post-award interest",
                "moral damages",

                # Treaty Interpretation
                "vclt", "vienna convention on the law of treaties",
                "article 31", "article 32", "treaty interpretation",
                "object and purpose", "preamble",
                "travaux preparatoires", "supplementary means",

                # Defences and Exceptions
                "necessity defence", "article 25 ilc", "essential security",
                "force majeure", "state of necessity",
                "cms", "cms v argentina", "lgee", "lgee v argentina",
                "enron", "enron v argentina", "sempra", "sempra v argentina",
                "argentina cases", "argentine crisis",

                # Annulment and Enforcement
                "annulment", "icsid annulment", "ad hoc committee",
                "manifest excess of powers", "serious departure",
                "failure to state reasons", "corruption",
                "enforcement", "new york convention", "recognition",

                # Modern Developments
                "new generation", "new generation bit", "new generation fta",
                "ceta", "comprehensive economic trade agreement",
                "usmca", "nafta", "chapter 11",
                "tpp", "cptpp", "rcep",
                "investment court system", "ics", "mic",
                "multilateral investment court",
                "achmea", "achmea v slovakia", "intra-eu bit",
                "sustainable development", "csr", "corporate social responsibility",
                "human rights", "environmental standards", "labour standards",

                # Key scholars and sources
                "sornarajah", "dolzer", "schreuer", "newcombe", "paradell",
                "vandevelde", "muchlinski", "douglas",
            ],

            # ================================================================================
            # INTERNATIONAL TRADE LAW (WTO)
            # ================================================================================
            "international_trade": [
                # Core WTO
                "wto", "world trade organization", "world trade organisation",
                "gatt", "general agreement on tariffs and trade",
                "gats", "general agreement on trade in services",
                "trips", "trade related intellectual property",
                "dispute settlement", "dsb", "dispute settlement body",
                "panel", "appellate body", "ab report",

                # Key principles
                "most favoured nation", "mfn", "national treatment",
                "like products", "likeness",
                "tariff", "tariffs", "customs duty", "bound rate",
                "quantitative restriction", "quota",
                "subsidies", "scm agreement", "countervailing",
                "anti-dumping", "dumping", "dumping margin",
                "safeguards", "safeguard measures",

                # Exceptions
                "article xx", "gatt article xx", "general exceptions",
                "public morals", "human health", "exhaustible natural resources",
                "chapeau", "arbitrary discrimination", "disguised restriction",
                "article xxi", "security exceptions", "essential security",

                # Key cases
                "us shrimp", "shrimp turtle",
                "ec hormones", "hormones",
                "ec asbestos", "asbestos",
                "us gambling", "gambling",
                "china rare earths", "rare earths",
                "ec seal products", "seal products",
                "australia tobacco plain packaging",

                # SPS and TBT
                "sps agreement", "sanitary and phytosanitary",
                "tbt agreement", "technical barriers to trade",
                "precautionary principle", "risk assessment",
                "international standards", "codex", "oie", "ippc",
            ],

            # ================================================================================
            # HUMAN RIGHTS LAW (International)
            # ================================================================================
            "international_human_rights": [
                # Core instruments
                "human rights", "international human rights", "ihrl",
                "udhr", "universal declaration",
                "iccpr", "international covenant civil political",
                "icescr", "economic social cultural rights",
                "echr", "european convention human rights", "european convention on human rights",
                "iachr", "inter-american", "american convention",
                "achpr", "african charter", "banjul charter",
                "cat", "convention against torture", "uncat",

                # Bodies and courts
                "ecthr", "european court of human rights", "strasbourg",
                "un human rights committee", "hrc",
                "un human rights council",
                "special rapporteur", "treaty body",
                "inter-american court",

                # ECHR Article 1 - Jurisdiction (CRITICAL for extraterritorial questions)
                "article 1 echr", "article 1", "jurisdiction echr",
                "extraterritorial jurisdiction", "extraterritoriality", "extraterritorial",
                "jurisdictional link", "jurisdictional nexus",
                "territorial jurisdiction", "primarily territorial",
                "exceptional circumstances", "state agent authority", "state agent control",
                "physical power and control", "effective control", "effective overall control",
                "spatial model", "personal model", "control model",

                # Key extraterritorial jurisdiction cases
                "al-skeini", "al skeini", "al-skeini v uk", "al-skeini v united kingdom",
                "bankovic", "banković", "bankovic v belgium",
                "loizidou", "loizidou v turkey",
                "cyprus v turkey", "northern cyprus",
                "georgia v russia", "georgia v russia ii", "georgia v russia (ii)",
                "hanan v germany", "hanan", "carter v russia", "carter",
                "al-jedda", "al jedda", "hassan v uk", "hassan v united kingdom",
                "jaloud v netherlands", "jaloud",
                "issa v turkey", "issa", "ocalan", "öcalan", "ocalan v turkey",
                "medvedyev", "medvedyev v france",
                "hirsi jamaa", "hirsi jamaa v italy",
                "m.n. and others v belgium", "mn v belgium",

                # Armed conflict and human rights
                "armed conflict", "military operations", "drone strike", "drone strikes",
                "kinetic force", "kinetic use of force", "instantaneous act",
                "bombing", "airstrike", "air strike", "aerial bombardment",
                "fog of war", "military base", "detention facility",
                "occupation", "military occupation", "occupying power",
                "ihrl ihl", "ihl ihrl", "lex specialis", "concurrent application",
                "international humanitarian law", "ihl", "geneva conventions",
                "hague law", "targeting", "proportionality in armed conflict",

                # Key rights
                "right to life", "article 2", "article 2 echr", "positive obligations",
                "substantive limb", "procedural limb", "duty to investigate",
                "effective investigation", "independent investigation", "mccann",
                "torture", "article 3", "article 3 echr", "inhuman treatment", "degrading treatment",
                "absolute prohibition", "ireland v uk", "selmouni",
                "liberty", "article 5", "article 5 echr", "detention", "deprivation of liberty",
                "habeas corpus", "lawful detention", "security detention",
                "fair trial", "article 6", "due process",
                "private life", "article 8", "family life",
                "expression", "article 10", "freedom of expression",
                "assembly", "article 11", "association",
                "discrimination", "article 14", "equality",
                "derogation", "article 15", "emergency", "public emergency",
                "margin of appreciation", "proportionality",
                "positive obligation", "negative obligation",

                # Detention and ill-treatment
                "detention abroad", "military detention", "internment",
                "ill-treatment", "mistreatment", "abuse", "beaten", "beating",
                "access to lawyer", "legal representation", "incommunicado",
                "conditions of detention",

                # Procedural obligations
                "duty to investigate", "procedural obligation", "procedural limb",
                "effective investigation", "independent", "impartial", "thorough",
                "adequate", "prompt", "public scrutiny", "next of kin",
                "mccann v uk", "mccann", "osman v uk", "kaya v turkey",

                # Key concepts
                "jus cogens", "peremptory norm", "erga omnes",
                "non-refoulement", "refugee", "asylum",
                "state responsibility", "ilc articles",
                "exhaustion of local remedies",
                "living instrument", "dynamic interpretation",
                "practical and effective", "autonomous meaning",
            ],

            # ================================================================================
            # CORPORATE CRIMINAL LIABILITY
            # ================================================================================
            "corporate_criminal": [
                # Core concepts
                "corporate criminal liability", "corporate liability", "corporate crime",
                "corporate criminality", "criminal liability of corporations",
                "corporate manslaughter", "corporate killing",
                "corporate homicide", "corporate responsibility",

                # Identification doctrine
                "identification doctrine", "identification principle",
                "directing mind", "directing mind and will",
                "alter ego", "controlling mind", "embodiment",
                "tesco v nattrass", "tesco supermarkets v nattrass", "nattrass",
                "lennards", "lennard's carrying", "meridian",
                "meridian global", "meridian global funds",
                "bolton engineering", "denning", "hl bolton",

                # Attribution and vicarious liability
                "attribution", "attribution of liability", "corporate attribution",
                "vicarious liability", "respondeat superior",
                "aggregation", "aggregated knowledge", "collective knowledge",
                "corporate culture", "organisational culture",
                "corporate ethos", "systemic failure", "systemic failures",

                # Failure to prevent model
                "failure to prevent", "failure to prevent fraud",
                "failure to prevent bribery", "failure to prevent offences",
                "reasonable procedures", "reasonable prevention procedures",
                "adequate procedures", "compliance procedures", "compliance defence",
                "associated person", "associate",

                # Key legislation
                "economic crime and corporate transparency act", "eccta", "eccta 2023",
                "bribery act", "bribery act 2010", "section 7",
                "corporate manslaughter act", "cmcha", "cmcha 2007",
                "corporate manslaughter and corporate homicide act",
                "health and safety at work act", "hswa", "hswa 1974",
                "health and safety offences", "gross breach",
                "senior management", "senior manager", "senior managers",
                "management failure",

                # Key cases
                "tesco", "nattrass", "meridian", "p&o ferries", "herald of free enterprise",
                "r v hm coroner for east kent", "oll ltd", "r v oll",
                "r v p&o european ferries", "p&o", "zeebrugge",
                "r v adomako", "adomako", "gross negligence manslaughter",
                "r v kite", "kite", "stoddart",
                "transco", "r v transco", "balfour beatty",
                "r v cotswold geotechnical", "cotswold geotechnical",
                "r v jmw farms", "jmw farms",
                "deferred prosecution agreement", "dpa", "serious fraud office", "sfo",
                "rolls royce", "airbus", "standard chartered",
                "g4s", "serco", "carillion",

                # Corporate governance
                "board", "board of directors", "directors",
                "non-executive directors", "ned", "chairman", "ceo",
                "corporate governance", "uk corporate governance code",
                "corporate veil", "lifting the veil", "piercing the veil",
                "separate legal personality", "salomon",
                "compliance", "compliance officer", "compliance function",
                "whistleblowing", "whistleblower", "speak up",

                # Reform and criticism
                "corporate shield", "accountability gap", "prosecution gap",
                "law commission", "corporate liability reform",
                "individual liability", "director disqualification",
                "deterrence", "corporate punishment", "corporate fines",
                "deferred prosecution", "non-prosecution agreement",
                "equity fine", "community service order",
                "corporate probation", "corporate rehabilitation",
            ],

            # ================================================================================
            # PUBLIC INTERNATIONAL LAW (General)
            # ================================================================================
            "public_international_general": [
                # Sources
                "sources of international law", "article 38", "icj statute",
                "custom", "customary international law", "opinio juris", "state practice",
                "treaty", "treaties", "convention", "protocol",
                "general principles", "subsidiary means",
                "soft law", "resolution", "declaration",

                # Subjects
                "state", "statehood", "montevideo convention",
                "recognition", "de jure", "de facto",
                "international organization", "international organisations",
                "united nations", "un charter",
                "non-state actors", "ngo",

                # State responsibility
                "state responsibility", "ilc articles", "wrongful act",
                "attribution", "breach", "circumstances precluding wrongfulness",
                "countermeasures", "reparation", "restitution", "satisfaction",

                # Use of force
                "use of force", "article 2(4)", "jus ad bellum",
                "self-defence", "article 51", "armed attack",
                "security council", "chapter vii", "collective security",
                "humanitarian intervention", "r2p", "responsibility to protect",
                "icj", "international court of justice",
                "nicaragua", "oil platforms", "armed activities",

                # Immunities
                "state immunity", "sovereign immunity",
                "restrictive theory", "commercial exception",
                "diplomatic immunity", "vcdr", "consular immunity",
                "head of state immunity", "act of state",
            ],

            # ================================================================================
            # EU LAW - Comprehensive coverage
            # ================================================================================
            "eu_law": [
                # Core concepts and sources
                "eu law", "european union law", "community law",
                "tfeu", "treaty on the functioning", "teu", "treaty on european union",
                "directive", "directives", "regulation", "regulations", "decision",
                "primary law", "secondary law", "soft law",
                "acquis communautaire", "acquis",
                "lisbon treaty", "maastricht", "amsterdam", "nice",

                # Institutions
                "european commission", "commission", "college of commissioners",
                "european parliament", "parliament", "mep",
                "council of the european union", "council", "qualified majority",
                "european council", "heads of state",
                "court of justice", "cjeu", "ecj", "european court of justice",
                "general court", "cfi", "court of first instance",
                "advocate general", "ag opinion",

                # Supremacy and Direct Effect
                "supremacy", "primacy", "costa v enel", "costa enel",
                "simmenthal", "internationale handelsgesellschaft",
                "direct effect", "van gend en loos", "van gend",
                "vertical direct effect", "horizontal direct effect",
                "direct applicability",
                "indirect effect", "von colson", "marleasing", "consistent interpretation",
                "incidental horizontal effect", "cia security", "unilever",

                # State Liability
                "state liability", "francovich", "francovich liability",
                "brasserie du pecheur", "factortame iii",
                "sufficiently serious breach", "manifest and grave",
                "dillenkofer", "kobler",

                # Preliminary References
                "preliminary reference", "article 267", "preliminary ruling",
                "cilfit", "acte clair", "acte eclaire",
                "mandatory reference", "discretionary reference",
                "courts of last resort", "highest court",
                "foto-frost", "validity",

                # Free Movement of Goods
                "free movement of goods", "article 34", "article 35", "article 36",
                "quantitative restrictions", "measures having equivalent effect", "meqr",
                "dassonville", "dassonville formula",
                "cassis de dijon", "cassis", "mutual recognition", "mandatory requirements",
                "keck", "selling arrangements", "product requirements",
                "commission v italy", "italian art",
                "justification", "public morality", "public policy", "public security",
                "health", "protection of national treasures",
                "proportionality", "least restrictive means",

                # Free Movement of Persons
                "free movement of persons", "free movement of workers",
                "article 45", "worker", "lawrie-blum", "lawrie blum",
                "citizenship", "union citizenship", "article 20", "article 21",
                "citizens directive", "directive 2004/38",
                "right of residence", "permanent residence", "five years",
                "family members", "derived rights",
                "public policy exception", "genuine present and sufficiently serious threat",
                "van duyn", "bonsignore", "orfanopoulos",
                "expulsion", "deportation", "procedural safeguards",

                # Freedom of Establishment and Services
                "freedom of establishment", "article 49",
                "freedom to provide services", "article 56",
                "gebhard", "gebhard test", "four conditions",
                "centros", "uberseering", "inspire art", "company law",
                "posted workers", "laval", "viking",

                # Competition Law (EU)
                "article 101", "article 102", "tfeu competition",
                "anti-competitive agreement", "concerted practice",
                "object or effect", "restriction of competition",
                "de minimis", "appreciable effect",
                "block exemption", "vertical agreements",
                "abuse of dominant position", "dominant position",
                "relevant market", "ssnip test",
                "excessive pricing", "predatory pricing", "refusal to supply",
                "essential facilities", "bronner",
                "merger regulation", "eumr", "significant impediment",

                # Brexit and Retained EU Law
                "brexit", "withdrawal agreement", "trade and cooperation agreement",
                "retained eu law", "assimilated law", "reul",
                "european union withdrawal act", "euwa 2018",
                "retained eu law act 2023", "reula",
                "interpretation", "general principles",

                # Key cases
                "van gend en loos", "costa v enel", "simmenthal",
                "francovich", "brasserie", "factortame",
                "dassonville", "cassis de dijon", "keck",
                "defrenne", "marshall", "foster v british gas",
            ],

            # ================================================================================
            # INTERNATIONAL COMMERCIAL ARBITRATION
            # ================================================================================
            "international_arbitration": [
                # Core concepts
                "international arbitration", "commercial arbitration",
                "arbitration", "arbitral tribunal", "arbitrator", "arbitrators",
                "party autonomy", "seat of arbitration", "lex arbitri",
                "institutional arbitration", "ad hoc arbitration",

                # Key instruments
                "new york convention", "nyc", "recognition and enforcement",
                "uncitral model law", "model law",
                "uncitral arbitration rules", "uncitral rules",

                # Institutions
                "icc", "international chamber of commerce", "icc arbitration",
                "lcia", "london court of international arbitration",
                "siac", "singapore international arbitration centre",
                "hkiac", "hong kong international arbitration centre",
                "scc", "stockholm chamber of commerce",
                "cietac", "china international economic",
                "icsid", "investment arbitration",

                # Arbitration Agreement
                "arbitration agreement", "arbitration clause", "submission agreement",
                "separability", "kompetenz-kompetenz", "competence-competence",
                "fiona trust", "prima facie", "pathological clause",
                "incorporation by reference",
                "multi-party arbitration", "joinder", "consolidation",
                "group of companies", "dow chemical",

                # Arbitral Procedure
                "constitution of tribunal", "appointment", "challenge",
                "impartiality", "independence", "iba guidelines",
                "disclosure", "conflict of interest",
                "procedural order", "terms of reference",
                "document production", "iba rules on evidence",
                "witness statement", "expert witness",
                "hearing", "oral argument", "post-hearing brief",

                # Applicable Law
                "applicable law", "governing law", "choice of law",
                "lex mercatoria", "transnational law",
                "voie directe", "conflict of laws rules",
                "substantive law", "procedural law",

                # Award
                "arbitral award", "final award", "partial award", "interim award",
                "reasons", "operative part", "dispositif",
                "res judicata", "issue estoppel",
                "correction", "interpretation", "additional award",

                # Enforcement and Challenge
                "enforcement", "recognition", "exequatur",
                "article v", "grounds for refusal",
                "public policy", "arbitrability",
                "incapacity", "invalid agreement",
                "due process", "inability to present case",
                "excess of authority", "ultra petita",
                "set aside", "annulment", "challenge",
                "serious irregularity", "section 68", "arbitration act 1996",
                "appeal on point of law", "section 69",

                # Emergency and Interim Measures
                "emergency arbitrator", "interim measures", "conservatory measures",
                "anti-suit injunction", "anti-arbitration injunction",
                "security for costs",

                # Third Party Funding
                "third party funding", "tpf", "litigation funding",
                "disclosure of funding", "adverse costs",

                # Key UK cases
                "fiona trust v privalov", "fiona trust",
                "dallah v pakistan", "dallah",
                "enka v chubb", "enka",
                "kabab-ji v kout", "kabab-ji",
                "halliburton v chubb", "halliburton",
                "jivraj v hashwani", "jivraj",
            ],

            # ================================================================================
            # RESTITUTION / UNJUST ENRICHMENT
            # ================================================================================
            "restitution": [
                # Core concepts
                "restitution", "unjust enrichment", "unjust benefit",
                "enrichment", "enriched", "at the expense of",
                "unjust factor", "absence of basis",
                "restitutionary", "disgorgement",

                # The Birks structure
                "birks", "peter birks", "unjust factors",
                "absence of basis", "civilian approach",
                "four questions", "enrichment enquiry",

                # Unjust factors
                "mistake", "mistake of fact", "mistake of law",
                "kleinwort benson", "kleinwort benson v lincoln",
                "failure of consideration", "total failure", "partial failure",
                "failure of basis", "roxborough",
                "duress", "undue influence", "exploitation",
                "legal compulsion", "necessity",
                "free acceptance", "officiousness",
                "ignorance", "powerlessness",

                # Defences
                "change of position", "lipkin gorman v karpnale", "lipkin gorman",
                "good faith", "disenrichment",
                "estoppel", "ministerial receipt",
                "passing on", "bona fide purchaser",
                "illegality", "patel v mirza",
                "limitation", "laches",
                "counter-restitution impossible",

                # Remedies
                "personal restitution", "proprietary restitution",
                "quantum meruit", "reasonable value",
                "account of profits", "disgorgement",
                "constructive trust", "resulting trust",
                "subrogation", "equitable lien",
                "tracing", "following", "claiming",

                # Specific contexts
                "void contract", "voidable contract",
                "frustrated contract", "lr(fc)a 1943",
                "ultra vires contract", "incapacity",
                "unenforceable contract", "illegal contract",
                "ministerial receipt", "agent",
                "benefits in kind", "services", "goods",

                # Key cases
                "lipkin gorman", "woolwich", "woolwich v irc",
                "kleinwort benson", "deutsche morgan grenfell",
                "banque financiere v parc", "banque financiere",
                "benedetti v sawiris", "benedetti",
                "investec v glenalla", "investec",
                "menelaou v bank of cyprus", "menelaou",
                "swynson v lowick rose", "swynson",
                "test claimants v hmrc", "test claimants",
            ],

            # ================================================================================
            # ADMINISTRATIVE LAW (UK) - Expanded
            # ================================================================================
            "administrative_law_uk": [
                # Core concepts
                "administrative law", "judicial review", "jr",
                "public law", "public authority", "amenability",
                "prerogative remedy", "prerogative order",

                # Procedural aspects
                "permission", "leave", "permission stage",
                "pre-action protocol", "letter before claim",
                "claim form", "statement of facts and grounds",
                "acknowledgement of service", "summary grounds",
                "time limit", "promptly", "three months",
                "section 31", "senior courts act 1981",

                # Standing
                "standing", "sufficient interest", "locus standi",
                "victim test", "hra standing",
                "public interest standing", "representative standing",
                "pressure group", "ngo standing",
                "fleet street casuals", "irc v national federation",
                "world development movement", "pergau dam",
                "child poverty action group",

                # Grounds of Review - Illegality
                "illegality", "ultra vires", "error of law",
                "jurisdictional error", "non-jurisdictional error",
                "anisminic", "ouster clause",
                "cart", "unappealable", "second-tier appeals",
                "relevant considerations", "irrelevant considerations",
                "padfield", "improper purpose",
                "fettering discretion", "rigid policy",
                "british oxygen", "kynoch",
                "venables", "mandatory consideration",
                "delegation", "acting under dictation",
                "carltona", "alter ego",

                # Grounds of Review - Irrationality
                "irrationality", "wednesbury", "unreasonableness",
                "wednesbury unreasonable", "so unreasonable",
                "associated provincial picture houses",
                "super-wednesbury", "sub-wednesbury",
                "anxious scrutiny", "ex p smith",
                "common law rights", "kennedy",
                "proportionality", "proportionate",
                "de smith", "four-stage test",
                "suitability", "necessity", "fair balance",
                "daly", "bank mellat", "pham",

                # Grounds of Review - Procedural Impropriety
                "procedural impropriety", "natural justice",
                "procedural fairness", "fair procedure",
                "audi alteram partem", "right to be heard",
                "ridge v baldwin", "administrative v judicial",
                "oral hearing", "written representations",
                "reasons", "duty to give reasons",
                "doody", "osborn", "oakley v south cambridgeshire",
                "disclosure", "gist", "closed material",
                "nemo iudex in causa sua", "bias", "rule against bias",
                "automatic disqualification", "pinochet",
                "apparent bias", "porter v magill",
                "fair-minded observer", "real possibility",
                "predetermination", "closed mind",

                # Legitimate Expectation
                "legitimate expectation", "substantive", "procedural",
                "coughlan", "ex p coughlan",
                "clear and unambiguous", "promise", "representation",
                "nadarajah", "abuse of power",
                "bibby", "paponette", "united policyholders",
                "machinists", "unfairness",
                "detrimental reliance",

                # Human Rights and Judicial Review
                "human rights act 1998", "hra", "section 6",
                "public authority", "hybrid public authority",
                "section 3", "interpretive obligation",
                "section 4", "declaration of incompatibility",
                "proportionality", "structured proportionality",
                "margin of appreciation", "deference",
                "constitutional rights", "common law rights",
                "principle of legality", "ex p simms",
                "fundamental rights", "unwritten constitution",

                # Remedies
                "quashing order", "certiorari",
                "mandatory order", "mandamus",
                "prohibiting order", "prohibition",
                "declaration", "declaratory relief",
                "injunction", "interim relief",
                "damages", "just satisfaction",
                "section 31(2a)", "highly likely",
                "no difference", "outcome focused",

                # Ouster Clauses
                "ouster clause", "privative clause",
                "anisminic", "nullity", "void",
                "privacy international", "adams v adams",
                "error of law on face of record",
                "presumption against ousting",

                # Key modern cases
                "unison", "access to justice",
                "miller", "miller i", "miller ii", "prorogation",
                "evans", "prince charles letters",
                "datafin", "amenability", "public function",
                "eba", "environment agency", "aarhus",
            ],

            # ================================================================================
            # EQUITY (General Principles)
            # ================================================================================
            "equity_general": [
                # Core maxims
                "equity", "equitable", "conscience",
                "maxims of equity", "equitable maxim",
                "equity will not suffer a wrong without a remedy",
                "equity follows the law", "equity acts in personam",
                "he who seeks equity must do equity",
                "he who comes to equity must come with clean hands",
                "clean hands", "unclean hands",
                "equity looks to intent rather than form",
                "equity looks on that as done which ought to be done",
                "walsh v lonsdale", "conversion",
                "equity will not assist a volunteer",
                "milroy v lord", "constitution",
                "where equities are equal the first in time prevails",
                "where equities are equal the law prevails",
                "equity imputes an intention to fulfil an obligation",
                "delay defeats equity", "laches",

                # Equitable interests
                "equitable interest", "equitable title",
                "beneficial interest", "beneficial owner",
                "mere equity", "equity's darling",
                "bona fide purchaser", "notice",
                "actual notice", "constructive notice", "imputed notice",

                # Equitable remedies
                "specific performance", "injunction",
                "rescission", "rectification", "account",
                "equitable compensation", "equitable damages",
                "lord cairns act", "damages in lieu",
                "mandatory injunction", "prohibitory injunction",
                "interim injunction", "interlocutory",
                "american cyanamid", "balance of convenience",
                "mareva", "freezing order", "freezing injunction",
                "anton piller", "search order",
                "quia timet", "apprehended wrong",

                # Bars to equitable relief
                "laches", "delay", "acquiescence",
                "hardship", "impossibility", "futility",
                "clean hands", "unclean hands",
                "mutuality", "want of mutuality",
                "damages adequate",

                # Fiduciary relationships
                "fiduciary", "fiduciary duty", "fiduciary relationship",
                "no conflict", "no profit",
                "undivided loyalty", "good faith",
                "bribes", "secret commission",
                "fhp v sinclair", "sinclair v versailles", "fhr european ventures",

                # Undue influence
                "undue influence", "presumed undue influence", "actual undue influence",
                "class 1", "class 2a", "class 2b",
                "etridge", "independent advice",
                "manifest disadvantage", "calls for explanation",

                # Unconscionability
                "unconscionable", "unconscionability",
                "unconscionable bargain", "catching bargain",
                "fry v lane", "poverty and ignorance",
                "alec lobb", "procedural and substantive",
            ],
        }

        def score_domain(domain: str) -> int:
            hits = 0
            for kw in domain_keywords[domain]:
                if not kw:
                    continue
                kw_norm = kw.lower()
                # For very short alphanumeric keywords (e.g. "ai", "ip", "cma"), require word-boundary matches.
                if len(kw_norm) <= 3 and re.fullmatch(r"[a-z0-9]+", kw_norm):
                    if re.search(rf"\b{re.escape(kw_norm)}\b", query_lower):
                        hits += 1
                    continue
                if kw_norm in query_lower:
                    hits += 1
            return hits

        domain_scores = []
        for domain in domain_keywords:
            domain_scores.append((domain, score_domain(domain)))
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        best_domain, best_score = domain_scores[0] if domain_scores else (None, 0)
        second_domain, second_score = domain_scores[1] if len(domain_scores) > 1 else (None, 0)
        is_mixed = best_score >= 2 and second_score >= 2 and (best_score - second_score) <= 1

        def infer_category_domain(cat_lower: str) -> Optional[str]:
            if "medicine" in cat_lower or "medical" in cat_lower or "clinical" in cat_lower:
                return "medical"
            if "biolaw" in cat_lower or "bioethics" in cat_lower:
                return "medical"
            if "law and medicine" in cat_lower:
                return "medical"
            if "consumer" in cat_lower or "consumer rights" in cat_lower or "cra 2015" in cat_lower or "digital content" in cat_lower:
                return "consumer_cra"
            if any(k in cat_lower for k in ["cyber", "computer misuse", "computer crime", "hacking", "internet security", "ddos", "denial of service", "ethical hacking"]):
                return "cyber_cma"
            if "defamation" in cat_lower or "libel" in cat_lower or "slander" in cat_lower:
                return "defamation"
            if "private international" in cat_lower or "conflict of laws" in cat_lower:
                return "private_international"
            if "merger" in cat_lower or "merger control" in cat_lower or "enterprise act" in cat_lower or "undertakings in lieu" in cat_lower:
                return "merger_control_uk"
            if "discrimination" in cat_lower or "equality act" in cat_lower:
                return "employment_discrimination"
            # AI/robotics/cyber-tech materials are often separate folders in this repo.
            if "ai related" in cat_lower or "robotics" in cat_lower or ("artificial intelligence" in cat_lower and "data protection" not in cat_lower and "gdpr" not in cat_lower):
                return "ai_techlaw"
            # Media/privacy (MPI) vs data protection: distinguish by signals like "media/press" vs "gdpr/data protection".
            if any(k in cat_lower for k in ["media", "press", "misuse", "breach of confidence", "super-injunction", "super injunction", "privacy law"]):
                return "media_privacy"
            if "public international" in cat_lower or "international law" in cat_lower:
                return "public_international"
            if "judicial review" in cat_lower and any(k in cat_lower for k in ["usa", "u.s.", "united states", "us "]):
                return "public_law_us"
            if "constitutional" in cat_lower or "public law" in cat_lower or "judicial review" in cat_lower:
                return "public_law_uk"
            # Repo contains a folder with misspelling "Interllectual".
            if "copyright" in cat_lower:
                return "ip"
            if "intellectual property" in cat_lower or "interllectual" in cat_lower or "ip" in cat_lower:
                return "ip"
            if "family" in cat_lower or "matrimonial" in cat_lower:
                return "family"
            if "evidence" in cat_lower:
                return "evidence"
            if "land" in cat_lower or "property" in cat_lower or "conveyancing" in cat_lower:
                return "land"
            if "competition" in cat_lower or "antitrust" in cat_lower:
                return "competition"
            if any(k in cat_lower for k in ["gdpr", "data protection", "dpa 2018", "data privacy", "ico", "pecr", "dsar", "subject access"]):
                return "data_privacy"
            # If a category is labelled "privacy" but is not clearly GDPR/data-protection, treat it as media/privacy.
            if "privacy" in cat_lower:
                return "media_privacy"
            if "mediation" in cat_lower or "arbitration" in cat_lower:
                return "adr"
            if "commercial" in cat_lower:
                return "commercial"
            # "Business law" is a catch-all folder in this repo; don't over-constrain it to a single domain.
            if "business law" in cat_lower:
                return None
            if "employment" in cat_lower or "conditions of employment" in cat_lower:
                return "employment"
            if "pension" in cat_lower:
                return "pensions"
            if "criminal" in cat_lower:
                return "criminal"
            if "trust" in cat_lower:
                return "trusts"
            if "business" in cat_lower or "company" in cat_lower or "corporate" in cat_lower:
                return "company"
            if "tort" in cat_lower:
                return "tort"
            if "contract" in cat_lower:
                return "contract"
            if "eu" in cat_lower:
                return "eu"
            if "insolvency" in cat_lower:
                return "insolvency"
            if "insurance" in cat_lower or "underwriting" in cat_lower:
                return "insurance"
            if "banking" in cat_lower or "finance" in cat_lower or "fca" in cat_lower:
                return "banking_finance"
            if "construction" in cat_lower or "adjudication" in cat_lower:
                return "construction"
            if "planning" in cat_lower or "tcpa" in cat_lower:
                return "planning"
            if "shipping" in cat_lower or "admiralty" in cat_lower or "maritime" in cat_lower:
                return "shipping_admiralty"
            if "tax" in cat_lower or "taxation" in cat_lower or "hmrc" in cat_lower:
                return "tax"
            if "immigration" in cat_lower or "asylum" in cat_lower:
                return "immigration"
            if "environment" in cat_lower or "pollution" in cat_lower:
                return "environmental"
            if "wills" in cat_lower or "probate" in cat_lower or "inheritance" in cat_lower:
                return "wills_probate"
            if "sport" in cat_lower:
                return "sports"
            return None

        if best_domain and best_score >= 2:
            category_domain = infer_category_domain(category_lower)
            # Treat "EU law" materials as aligned when the query is clearly competition law.
            if best_domain == "competition" and category_domain == "eu":
                category_domain = "competition"
            # Treat "EU law" materials as aligned when the query is clearly PIL / use-of-force / immunities.
            if best_domain == "public_international" and category_domain == "eu":
                category_domain = "public_international"
            # Public law categories may be split UK/US; treat them as aligned with the generic public_law domain.
            if best_domain == "public_law" and category_domain in {"public_law_uk", "public_law_us"}:
                category_domain = "public_law"
            if best_domain in {"public_law_uk", "public_law_us"} and category_domain == "public_law":
                category_domain = best_domain
            # Merger control materials often live under generic Competition folders.
            if best_domain == "merger_control_uk" and category_domain in {"competition", "eu"}:
                category_domain = "merger_control_uk"
            # Evidence materials often live under Criminal Law folders in this repo.
            if best_domain == "evidence" and category_domain == "criminal":
                category_domain = "evidence"
            # Wills/probate materials often live under Trusts folders.
            if best_domain == "wills_probate" and category_domain == "trusts":
                category_domain = "wills_probate"
            # Insurance materials often live under Commercial Law.
            if best_domain == "insurance" and category_domain == "commercial":
                category_domain = "insurance"
            # Shipping materials often under Commercial Law.
            if best_domain == "shipping_admiralty" and category_domain == "commercial":
                category_domain = "shipping_admiralty"
            # Tax materials may live under Business/Company law.
            if best_domain == "tax" and category_domain in {"company", "commercial"}:
                category_domain = "tax"
            # Planning materials may live under Land Law.
            if best_domain == "planning" and category_domain == "land":
                category_domain = "planning"
            # Environmental may overlap with planning.
            if best_domain == "environmental" and category_domain == "planning":
                category_domain = "environmental"
            # Immigration may live under Human Rights or Public Law.
            if best_domain == "immigration" and category_domain in {"public_law", "public_law_uk"}:
                category_domain = "immigration"
            if category_domain and category_domain != best_domain:
                if is_mixed:
                    weight *= 0.75
                else:
                    weight *= 0.25 if best_score >= 3 else 0.60
            elif category_domain is None:
                weight *= 0.85 if is_mixed else (0.70 if best_score >= 3 else 0.85)

            # If category aligns with the dominant domain, allow a gentle boost even if token overlap is imperfect.
            if best_domain == "employment" and ("employment" in category_lower or "conditions of employment" in category_lower or "business" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "company" and ("business" in category_lower or "company" in category_lower or "corporate" in category_lower):
                weight = max(weight, CATEGORY_WEIGHTS.get("Business law", 1.0))
            if best_domain == "trusts" and "trust" in category_lower:
                weight = max(weight, CATEGORY_WEIGHTS.get("Trusts law", 1.0))
            if best_domain == "competition" and ("competition" in category_lower or "eu" in category_lower):
                weight = max(weight, CATEGORY_WEIGHTS.get("Competition Law", 1.0))
            if best_domain == "data_privacy" and ("data protection" in category_lower or "ai" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "media_privacy" and ("media" in category_lower or "privacy" in category_lower or "press" in category_lower or "human rights" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "defamation" and ("defamation" in category_lower or "libel" in category_lower or "slander" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "medical" and ("medicine" in category_lower or "medical" in category_lower or "law and medicine" in category_lower or "clinical" in category_lower or "biolaw" in category_lower):
                weight = max(weight, 1.15)
            if best_domain == "criminal" and ("criminal" in category_lower or "evidence" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "merger_control_uk" and ("competition" in category_lower or "merger" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "employment_discrimination" and ("employment" in category_lower or "discrimination" in category_lower or "equality" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "private_international" and ("private international" in category_lower or "conflict" in category_lower or "international" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "ai_techlaw" and ("ai" in category_lower or "robotics" in category_lower or "technology" in category_lower):
                weight = max(weight, 1.05)
            if best_domain == "consumer_cra" and ("consumer" in category_lower or "contract" in category_lower or "commercial" in category_lower):
                weight = max(weight, 1.05)
            if best_domain == "cyber_cma" and ("cyber" in category_lower or "computer" in category_lower or "technology" in category_lower or "ai" in category_lower):
                weight = max(weight, 1.05)
            if best_domain == "commercial" and "commercial" in category_lower:
                weight = max(weight, 1.1)
            if best_domain == "adr" and "mediation" in category_lower:
                weight = max(weight, 1.1)
            if best_domain == "public_international" and ("international" in category_lower or "eu" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "ip" and ("intellectual" in category_lower or "commercial" in category_lower or "business" in category_lower):
                weight = max(weight, 1.05)
            if best_domain == "family" and ("family" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "evidence" and ("criminal" in category_lower or "evidence" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "insurance" and ("insurance" in category_lower or "commercial" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "banking_finance" and ("banking" in category_lower or "finance" in category_lower or "commercial" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "construction" and ("construction" in category_lower or "commercial" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "planning" and ("planning" in category_lower or "land" in category_lower or "property" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "shipping_admiralty" and ("shipping" in category_lower or "admiralty" in category_lower or "commercial" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "tax" and ("tax" in category_lower or "revenue" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "immigration" and ("immigration" in category_lower or "human rights" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "environmental" and ("environment" in category_lower or "planning" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "wills_probate" and ("wills" in category_lower or "probate" in category_lower or "trust" in category_lower):
                weight = max(weight, 1.1)
            if best_domain == "sports" and ("sport" in category_lower):
                weight = max(weight, 1.1)

        # Apply predefined weights only when the query actually signals that domain.
        for cat_key, predefined_weight in CATEGORY_WEIGHTS.items():
            if cat_key == "default":
                continue
            if cat_key.lower() not in category_lower:
                continue

            key_tokens = {t for t in re.findall(r"[a-z]+", cat_key.lower()) if t not in CATEGORY_MATCH_STOPWORDS}
            if key_tokens and (query_tokens & key_tokens):
                weight = max(weight, predefined_weight)

        return max(0.1, weight)

    # ============================================================================
    # CROSS-REFERENCE / CITATION GRAPH BOOSTING (Improvement 4)
    # ============================================================================

    _CITATION_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "citation_graph.pkl")

    # Regex patterns for UK case citations found in chunk text
    _CASE_CITE_PATTERNS = [
        # Neutral citations: [2020] UKSC 5, [2003] EWCA Civ 1140, etc.
        re.compile(r'\[\d{4}\]\s*(?:UKSC|UKHL|UKPC|EWCA\s*(?:Civ|Crim)|EWHC)\s*\d+', re.IGNORECASE),
        # Law report citations: [2020] 1 AC 123, [2019] 2 WLR 456
        re.compile(r'\[\d{4}\]\s*\d?\s*(?:AC|WLR|All\s*ER|QB|KB|Ch|Fam|Lloyd|BCLC|BCC)\s*\d+', re.IGNORECASE),
        # Older-style citations: (1932) AC 562
        re.compile(r'\(\d{4}\)\s*\d?\s*(?:AC|WLR|All\s*ER|QB|KB|Ch|Fam)\s*\d+', re.IGNORECASE),
    ]

    @staticmethod
    def extract_case_citations(text: str) -> set:
        """Extract case citation strings from a chunk of text."""
        citations = set()
        for pat in RAGService._CASE_CITE_PATTERNS:
            for m in pat.finditer(text):
                # Normalize whitespace in the citation
                citations.add(re.sub(r'\s+', ' ', m.group(0)).strip())
        return citations

    def build_citation_graph(self) -> Dict[str, set]:
        """
        Build a mapping from each chunk_id to the set of case citations it contains.
        Persists to disk so it only needs rebuilding after re-indexing.
        """
        # Try loading from cache first
        if os.path.exists(self._CITATION_GRAPH_PATH):
            try:
                with open(self._CITATION_GRAPH_PATH, 'rb') as f:
                    graph = pickle.load(f)
                if isinstance(graph, dict) and len(graph) > 0:
                    self._citation_graph = graph
                    return graph
            except Exception:
                pass

        # Build fresh from ChromaDB
        graph: Dict[str, set] = {}
        try:
            total = self.collection.count()
            if total == 0:
                self._citation_graph = graph
                return graph

            batch_size = 500
            for offset in range(0, total, batch_size):
                batch = self.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=['documents']
                )
                for cid, doc in zip(batch['ids'], batch['documents']):
                    cites = self.extract_case_citations(doc or "")
                    if cites:
                        graph[cid] = cites
        except Exception as e:
            print(f"⚠️ Citation graph build error: {e}")

        self._citation_graph = graph

        # Persist
        try:
            with open(self._CITATION_GRAPH_PATH, 'wb') as f:
                pickle.dump(graph, f)
        except Exception:
            pass

        return graph

    def _apply_citation_boost(self, results: list) -> list:
        """
        Re-rank retrieval results by boosting chunks that share case citations
        with the top-scoring chunks. This surfaces related authorities that the
        semantic + BM25 scoring might have ranked lower.

        Boost factor: 1.0 + 0.05 * shared_citation_count (capped at 1.25)
        Only applies to chunks outside the top-5 (top-5 stay untouched).
        """
        if not hasattr(self, '_citation_graph'):
            try:
                self.build_citation_graph()
            except Exception:
                return results

        graph = getattr(self, '_citation_graph', {})
        if not graph or len(results) <= 5:
            return results

        # Collect citations from top-5 results
        top_cites: set = set()
        for r in results[:5]:
            top_cites |= graph.get(r.chunk_id, set())

        if not top_cites:
            return results

        # Boost lower-ranked results that share citations with top-5
        boosted = list(results[:5])
        rest = []
        for r in results[5:]:
            chunk_cites = graph.get(r.chunk_id, set())
            shared = len(chunk_cites & top_cites)
            if shared > 0:
                boost = min(1.25, 1.0 + 0.05 * shared)
                r = RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    metadata=r.metadata,
                    semantic_score=r.semantic_score,
                    bm25_score=r.bm25_score,
                    category_weight=r.category_weight,
                    final_score=r.final_score * boost
                )
            rest.append(r)

        # Re-sort the rest by boosted score
        rest.sort(key=lambda x: x.final_score, reverse=True)
        return boosted + rest

    def hybrid_search(
        self,
        query: str,
        max_results: int = 20,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        max_per_document: int = MAX_CHUNKS_PER_DOCUMENT,
        semantic_weight: float = SEMANTIC_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        query_type: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Features:
        1. HYBRID SEARCH: Combines ChromaDB embeddings with BM25 keyword matching
        2. RELEVANCE THRESHOLD: Filters out low-quality matches
        3. DOCUMENT DIVERSITY: Limits chunks from any single document
        4. CATEGORY WEIGHTING: Boosts results from relevant categories
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            relevance_threshold: Minimum score threshold (0-1)
            max_per_document: Maximum chunks from single document
            semantic_weight: Weight for semantic search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            
        Returns:
            List of RetrievalResult objects sorted by final score
        """
        # Get semantic results
        semantic_results = self._get_semantic_results(query, n_results=max_results * 3)
        
        # Get BM25 results
        bm25_results = self._get_bm25_results(query, n_results=max_results * 3)
        
        # Combine all chunk IDs
        all_chunk_ids = set(semantic_results.keys()) | set(bm25_results.keys())
        
        # Calculate combined scores
        results = []
        for chunk_id in all_chunk_ids:
            # Get semantic score (default 0 if not in results)
            sem_data = semantic_results.get(chunk_id, {'score': 0, 'content': '', 'metadata': {}})
            semantic_score = sem_data['score']
            
            # Get BM25 score
            bm25_score = bm25_results.get(chunk_id, 0)
            
            # Get content and metadata
            if chunk_id in semantic_results:
                content = sem_data['content']
                metadata = sem_data['metadata']
            else:
                # Need to fetch from ChromaDB
                result = self.collection.get(ids=[chunk_id], include=['documents', 'metadatas'])
                content = result['documents'][0] if result['documents'] else ''
                metadata = result['metadatas'][0] if result['metadatas'] else {}
            
            # Calculate category weight
            category = metadata.get('category', '')
            category_weight = self._get_category_weight(query, category)

            # Penalize low-quality chunks and weight by document type.
            quality_mult = self._chunk_quality_multiplier(content)
            if quality_mult <= 0.0:
                continue
            doc_weight = self._doc_type_weight(query_type, metadata)
            
            # Calculate final score
            base_score = (semantic_score * semantic_weight) + (bm25_score * bm25_weight)
            final_score = base_score * category_weight * doc_weight * quality_mult
            
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=content,
                metadata=metadata,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                category_weight=category_weight,
                final_score=final_score
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Apply cross-reference citation boosting (Improvement 4)
        # Chunks sharing case citations with top results get a score bump.
        results = self._apply_citation_boost(results)

        # Apply relevance threshold filtering
        results = [r for r in results if r.final_score >= relevance_threshold]

        # Apply document diversity limiting (dedupe "copy" files and duplicate chunks)
        def canonical_document_key(metadata: Dict[str, Any]) -> str:
            doc_name = (metadata or {}).get('document_name') or ''
            if not doc_name:
                return (metadata or {}).get('document_id', 'unknown')
            # Normalize whitespace/punctuation so that "X .pdf" and "X.pdf" canonicalize identically.
            doc_name = re.sub(r"\s+", " ", doc_name).strip()
            doc_name = re.sub(r"\s+\.", ".", doc_name)  # "X .pdf" -> "X.pdf"
            base, ext = os.path.splitext(doc_name)
            base = re.sub(r'\s+copy(?:\s*\d+)?$', '', base, flags=re.IGNORECASE)
            base = re.sub(r'\s*\(\s*copy\s*\d+\s*\)$', '', base, flags=re.IGNORECASE)
            base = re.sub(r"[.\s]+$", "", base).strip()
            ext = ext.strip().lower()
            return (base + ext).strip().lower()

        def document_id(metadata: Dict[str, Any], fallback: str) -> str:
            return (metadata or {}).get('document_id') or (metadata or {}).get('document_name') or fallback

        def is_copy_document(doc_name: str) -> bool:
            if not doc_name:
                return False
            return bool(re.search(r"\bcopy\b", doc_name, flags=re.IGNORECASE))

        # Choose a single representative document per canonical key (prefer non-copy if available).
        best_non_copy_doc: Dict[str, Tuple[float, str]] = {}
        best_any_doc: Dict[str, Tuple[float, str]] = {}
        for r in results:
            doc_key = canonical_document_key(r.metadata)
            doc_name = (r.metadata or {}).get('document_name', '')
            doc_id = document_id(r.metadata, doc_key)
            score = r.final_score

            prev_any = best_any_doc.get(doc_key)
            if prev_any is None or score > prev_any[0]:
                best_any_doc[doc_key] = (score, doc_id)

            if not is_copy_document(doc_name):
                prev_nc = best_non_copy_doc.get(doc_key)
                if prev_nc is None or score > prev_nc[0]:
                    best_non_copy_doc[doc_key] = (score, doc_id)

        chosen_doc_id: Dict[str, str] = {}
        for doc_key, any_pair in best_any_doc.items():
            chosen_doc_id[doc_key] = (best_non_copy_doc.get(doc_key) or any_pair)[1]

        document_counts = defaultdict(int)
        seen_chunk_hashes = set()
        diverse_results = []
        for result in results:
            doc_key = canonical_document_key(result.metadata)
            doc_id = document_id(result.metadata, doc_key)
            if doc_id != chosen_doc_id.get(doc_key, doc_id):
                continue
            content_hash = hashlib.sha1((result.content or '').encode('utf-8', errors='ignore')).hexdigest()
            if content_hash in seen_chunk_hashes:
                continue

            if document_counts[doc_key] < max_per_document:
                diverse_results.append(result)
                document_counts[doc_key] += 1
                seen_chunk_hashes.add(content_hash)
            
            if len(diverse_results) >= max_results:
                break
        
        return diverse_results
    
    # ============================================================================
    # MULTI-HOP RETRIEVAL (Improvement 5)
    # ============================================================================

    _STATUTE_PATTERNS = [
        re.compile(r'\b(?:Act|Regulation|Order|Rules?)\s+\d{4}\b', re.IGNORECASE),
        re.compile(r'\b(?:Human Rights Act|Equality Act|Companies Act|Insolvency Act|Theft Act|Criminal Justice Act|Mental Capacity Act|Children Act|Consumer Rights Act|Data Protection Act|Employment Rights Act|Land Registration Act|Landlord and Tenant Act|Immigration Act|Nationality, Immigration and Asylum Act|Borders, Citizenship and Immigration Act)\b', re.IGNORECASE),
    ]

    _CASE_NAME_PATTERN = re.compile(
        r'\b([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?)\s+v\s+([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?)\b'
    )

    @staticmethod
    def extract_legal_entities(text: str) -> List[str]:
        """
        Extract case names and statute references from text for multi-hop queries.
        Returns a list of search-ready entity strings.
        """
        entities = []

        # Extract case names (e.g., "Donoghue v Stevenson")
        for m in RAGService._CASE_NAME_PATTERN.finditer(text):
            entities.append(m.group(0).strip())

        # Extract statute references
        for pat in RAGService._STATUTE_PATTERNS:
            for m in pat.finditer(text):
                entities.append(m.group(0).strip())

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for e in entities:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                unique.append(e)
        return unique

    def _multi_hop_retrieval(
        self,
        initial_results: List['RetrievalResult'],
        query: str,
        config: dict,
        max_extra_chunks: int = 8,
        query_type: Optional[str] = None
    ) -> List['RetrievalResult']:
        """
        Perform a second retrieval pass using legal entities extracted from
        the initial results. This helps surface related authorities that
        weren't directly matched by the original query.

        Only activates for complex query types (problem questions & essays
        with high word counts) where multi-authority analysis is expected.

        Args:
            initial_results: Results from the first retrieval pass
            query: Original user query
            config: Retrieval config dict
            max_extra_chunks: Max additional chunks to add
            query_type: Query type string

        Returns:
            Merged list of results (initial + multi-hop, deduplicated)
        """
        # Extract entities from top initial results
        top_text = " ".join(r.content for r in initial_results[:10])
        entities = self.extract_legal_entities(top_text)

        if not entities:
            return initial_results

        # Collect existing chunk IDs to avoid duplicates
        existing_ids = {r.chunk_id for r in initial_results}

        extra_results = []
        # Query for each entity (limit to top 6 entities to avoid excessive queries)
        for entity in entities[:6]:
            try:
                hop_results = self.hybrid_search(
                    query=entity,
                    max_results=5,
                    relevance_threshold=config.get("relevance_threshold", RELEVANCE_THRESHOLD),
                    max_per_document=config.get("max_per_document", MAX_CHUNKS_PER_DOCUMENT),
                    semantic_weight=config.get("semantic_weight", SEMANTIC_WEIGHT),
                    bm25_weight=config.get("bm25_weight", BM25_WEIGHT),
                    query_type=query_type
                )
                for r in hop_results:
                    if r.chunk_id not in existing_ids:
                        existing_ids.add(r.chunk_id)
                        extra_results.append(r)
            except Exception:
                continue

        if not extra_results:
            return initial_results

        # Sort extra results by score, take top N
        extra_results.sort(key=lambda x: x.final_score, reverse=True)
        extra_results = extra_results[:max_extra_chunks]

        return initial_results + extra_results

    # ============================================================================
    # MAIN RETRIEVAL API
    # ============================================================================

    # Query types that benefit from multi-hop retrieval (complex analysis questions)
    _MULTI_HOP_QUERY_TYPES = {
        "pb_2000", "pb_2500", "pb_3000", "pb_3500", "pb_4000", "pb_4500",
        "pb_5000", "pb_5500", "pb_6000", "pb_6500", "pb_7000",
        "essay_2000", "essay_2500", "essay_3000", "essay_3500", "essay_4000",
        "essay_4500", "essay_5000", "essay_5500", "essay_6000", "essay_6500",
        "essay_7000",
    }

    def get_relevant_context(
        self,
        query: str,
        max_chunks: int = 20,
        query_type: str = None,
        max_chars: int = 28000
    ) -> str:
        """
        Get relevant context for a query in a format suitable for LLM prompting.
        
        This is the main API used by gemini_service.py
        
        Args:
            query: The user's question
            max_chunks: Maximum number of chunks to retrieve
            query_type: Type of query for retrieval config selection
            
        Returns:
            Formatted context string for LLM
        """
        # Get query-type specific configuration
        config = get_retrieval_config(query_type) if query_type else {
            "semantic_weight": SEMANTIC_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "max_per_document": MAX_CHUNKS_PER_DOCUMENT
        }
        
        results = self.hybrid_search(
            query=query,
            max_results=max_chunks,
            relevance_threshold=config["relevance_threshold"],
            max_per_document=config["max_per_document"],
            semantic_weight=config["semantic_weight"],
            bm25_weight=config["bm25_weight"],
            query_type=query_type
        )
        
        if not results:
            # Light fallback: relax threshold once so relevant material still surfaces
            # when the initial threshold is too strict for the query.
            fallback_threshold = max(0.0, float(config.get("relevance_threshold", 0.0)) * 0.5)
            if fallback_threshold < float(config.get("relevance_threshold", 0.0)):
                results = self.hybrid_search(
                    query=query,
                    max_results=max_chunks,
                    relevance_threshold=fallback_threshold,
                    max_per_document=config["max_per_document"],
                    semantic_weight=config["semantic_weight"],
                    bm25_weight=config["bm25_weight"],
                    query_type=query_type
                )
            if not results:
                return ""

        # Multi-hop retrieval for complex queries (Improvement 5)
        # For problem questions and essays, extract legal entities from initial
        # results and fetch additional related chunks.
        if query_type and query_type in self._MULTI_HOP_QUERY_TYPES and results:
            try:
                results = self._multi_hop_retrieval(
                    initial_results=results,
                    query=query,
                    config=config,
                    max_extra_chunks=8,
                    query_type=query_type
                )
            except Exception as e:
                print(f"⚠️ Multi-hop retrieval skipped: {e}")

        def _extract_author_hint(text: str, doc_name: str) -> str:
            """Try to extract author surname + short title from chunk content for OSCOLA citation help."""
            import re
            if not text:
                return ""
            # Pattern 1: "Author Name, 'Article Title'" (journal articles)
            m = re.search(r"(?m)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}),\s*'([^']{5,80})'", text[:600])
            if m:
                return f"[Author hint: {m.group(1)}, '{m.group(2)[:50]}']"
            # Pattern 2: "Author Name, Title of Book (Publisher Year)"
            m = re.search(r"(?m)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}),\s+([A-Z][A-Za-z :&-]{5,60})\s*\(", text[:600])
            if m:
                return f"[Author hint: {m.group(1)}, {m.group(2).strip()[:50]}]"
            # Pattern 3: "Firstname Surname*" or "Firstname Surname," at start (common in journal PDFs)
            m = re.search(r"(?m)^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[\*,]", text[:300])
            if m:
                return f"[Author hint: {m.group(1)}]"
            # Pattern 4: author info from doc_name itself if it contains "Author - Title" or "Author, Title"
            m = re.match(r"^(.+?)\s*[-–]\s*(.+?)\.pdf$", doc_name, re.IGNORECASE)
            if m and len(m.group(1)) < 60:
                return f"[Author hint: {m.group(1).strip()}, {m.group(2).strip()[:50]}]"
            return ""

        # Format context for LLM (explicitly marked as internal to reduce echoing in outputs)
        context_parts = [
            "[RAG CONTEXT - INTERNAL - DO NOT OUTPUT]",
            ""
        ]
        
        stitched = 0
        # Conservative prompt-size cap to reduce timeouts and "no answer" failures on long prompts.
        # Note: callers can pass a different max_chars if needed.
        max_chars = int(max_chars) if max_chars else 0
        running_len = sum(len(p) + 1 for p in context_parts)
        for i, result in enumerate(results, 1):
            doc_name = result.metadata.get('document_name', 'Unknown')
            chunk_idx = result.metadata.get('chunk_index', 0)
            total_chunks = result.metadata.get('total_chunks', 1)
            cleaned = _clean_text_for_rag(result.content)
            author_hint = _extract_author_hint(cleaned, doc_name)
            header_line = f"[SOURCE {i}] {doc_name} (chunk {chunk_idx + 1}/{total_chunks})"
            if author_hint:
                header_line += f" {author_hint}"

            # Optional stitch: if the chunk looks truncated, append a snippet of the next chunk from the same doc.
            try:
                needs_stitch = (
                    (len(cleaned) < 250)
                    or cleaned.rstrip().endswith("...")
                    or (cleaned and cleaned[-1] not in ".!?;:")
                )
                if needs_stitch and stitched < 3:
                    doc_id = result.metadata.get("document_id")
                    if doc_id and isinstance(chunk_idx, int) and (chunk_idx + 1) < int(total_chunks or 0):
                        neighbor_id = f"{doc_id}_chunk_{int(chunk_idx) + 1}"
                        neighbor = self.collection.get(ids=[neighbor_id], include=["documents"])
                        if neighbor and neighbor.get("documents"):
                            next_clean = _clean_text_for_rag(neighbor["documents"][0] or "")
                            if next_clean:
                                cleaned = (cleaned.rstrip() + "\n\n[CONTINUED]\n" + next_clean[:800]).strip()
                                stitched += 1
            except Exception:
                pass

            # Enforce max_chars by truncating the last chunk if needed and stopping.
            if max_chars > 0:
                # Reserve space for closing marker and some newlines.
                reserve = len("\n[END RAG CONTEXT]\n") + 10
                available = max_chars - running_len - reserve
                # Need room for header + blank line + content + blank line.
                block_overhead = len(header_line) + 2
                if available <= block_overhead:
                    break
                if len(cleaned) > available - block_overhead:
                    cleaned = (cleaned[: max(0, available - block_overhead)]).rstrip()
                    if cleaned:
                        cleaned += "\n\n[TRUNCATED]"
                    else:
                        break
                    # Add the truncated final block and stop.
                    context_parts.append(header_line)
                    context_parts.append("")
                    context_parts.append(cleaned)
                    context_parts.append("")
                    running_len = sum(len(p) + 1 for p in context_parts)
                    break

            context_parts.append(header_line)
            context_parts.append("")
            context_parts.append(cleaned)
            context_parts.append("")
            running_len += len(header_line) + 1 + len(cleaned) + 2
        
        context_parts.append("[END RAG CONTEXT]")
        
        return "\n".join(context_parts)


# ================================================================================
# MODULE-LEVEL API (for compatibility with existing code)
# ================================================================================

_rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    """Get the singleton RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

def get_relevant_context(query: str, max_chunks: int = 20, query_type: str = None, max_chars: int = 28000) -> str:
    """Get relevant context for a query (convenience function)."""
    return get_rag_service().get_relevant_context(query, max_chunks, query_type, max_chars=max_chars)
