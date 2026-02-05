"""
Knowledge Base Service for Legal AI
Manages the Law Resources as a default knowledge base for the AI
"""
import json
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class LawResourceEntry:
    id: str
    name: str
    path: str
    category: str
    subcategory: str
    mimeType: str
    size: int

@dataclass 
class LawResourceIndex:
    generatedAt: str
    totalFiles: int
    categories: List[str]
    resources: List[LawResourceEntry]

# Global state
law_resource_index: Optional[LawResourceIndex] = None

def load_law_resource_index() -> Optional[LawResourceIndex]:
    """Load the law resources index from the JSON file"""
    global law_resource_index
    
    if law_resource_index:
        return law_resource_index
    
    index_path = os.path.join(os.path.dirname(__file__), 'law-resources-index.json')
    
    try:
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            resources = [
                LawResourceEntry(
                    id=r.get('id', ''),
                    name=r.get('name', ''),
                    path=r.get('path', ''),
                    category=r.get('category', ''),
                    subcategory=r.get('subcategory', ''),
                    mimeType=r.get('mimeType', ''),
                    size=r.get('size', 0)
                )
                for r in data.get('resources', [])
            ]
            
            law_resource_index = LawResourceIndex(
                generatedAt=data.get('generatedAt', ''),
                totalFiles=data.get('totalFiles', 0),
                categories=data.get('categories', []),
                resources=resources
            )
            
            print(f"📚 Loaded {law_resource_index.totalFiles} law resources from index")
            return law_resource_index
    except Exception as e:
        print(f"Could not load law resources index: {e}")
    
    return None

def get_categories() -> List[str]:
    """Get all available categories"""
    if law_resource_index:
        return law_resource_index.categories
    return []

def get_resources_by_category(category: str) -> List[LawResourceEntry]:
    """Get resources by category"""
    if not law_resource_index:
        return []
    return [r for r in law_resource_index.resources if r.category == category]

def search_resources(query: str) -> List[LawResourceEntry]:
    """Search resources by name"""
    if not law_resource_index:
        return []
    lower_query = query.lower()
    return [
        r for r in law_resource_index.resources 
        if lower_query in r.name.lower() or lower_query in r.category.lower()
    ]

def get_knowledge_base_summary() -> str:
    """Get a comprehensive summary of the knowledge base for the AI system prompt"""
    if not law_resource_index:
        return 'No knowledge base loaded.'
    
    # Build detailed category information with sample documents
    category_details = []
    for cat in law_resource_index.categories:
        resources = [r for r in law_resource_index.resources if r.category == cat]
        sample_docs = '\n'.join([f"    • {r.name}" for r in resources[:5]])
        more_text = f"\n    ... and {len(resources) - 5} more documents" if len(resources) > 5 else ""
        category_details.append(f"""
📁 {cat.upper()} ({len(resources)} documents)
{sample_docs}{more_text}""")
    
    return f"""
================================================================================
DEFAULT KNOWLEDGE BASE: UK LAW RESOURCES LIBRARY
================================================================================

Total Documents Available: {law_resource_index.totalFiles} legal texts, cases, and academic materials

AVAILABLE CATEGORIES:
{''.join(category_details)}

================================================================================
USAGE INSTRUCTIONS (MANDATORY):
================================================================================

1. PRIMARY SOURCE RULE: For ANY legal question, FIRST check if relevant materials exist in this knowledge base. Use these as your primary authoritative sources.

2. CATEGORY MAPPING - When a user asks about:
   • Contract law → Use "Contract law" category documents
   • Torts, negligence, duty of care → Use "Tort law" category documents  
   • Trusts, fiduciary duties, trustees → Use "Trusts law" category documents
   • Pensions, pension schemes, trustees → Use "Pensions Law" category documents
   • Criminal offences, criminal liability → Use "Criminal law" category documents
   • EU law, European Union, Brexit → Use "EU law" category documents
   • Competition, antitrust, monopoly → Use "Competition Law" category documents
   • Commercial transactions, sale of goods → Use "Commercial Law" or "Commercial law revision" documents
   • Business, company law, employment → Use "Business law" category documents
   • AI, data protection, GDPR, privacy → Use "Ai and data protection act" category documents
   • Bioethics, medical law → Use "Biolaw" category documents
   • Mediation, ADR, dispute resolution → Use "International Commercial Mediation" category documents

3. CITATION RULE: When referencing knowledge from this database, cite the document name using OSCOLA format. Example:
   [[{{"ref": "Caparo Industries plc v Dickman [1990] UKHL 2", "doc": "Tort law/Caparo case.pdf", "loc": ""}}]]

4. OSCOLA GUIDE: The knowledge base includes the official OSCOLA 4th Edition referencing guide - use this for all citation formatting.

5. CROSS-REFERENCE: For complex questions spanning multiple areas (e.g., "AI in employment discrimination"), draw from multiple relevant categories.

================================================================================
LAWYER-QUALITY OUTPUT REQUIREMENTS:
================================================================================

A. PRACTICAL APPLICATION
   For every legal concept explained, provide:
   1. A real-world example of how it applies in practice.
   2. The practical implications for clients, businesses, or individuals.
   3. Any relevant statutory provisions or regulations.

B. CRITICAL ANALYSIS (For Essays/Discussions)
   Every essay-style response MUST include:
   1. STRENGTHS of the current legal position.
   2. WEAKNESSES or criticisms in academic literature.
   3. REFORM PROPOSALS (if any exist in the knowledge base or academic discourse).
   4. COMPARATIVE PERSPECTIVES (e.g., how other jurisdictions handle the issue).

C. COUNTER-ARGUMENTS
   For any proposition you make, acknowledge and address potential counter-arguments.
   Use phrases like:
   - "Critics argue that..."
   - "However, this view has been challenged by..."
   - "The opposing position, as articulated by [Author], contends..."

D. PROFESSIONAL VOCABULARY
   Use precise legal terminology:
   - "It is submitted that..." (not "I think")
   - "The better view is..." (not "probably")
   - "On balance, the authorities suggest..." (not "it seems like")
   - "This proposition finds support in..." (not "this is backed by")

E. STRUCTURED RECOMMENDATIONS
   When giving advice, end with a clear structure:
   1. SUMMARY of key legal points.
   2. RISKS and potential liabilities.
   3. RECOMMENDED ACTIONS in priority order.
   4. FURTHER CONSIDERATIONS (e.g., limitation periods, costs, alternatives).

F. ACADEMIC DEPTH
   Always cite at least:
   - 2-3 authoritative cases from the knowledge base.
   - 1-2 academic articles or textbook sources when discussing doctrine.
   - Any relevant statutory provisions with section numbers.

YOU ARE NOW A DISTINGUISHED BARRISTER AND LEGAL SCHOLAR WITH ACCESS TO THIS COMPREHENSIVE UK LAW LIBRARY.
Answer all legal questions with the rigour, depth, and professionalism expected at the highest levels of legal practice.
"""

def get_relevant_resources(query: str, limit: int = 10) -> List[LawResourceEntry]:
    """Get relevant resources for a query (simple keyword matching)"""
    if not law_resource_index:
        return []
    
    lower_query = query.lower()
    keywords = [k for k in lower_query.split() if len(k) > 3]
    
    # Score each resource based on keyword matches
    scored = []
    for resource in law_resource_index.resources:
        score = 0
        resource_text = f"{resource.name} {resource.category} {resource.subcategory}".lower()
        
        for keyword in keywords:
            if keyword in resource_text:
                score += 1
                # Bonus for exact word match
                if keyword in resource_text.split():
                    score += 2
        
        if score > 0:
            scored.append((resource, score))
    
    # Sort by score and return top matches
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, s in scored[:limit]]
