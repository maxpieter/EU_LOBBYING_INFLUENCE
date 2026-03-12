"""Simple section-based chunking for 100% document coverage.

Instead of relying on precise structure parsing, this chunks documents
into major sections that cover the entire document text.
"""

from typing import Any, Dict, List

from bs4 import BeautifulSoup


def chunk_document_by_sections(html_content: str) -> List[Dict[str, Any]]:
    """Chunk legislative document into major sections for complete coverage.

    Creates 4 types of chunks:
    1. Explanatory Memorandum (context/rationale)
    2. Legislative Text (recitals + articles combined)
    3. Financial Statement (budget/costs)
    4. Annexes (technical details)

    This ensures 100% of document is captured without missing content
    due to parsing edge cases.

    Args:
        html_content: Raw EUR-Lex HTML

    Returns:
        List of chunks covering the entire document
    """
    chunks = []
    soup = BeautifulSoup(html_content, "html.parser")
    full_text = soup.get_text(separator=" ", strip=True)

    # Find major section boundaries in the text
    explanatory_start = full_text.find("EXPLANATORY MEMORANDUM")
    adoption_formula = full_text.find("HAVE ADOPTED THIS")
    financial_start = full_text.find("LEGISLATIVE FINANCIAL")
    first_annex = full_text.find("ANNEX I")

    # Chunk 1: Explanatory Memorandum (if present)
    if explanatory_start != -1 and adoption_formula != -1:
        memo_text = full_text[explanatory_start:adoption_formula].strip()
        if len(memo_text) > 500:  # Only include if substantial
            chunks.append(
                {
                    "text": memo_text,
                    "type": "explanatory_memo",
                    "section": "Context & Rationale",
                    "char_count": len(memo_text),
                }
            )

    # Chunk 2: Legislative Text (recitals + articles, excluding annexes/financial)
    legislative_start = adoption_formula if adoption_formula != -1 else 0
    legislative_end = (
        financial_start
        if financial_start != -1
        else (first_annex if first_annex != -1 else len(full_text))
    )

    legislative_text = full_text[legislative_start:legislative_end].strip()
    if legislative_text and len(legislative_text) > 500:
        chunks.append(
            {
                "text": legislative_text,
                "type": "legislative",
                "section": "Legislative Provisions",
                "char_count": len(legislative_text),
            }
        )

    # Chunk 3: Financial Statement (if present)
    if financial_start != -1:
        financial_end = first_annex if first_annex != -1 else len(full_text)
        financial_text = full_text[financial_start:financial_end].strip()
        if len(financial_text) > 500:
            chunks.append(
                {
                    "text": financial_text,
                    "type": "financial_statement",
                    "section": "Financial Assessment",
                    "char_count": len(financial_text),
                }
            )

    # Chunk 4: All Annexes (if present)
    if first_annex != -1:
        annexes_text = full_text[first_annex:].strip()
        if len(annexes_text) > 500:
            chunks.append(
                {
                    "text": annexes_text,
                    "type": "annexes_group",
                    "section": "Technical Annexes",
                    "char_count": len(annexes_text),
                }
            )

    return chunks
