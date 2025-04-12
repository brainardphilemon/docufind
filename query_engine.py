from typing import List, Dict, Any
from langchain.schema import Document


class QueryEngine:
    def __init__(self, document_processor):
        """Initialize the query engine with a document processor"""
        self.document_processor = document_processor

    def search_documents(self, query: str, k: int = 5, query_language: str = None) -> List[Dict[str, Any]]:
        """Search for documents relevant to the query with language support"""
        return self.document_processor.search_documents(query, k, query_language)

    def get_document(self, doc_id: str) -> List[Document]:
        """Get all chunks for a specific document"""
        return self.document_processor.get_document(doc_id)

    def get_similar_documents(self, query: str, current_doc_id: str = None, k: int = 3, query_language: str = None) -> \
    List[Dict[str, Any]]:
        """Get documents similar to the query, excluding the current document with language support"""
        return self.document_processor.get_similar_documents(query, current_doc_id, k, query_language)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to identify key concepts and intent (optional enhancement)"""
        # This is a placeholder for more advanced query analysis
        words = query.lower().split()

        # Simple intent detection
        intent = "informational"
        if any(q in words for q in ["how", "what", "explain", "describe"]):
            intent = "informational"
        elif any(q in words for q in ["compare", "difference", "versus", "vs"]):
            intent = "comparison"
        elif any(q in words for q in ["when", "where", "who"]):
            intent = "factual"

        # Extract potential key terms (very basic approach)
        key_terms = []
        for word in words:
            if len(word) > 4 and word not in ["about", "where", "when", "which", "there", "their", "these", "those"]:
                key_terms.append(word)

        # Detect language
        language = self.document_processor.detect_language(query)

        return {
            "intent": intent,
            "key_terms": key_terms[:5],  # Return up to 5 key terms
            "language": language
        }

    def process_multilingual_query(self, query: str, target_lang: str = None) -> Dict[str, Any]:
        """Process a query with language detection and translation if needed"""
        # Detect language of the query
        query_lang = self.document_processor.detect_language(query)

        processed_query = query

        # If target language specified and different from query language, translate
        if target_lang and query_lang != target_lang:
            try:
                processed_query = self.document_processor.translate_text(
                    query,
                    target_lang=target_lang,
                    source_lang=query_lang
                )
            except Exception as e:
                print(f"Error translating query: {e}")

        return {
            "original_query": query,
            "processed_query": processed_query,
            "original_language": query_lang,
            "target_language": target_lang or query_lang
        }