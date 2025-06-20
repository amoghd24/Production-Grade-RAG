"""
Query Expansion Service for Advanced RAG.
Intelligently expands user queries with related terms, synonyms, and context.
"""

import re
from typing import List, Dict, Set, Optional
from src.config.feature_flags import get_feature_flags
from src.utils.logger import LoggerMixin


class QueryExpansionService(LoggerMixin):
    """Expands user queries with related terms and synonyms."""
    
    def __init__(self):
        """Initialize the query expansion service."""
        self.feature_flags = get_feature_flags()
        
        # Domain-specific expansion mappings
        self.expansion_dict = {
            # Technical terms
            'ml': {'machine learning', 'ML', 'artificial intelligence', 'AI'},
            'ai': {'artificial intelligence', 'machine learning', 'ML', 'neural networks'},
            'nn': {'neural networks', 'neural nets', 'deep learning', 'neural network'},
            'cnn': {'convolutional neural network', 'convolutional neural networks', 'CNN', 'convnet'},
            'rnn': {'recurrent neural network', 'recurrent neural networks', 'RNN', 'LSTM', 'GRU'},
            'transformer': {'transformers', 'attention mechanism', 'self-attention', 'BERT', 'GPT'},
            'llm': {'large language model', 'large language models', 'LLM', 'language model'},
            
            # Data science terms
            'model': {'models', 'algorithm', 'algorithms', 'architecture', 'framework'},
            'training': {'train', 'learning', 'optimization', 'gradient descent', 'backpropagation'},
            'data': {'dataset', 'datasets', 'information', 'records', 'samples'},
            'feature': {'features', 'attributes', 'variables', 'inputs', 'dimensions'},
            
            # RAG specific
            'retrieval': {'search', 'query', 'find', 'lookup', 'retrieve'},
            'embedding': {'embeddings', 'vectors', 'vector space', 'representation'},
            'similarity': {'cosine similarity', 'semantic similarity', 'distance', 'relevance'},
            'chunk': {'chunks', 'segments', 'passages', 'documents', 'pieces'},
            
            # General synonyms
            'method': {'approach', 'technique', 'strategy', 'way', 'procedure'},
            'improve': {'enhance', 'optimize', 'better', 'upgrade', 'boost'},
            'performance': {'accuracy', 'results', 'effectiveness', 'quality', 'metrics'},
            'problem': {'issue', 'challenge', 'task', 'difficulty', 'question'},
        }
        
        # Common abbreviations
        self.abbreviations = {
            'NLP': 'natural language processing',
            'CV': 'computer vision', 
            'RL': 'reinforcement learning',
            'DL': 'deep learning',
            'SGD': 'stochastic gradient descent',
            'API': 'application programming interface',
            'GPU': 'graphics processing unit',
            'CPU': 'central processing unit',
        }
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand a query with related terms and synonyms.
        
        Args:
            query: Original query string
            max_expansions: Maximum number of expansion terms to add
            
        Returns:
            List of expanded query variations
        """
        if not self.feature_flags.should_use_advanced_search():
            return [query]  # Return original if feature disabled
        
        original_query = query.strip()
        if not original_query:
            return [query]
        
        # Generate different expansion strategies
        expanded_queries = set([original_query])
        
        # Strategy 1: Synonym expansion
        synonym_expanded = self._expand_with_synonyms(original_query, max_expansions)
        expanded_queries.update(synonym_expanded)
        
        # Strategy 2: Abbreviation expansion
        abbrev_expanded = self._expand_abbreviations(original_query)
        expanded_queries.update(abbrev_expanded)
        
        # Strategy 3: Context-aware expansion
        context_expanded = self._expand_with_context(original_query, max_expansions)
        expanded_queries.update(context_expanded)
        
        result = list(expanded_queries)[:max_expansions + 1]  # Include original + expansions
        
        self.logger.debug(f"Expanded query '{original_query}' to {len(result)} variations")
        return result
    
    def _expand_with_synonyms(self, query: str, max_expansions: int) -> List[str]:
        """Expand query using synonym dictionary."""
        words = re.findall(r'\b\w+\b', query.lower())
        expanded_queries = []
        
        for word in words:
            if word in self.expansion_dict and len(expanded_queries) < max_expansions:
                synonyms = self.expansion_dict[word]
                for synonym in list(synonyms)[:2]:  # Take max 2 synonyms per word
                    expanded_query = re.sub(
                        r'\b' + re.escape(word) + r'\b', 
                        synonym, 
                        query, 
                        flags=re.IGNORECASE
                    )
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def _expand_abbreviations(self, query: str) -> List[str]:
        """Expand abbreviations in the query."""
        expanded_queries = []
        
        for abbrev, full_form in self.abbreviations.items():
            if abbrev.lower() in query.lower():
                expanded_query = re.sub(
                    r'\b' + re.escape(abbrev) + r'\b', 
                    full_form, 
                    query, 
                    flags=re.IGNORECASE
                )
                if expanded_query != query:
                    expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def _expand_with_context(self, query: str, max_expansions: int) -> List[str]:
        """Expand query with contextual terms."""
        expanded_queries = []
        
        # Detect query patterns and add contextual terms
        query_lower = query.lower()
        
        # If asking about models, add related model terms
        if any(term in query_lower for term in ['model', 'algorithm', 'architecture']):
            context_terms = ['performance', 'training', 'evaluation', 'accuracy']
            for term in context_terms[:max_expansions]:
                expanded_queries.append(f"{query} {term}")
        
        # If asking about implementation, add technical terms
        elif any(term in query_lower for term in ['how to', 'implement', 'build', 'create']):
            context_terms = ['example', 'tutorial', 'guide', 'steps']
            for term in context_terms[:max_expansions]:
                expanded_queries.append(f"{query} {term}")
        
        # If asking about problems/issues, add solution terms
        elif any(term in query_lower for term in ['problem', 'issue', 'error', 'fix']):
            context_terms = ['solution', 'resolve', 'debug', 'troubleshoot']
            for term in context_terms[:max_expansions]:
                expanded_queries.append(f"{query} {term}")
        
        return expanded_queries[:max_expansions]
    
    def get_query_intent(self, query: str) -> str:
        """
        Identify the intent of a query for strategy selection.
        
        Args:
            query: Query string
            
        Returns:
            Intent category (conceptual, technical, how-to, problem-solving)
        """
        query_lower = query.lower()
        
        # Technical queries
        if any(term in query_lower for term in ['api', 'code', 'function', 'class', 'implement']):
            return 'technical'
        
        # How-to queries
        elif any(term in query_lower for term in ['how to', 'how do', 'steps', 'tutorial']):
            return 'how-to'
        
        # Problem-solving queries
        elif any(term in query_lower for term in ['error', 'problem', 'issue', 'fix', 'debug']):
            return 'problem-solving'
        
        # Conceptual queries
        elif any(term in query_lower for term in ['what is', 'explain', 'concept', 'theory']):
            return 'conceptual'
        
        # Default
        return 'general'
    
    def preprocess_query(self, query: str) -> str:
        """
        Clean and preprocess query for better search results.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        if not query:
            return query
        
        # Remove extra whitespace
        cleaned = ' '.join(query.split())
        
        # Remove common stop words that don't add semantic value
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = cleaned.split()
        
        # Keep stop words if query is very short
        if len(words) > 3:
            filtered_words = [word for word in words if word.lower() not in stop_words]
            if filtered_words:  # Don't return empty query
                cleaned = ' '.join(filtered_words)
        
        return cleaned.strip()
    
    def should_use_expansion(self, query: str) -> bool:
        """
        Determine if query expansion should be used.
        
        Args:
            query: Query string
            
        Returns:
            True if expansion should be used
        """
        # Don't expand very long queries
        if len(query.split()) > 10:
            return False
        
        # Don't expand quoted strings (exact matches)
        if query.startswith('"') and query.endswith('"'):
            return False
        
        # Don't expand if it contains technical symbols (code queries)
        if any(symbol in query for symbol in ['()', '{}', '[]', '==', '!=', '->']):
            return False
        
        return True 