"""
HyDE (Hypothetical Document Embeddings) Retriever Implementation

HyDE improves retrieval by:
1. Generating a hypothetical answer to the query
2. Embedding the hypothetical answer
3. Using it to retrieve relevant documents

This often works better than embedding the query directly because:
- Answers are semantically closer to document content
- Reduces the semantic gap between query and documents
"""

from typing import List, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


class HyDERetriever(BaseRetriever):
    """
    HyDE Retriever that generates hypothetical documents for better retrieval.
    """
    
    llm: OllamaLLM
    embeddings: HuggingFaceEmbeddings
    vectorstore: WeaviateVectorStore
    k: int = 5  # Number of documents to retrieve
    
    # Prompt for generating hypothetical document
    hyde_prompt_template: str = """Please write a passage to answer the question. 
Question: {question}
Passage:"""
    
    def __init__(
        self,
        llm: OllamaLLM,
        embeddings: HuggingFaceEmbeddings,
        vectorstore: WeaviateVectorStore,
        k: int = 5
    ):
        """
        Initialize HyDE retriever.
        
        Args:
            llm: Language model for generating hypothetical documents
            embeddings: Embedding model
            vectorstore: Vector store containing documents
            k: Number of documents to retrieve
        """
        super().__init__()
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.k = k
        
    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document/answer for the query.
        
        Args:
            query: User's question
            
        Returns:
            Hypothetical document text
        """
        prompt = PromptTemplate(
            template=self.hyde_prompt_template,
            input_variables=["question"]
        )
        
        # Generate hypothetical answer
        formatted_prompt = prompt.format(question=query)
        hypothetical_doc = self.llm.invoke(formatted_prompt)
        
        return hypothetical_doc.strip()
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using HyDE approach.
        
        Args:
            query: User's question
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Step 1: Generate hypothetical document
        print(f"  [HyDE] Generating hypothetical document for: {query[:50]}...")
        hypothetical_doc = self._generate_hypothetical_document(query)
        print(f"  [HyDE] Generated: {hypothetical_doc[:100]}...")
        
        # Step 2: Embed the hypothetical document
        hypo_embedding = self.embeddings.embed_query(hypothetical_doc)
        
        # Step 3: Retrieve documents similar to the hypothetical document
        print(f"  [HyDE] Retrieving {self.k} documents...")
        # Use the vectorstore's similarity search with the hypothetical embedding
        docs = self.vectorstore.similarity_search_by_vector(
            hypo_embedding,
            k=self.k
        )
        
        print(f"  [HyDE] Retrieved {len(docs)} documents")
        return docs


class MultiQueryRetriever(BaseRetriever):
    """
    Multi-Query Retriever that generates multiple perspectives of a question.
    Can be combined with HyDE for even better results.
    """
    
    llm: OllamaLLM
    base_retriever: Any  # Can be standard retriever or HyDE retriever
    num_queries: int = 3
    
    multi_query_prompt: str = """You are an AI language model assistant. Your task is to generate {num_queries} 
different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some 
of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.

Original question: {question}
Alternative questions:"""
    
    def __init__(
        self,
        llm: OllamaLLM,
        base_retriever: Any,
        num_queries: int = 3
    ):
        """
        Initialize Multi-Query retriever.
        
        Args:
            llm: Language model for generating queries
            base_retriever: Base retriever (can be standard or HyDE)
            num_queries: Number of alternative queries to generate
        """
        super().__init__()
        self.llm = llm
        self.base_retriever = base_retriever
        self.num_queries = num_queries
        
    def _generate_queries(self, query: str) -> List[str]:
        """Generate multiple versions of the query."""
        prompt = self.multi_query_prompt.format(
            num_queries=self.num_queries,
            question=query
        )
        
        response = self.llm.invoke(prompt)
        
        # Parse the response to get individual queries
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        
        # Include original query
        all_queries = [query] + queries[:self.num_queries]
        
        return all_queries
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using multiple query perspectives.
        
        Args:
            query: User's question
            run_manager: Callback manager
            
        Returns:
            List of relevant documents (deduplicated)
        """
        # Generate multiple queries
        print(f"  [MultiQuery] Generating {self.num_queries} query variations...")
        queries = self._generate_queries(query)
        print(f"  [MultiQuery] Generated queries: {queries}")
        
        # Retrieve documents for each query
        all_docs = []
        seen_content = set()
        
        for q in queries:
            docs = self.base_retriever.invoke(q)
            for doc in docs:
                # Deduplicate based on content
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        print(f"  [MultiQuery] Retrieved {len(all_docs)} unique documents")
        return all_docs


def create_hyde_retriever(
    llm: OllamaLLM,
    embeddings: HuggingFaceEmbeddings,
    vectorstore: WeaviateVectorStore,
    k: int = 5
) -> HyDERetriever:
    """
    Factory function to create a HyDE retriever.
    
    Args:
        llm: Language model
        embeddings: Embedding model
        vectorstore: Vector store
        k: Number of documents to retrieve
        
    Returns:
        HyDERetriever instance
    """
    return HyDERetriever(
        llm=llm,
        embeddings=embeddings,
        vectorstore=vectorstore,
        k=k
    )


def create_hybrid_retriever(
    llm: OllamaLLM,
    embeddings: HuggingFaceEmbeddings,
    vectorstore: WeaviateVectorStore,
    use_hyde: bool = True,
    use_multi_query: bool = False,
    k: int = 5,
    num_queries: int = 3
) -> BaseRetriever:
    """
    Create a hybrid retriever with optional HyDE and Multi-Query.
    
    Args:
        llm: Language model
        embeddings: Embedding model
        vectorstore: Vector store
        use_hyde: Whether to use HyDE
        use_multi_query: Whether to use multi-query
        k: Number of documents to retrieve
        num_queries: Number of query variations (if using multi-query)
        
    Returns:
        Configured retriever
    """
    # Start with base retriever
    if use_hyde:
        base_retriever = HyDERetriever(
            llm=llm,
            embeddings=embeddings,
            vectorstore=vectorstore,
            k=k
        )
        print("✓ HyDE retriever created")
    else:
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        print("✓ Standard retriever created")
    
    # Optionally wrap with multi-query
    if use_multi_query:
        retriever = MultiQueryRetriever(
            llm=llm,
            base_retriever=base_retriever,
            num_queries=num_queries
        )
        print("✓ Multi-Query wrapper added")
    else:
        retriever = base_retriever
    
    return retriever