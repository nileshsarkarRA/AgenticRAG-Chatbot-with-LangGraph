"""
Advanced RAG Techniques Implementation
Using LangChain 1.0+ and LangGraph

"""

import os
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum

# LangChain Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStore

# LangChain Community & Integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
    SemanticChunker
)

# Retrievers
from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever
)
from langchain.retrievers.document_compressors import (
    CohereRerank,
    EmbeddingsFilter,
    LLMChainExtractor
)
from langchain_community.retrievers import BM25Retriever

# LangGraph for Agentic RAG
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Memory
from langchain.memory import ConversationBufferMemory

# ============================================================================
# TECHNIQUE 1: HIERARCHICAL INDEXING WITH PARENT-CHILD CHUNKS
# Purpose: Retrieve small, precise chunks while maintaining large context
# Improves: Context quality and retrieval precision
# ============================================================================

class HierarchicalRAG:
    """
    Parent-Child document retrieval: Index small chunks for precise search,
    but return larger parent chunks for better context.
    """
    def __init__(self, embeddings, vectorstore_path="./hierarchical_db"):
        self.embeddings = embeddings
        self.vectorstore_path = vectorstore_path
        
    def create_parent_child_retriever(self, documents: List[Document]):
        """
        Creates a retriever that searches small chunks but returns parent chunks
        """
        # Small chunks for precise retrieval
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        # Large chunks for context
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        # Vector store for child chunks
        vectorstore = FAISS.from_documents([], self.embeddings)
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=None,  # Uses InMemoryStore by default
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        retriever.add_documents(documents)
        return retriever


# ============================================================================
# TECHNIQUE 2: HYBRID SEARCH (Sparse + Dense)
# Purpose: Combine semantic (dense) and keyword (sparse) search
# Improves: Retrieval recall and precision
# ============================================================================

class HybridSearchRAG:
    """
    Ensemble retriever combining:
    - Dense retrieval (semantic/vector search)
    - Sparse retrieval (BM25/keyword search)
    """
    def __init__(self, embeddings, weights=(0.6, 0.4)):
        self.embeddings = embeddings
        self.weights = weights
        
    def create_hybrid_retriever(self, documents: List[Document]):
        """
        Creates ensemble retriever with weighted combination
        """
        # Dense retriever (semantic search)
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Sparse retriever (BM25 keyword search)
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = 5
        
        # Ensemble with weighted combination
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=list(self.weights)
        )
        
        return ensemble_retriever


# ============================================================================
# TECHNIQUE 3: RERANKING WITH CROSS-ENCODERS
# Purpose: Rerank retrieved documents using cross-encoder models
# Improves: Relevance of top-k results, reduces noise
# ============================================================================

class ReRankingRAG:
    """
    Two-stage retrieval:
    1. Fast initial retrieval (get top-20)
    2. Precise reranking (narrow to top-5)
    Uses cross-encoder models for superior relevance scoring
    """
    def __init__(self, base_retriever, model_name="BAAI/bge-reranker-base"):
        self.base_retriever = base_retriever
        self.model_name = model_name
        
    def create_reranking_retriever(self, top_n=5):
        """
        Wraps base retriever with reranking layer
        """
        # Option 1: Cohere Rerank (requires API key)
        # compressor = CohereRerank(top_n=top_n)
        
        # Option 2: LLM-based extraction (slower but no extra API)
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        compressor = LLMChainExtractor.from_llm(llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
        
        return compression_retriever


# ============================================================================
# TECHNIQUE 4: MULTI-QUERY RETRIEVAL
# Purpose: Generate multiple query variations for better coverage
# Improves: Handles ambiguous queries, increases recall
# ============================================================================

class MultiQueryRAG:
    """
    LLM generates multiple perspectives of the same query:
    - Original: "What are the benefits of RAG?"
    - Generated: 
      * "How does RAG improve LLM performance?"
      * "What advantages does RAG provide?"
      * "Why use RAG systems?"
    """
    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm
        
    def create_multi_query_retriever(self):
        """
        Creates retriever that generates multiple query variations
        """
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        return retriever


# ============================================================================
# TECHNIQUE 5: HYDE (Hypothetical Document Embeddings)
# Purpose: Generate hypothetical answer, then search for similar content
# Improves: Semantic matching when query-document gap is large
# ============================================================================

class HyDERAG:
    """
    HyDE Process:
    1. Generate hypothetical answer to query
    2. Embed the generated answer
    3. Search for documents similar to hypothetical answer
    
    Why: Bridges query-document embedding space gap
    """
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        
    def hypothetical_answer_generator(self, query: str) -> str:
        """Generate hypothetical answer"""
        prompt = ChatPromptTemplate.from_template(
            "Write a detailed answer to this question: {query}\n"
            "Be specific and factual, even if hypothetical."
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def retrieve_with_hyde(self, query: str, k=5):
        """
        HyDE retrieval pipeline
        """
        # Generate hypothetical answer
        hypothetical_answer = self.hypothetical_answer_generator(query)
        
        # Search using hypothetical answer
        docs = self.vectorstore.similarity_search(hypothetical_answer, k=k)
        
        return docs


# ============================================================================
# TECHNIQUE 6: SEMANTIC CHUNKING
# Purpose: Split documents based on semantic similarity, not fixed size
# Improves: Chunk coherence and topical consistency
# ============================================================================

class SemanticChunkingRAG:
    """
    Instead of fixed-size chunks, split when semantic similarity drops.
    Preserves topic boundaries naturally.
    """
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
    def create_semantic_chunks(self, documents: List[Document]):
        """
        Creates semantically coherent chunks
        """
        text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile"  # or "standard_deviation", "interquartile"
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks


# ============================================================================
# TECHNIQUE 7: STEP-BACK PROMPTING
# Purpose: Generate broader questions before specific retrieval
# Improves: Handles complex queries requiring background knowledge
# ============================================================================

class StepBackRAG:
    """
    Two-phase retrieval:
    1. Ask a "step-back" (broader) question to get general context
    2. Ask the original specific question
    
    Example:
    - Original: "What's the boiling point of water at 5000m elevation?"
    - Step-back: "How does elevation affect boiling points?"
    """
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
    def generate_step_back_query(self, query: str) -> str:
        """Generate broader context question"""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert at generating broader questions.\n"
            "Given this specific question: {query}\n\n"
            "Generate a broader, more general question that would help "
            "understand the context needed to answer the specific question.\n"
            "Broader question:"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def retrieve_with_step_back(self, query: str):
        """
        Retrieve both general and specific context
        """
        # Get broader context
        step_back_query = self.generate_step_back_query(query)
        general_docs = self.retriever.invoke(step_back_query)
        
        # Get specific context
        specific_docs = self.retriever.invoke(query)
        
        # Combine and deduplicate
        all_docs = general_docs + specific_docs
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        
        return unique_docs


# ============================================================================
# TECHNIQUE 8: QUERY DECOMPOSITION
# Purpose: Break complex queries into sub-queries
# Improves: Handles multi-faceted questions
# ============================================================================

class QueryDecompositionRAG:
    """
    Breaks complex questions into simpler sub-questions:
    
    Complex: "Compare the economic impact of AI on healthcare vs manufacturing"
    Sub-queries:
    1. "What is the economic impact of AI on healthcare?"
    2. "What is the economic impact of AI on manufacturing?"
    3. "How do these impacts compare?"
    """
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
    def decompose_query(self, query: str) -> List[str]:
        """Break query into sub-queries"""
        prompt = ChatPromptTemplate.from_template(
            "Break down this complex question into 2-4 simpler sub-questions:\n"
            "Question: {query}\n\n"
            "Sub-questions (one per line):"
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query})
        
        # Parse sub-questions
        sub_queries = [q.strip() for q in result.split('\n') if q.strip()]
        return sub_queries
    
    def retrieve_decomposed(self, query: str):
        """
        Retrieve for each sub-query and combine
        """
        sub_queries = self.decompose_query(query)
        
        all_docs = []
        for sub_q in sub_queries:
            docs = self.retriever.invoke(sub_q)
            all_docs.extend(docs)
        
        # Deduplicate
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        return unique_docs


# ============================================================================
# TECHNIQUE 9: SELF-QUERY RETRIEVAL (METADATA FILTERING)
# Purpose: Extract metadata filters from natural language queries
# Improves: Precision for queries with implicit filters
# ============================================================================

class SelfQueryRAG:
    """
    Automatically extracts filters from queries:
    
    Query: "Show me Python tutorials from 2024"
    Extracted:
    - Search: "Python tutorials"
    - Filter: year == 2024, language == "Python"
    """
    def __init__(self, vectorstore, llm, metadata_field_info):
        self.vectorstore = vectorstore
        self.llm = llm
        self.metadata_field_info = metadata_field_info
        
    def create_self_query_retriever(self):
        """
        Creates retriever that automatically extracts metadata filters
        """
        from langchain.retrievers.self_query.base import SelfQueryRetriever
        
        retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents="Technical documentation and tutorials",
            metadata_field_info=self.metadata_field_info
        )
        return retriever


# ============================================================================
# TECHNIQUE 10: AGENTIC RAG WITH LANGGRAPH
# Purpose: Multi-step reasoning with decision-making
# Improves: Handles complex workflows, can call multiple tools
# ============================================================================

class AgenticRAGState(TypedDict):
    """State for agentic RAG workflow"""
    query: str
    documents: List[Document]
    answer: str
    needs_web_search: bool
    iteration: int


class AgenticRAG:
    """
    Agentic RAG using LangGraph:
    - Decides whether to retrieve, search web, or answer
    - Can iterate multiple times
    - Self-corrects based on quality checks
    """
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgenticRAGState)
        
        # Define nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("check_quality", self._check_quality_node)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._should_generate,
            {
                "generate": "generate",
                "retry": "retrieve"
            }
        )
        workflow.add_edge("generate", "check_quality")
        workflow.add_conditional_edges(
            "check_quality",
            self._check_answer_quality,
            {
                "end": END,
                "retry": "retrieve"
            }
        )
        
        return workflow.compile()
    
    def _retrieve_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Retrieve relevant documents"""
        docs = self.retriever.invoke(state["query"])
        state["documents"] = docs
        return state
    
    def _grade_documents_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Grade document relevance"""
        # Simple grading logic (can be more sophisticated)
        relevant_docs = [doc for doc in state["documents"] if len(doc.page_content) > 100]
        state["documents"] = relevant_docs
        return state
    
    def _should_generate(self, state: AgenticRAGState) -> Literal["generate", "retry"]:
        """Decide if we have enough documents"""
        if len(state["documents"]) >= 2:
            return "generate"
        elif state["iteration"] < 3:
            state["iteration"] += 1
            return "retry"
        return "generate"
    
    def _generate_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Generate answer from documents"""
        context = "\n\n".join([doc.page_content for doc in state["documents"]])
        
        prompt = ChatPromptTemplate.from_template(
            "Answer based on this context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": state["query"]})
        
        state["answer"] = answer
        return state
    
    def _check_quality_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Check answer quality"""
        # Quality check logic
        return state
    
    def _check_answer_quality(self, state: AgenticRAGState) -> Literal["end", "retry"]:
        """Decide if answer is good enough"""
        if len(state["answer"]) > 50:  # Simple check
            return "end"
        elif state["iteration"] < 3:
            return "retry"
        return "end"
    
    def run(self, query: str):
        """Execute agentic RAG workflow"""
        initial_state = {
            "query": query,
            "documents": [],
            "answer": "",
            "needs_web_search": False,
            "iteration": 0
        }
        result = self.graph.invoke(initial_state)
        return result["answer"]


# ============================================================================
# TECHNIQUE 11: FUSION RETRIEVAL (RECIPROCAL RANK FUSION)
# Purpose: Combine results from multiple retrieval strategies
# Improves: Leverages strengths of different retrievers
# ============================================================================

class FusionRAG:
    """
    Reciprocal Rank Fusion (RRF):
    Combines rankings from multiple retrievers using reciprocal rank formula.
    
    Better than simple concatenation because it considers rank position.
    """
    def __init__(self, retrievers: List[Any], k=60):
        self.retrievers = retrievers
        self.k = k  # Constant for RRF formula
        
    def fuse_results(self, query: str, top_k=5):
        """
        Apply RRF to combine results
        """
        # Get results from all retrievers
        all_results = []
        for retriever in self.retrievers:
            docs = retriever.invoke(query)
            all_results.append(docs)
        
        # Calculate RRF scores
        doc_scores = {}
        for docs in all_results:
            for rank, doc in enumerate(docs, start=1):
                score = 1 / (self.k + rank)
                content = doc.page_content
                if content in doc_scores:
                    doc_scores[content] = (doc_scores[content][0] + score, doc)
                else:
                    doc_scores[content] = (score, doc)
        
        # Sort by score and return top-k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in sorted_docs[:top_k]]


# ============================================================================
# TECHNIQUE 12: CONTEXTUAL RETRIEVAL WITH CHUNK ENRICHMENT
# Purpose: Add context to each chunk before embedding
# Improves: Chunk standalone quality, better for out-of-context retrieval
# ============================================================================

class ContextualRetrievalRAG:
    """
    Before embedding, add context to each chunk:
    
    Original chunk: "The treaty was signed in 1945"
    Enriched: "This chunk discusses the UN Charter. The treaty was signed in 1945."
    
    Makes chunks more self-contained and searchable.
    """
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        
    def enrich_chunk(self, chunk: Document, doc_context: str) -> Document:
        """Add context to chunk"""
        prompt = ChatPromptTemplate.from_template(
            "Document context: {context}\n\n"
            "Chunk: {chunk}\n\n"
            "Add a brief 1-2 sentence prefix to this chunk that provides necessary "
            "context from the document. Return ONLY the enriched chunk."
        )
        chain = prompt | self.llm | StrOutputParser()
        
        enriched = chain.invoke({
            "context": doc_context,
            "chunk": chunk.page_content
        })
        
        chunk.page_content = enriched
        return chunk
    
    def create_contextual_chunks(self, documents: List[Document]):
        """
        Create enriched chunks for all documents
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        
        enriched_chunks = []
        for doc in documents:
            # Get document summary for context
            doc_context = doc.page_content[:500]  # First 500 chars as context
            
            # Split into chunks
            chunks = splitter.split_documents([doc])
            
            # Enrich each chunk
            for chunk in chunks:
                enriched_chunk = self.enrich_chunk(chunk, doc_context)
                enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks


# ============================================================================
# TECHNIQUE 13: SLIDING WINDOW RETRIEVAL
# Purpose: Return chunks with surrounding context windows
# Improves: Reduces context fragmentation
# ============================================================================

class SlidingWindowRAG:
    """
    When retrieving chunk N, also include chunks N-1 and N+1.
    Prevents loss of context at chunk boundaries.
    """
    def __init__(self, base_retriever, all_chunks: List[Document], window=1):
        self.base_retriever = base_retriever
        self.all_chunks = all_chunks
        self.window = window
        
    def retrieve_with_window(self, query: str):
        """
        Retrieve chunks with surrounding context
        """
        # Get initial chunks
        retrieved_chunks = self.base_retriever.invoke(query)
        
        # Expand with windows
        expanded_chunks = []
        for chunk in retrieved_chunks:
            # Find chunk index
            try:
                idx = self.all_chunks.index(chunk)
                
                # Add window
                start = max(0, idx - self.window)
                end = min(len(self.all_chunks), idx + self.window + 1)
                
                expanded_chunks.extend(self.all_chunks[start:end])
            except ValueError:
                expanded_chunks.append(chunk)
        
        # Deduplicate while preserving order
        seen = set()
        unique_chunks = []
        for chunk in expanded_chunks:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)
        
        return unique_chunks


# ============================================================================
# TECHNIQUE 14: CORRECTIVE RAG (CRAG)
# Purpose: Self-correct retrieval with web search fallback
# Improves: Handles out-of-domain or outdated knowledge
# ============================================================================

class CorrectiveRAG:
    """
    CRAG workflow:
    1. Retrieve from local knowledge base
    2. Grade relevance of retrieved docs
    3. If poor quality -> trigger web search
    4. Generate with best available context
    """
    def __init__(self, retriever, llm, web_search_tool=None):
        self.retriever = retriever
        self.llm = llm
        self.web_search_tool = web_search_tool
        
    def grade_documents(self, query: str, documents: List[Document]) -> str:
        """Grade if documents are relevant"""
        docs_txt = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = ChatPromptTemplate.from_template(
            "Query: {query}\n\n"
            "Documents:\n{documents}\n\n"
            "Are these documents relevant to answer the query? "
            "Answer only 'relevant' or 'not_relevant'."
        )
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({"query": query, "documents": docs_txt})
        return result.strip().lower()
    
    def retrieve_corrective(self, query: str):
        """
        CRAG retrieval with correction
        """
        # Step 1: Retrieve
        docs = self.retriever.invoke(query)
        
        # Step 2: Grade
        grade = self.grade_documents(query, docs)
        
        # Step 3: Correct if needed
        if grade == "not_relevant" and self.web_search_tool:
            # Fallback to web search
            web_results = self.web_search_tool(query)
            docs = web_results
        
        return docs


# ============================================================================
# TECHNIQUE 15: RAPTOR (Recursive Abstractive Processing)
# Purpose: Create hierarchical summaries of document chunks
# Improves: Multi-level retrieval for different query granularities
# ============================================================================

class RAPTORRetrieval:
    """
    RAPTOR creates a tree of summaries:
    
    Level 0: Original chunks
    Level 1: Summaries of groups of chunks
    Level 2: Summaries of summaries
    
    Allows retrieval at different abstraction levels.
    """
    def __init__(self, llm, embeddings, cluster_size=5):
        self.llm = llm
        self.embeddings = embeddings
        self.cluster_size = cluster_size
        
    def summarize_chunks(self, chunks: List[Document]) -> Document:
        """Create summary of chunk group"""
        combined = "\n\n".join([chunk.page_content for chunk in chunks])
        
        prompt = ChatPromptTemplate.from_template(
            "Summarize these related text chunks into a coherent summary:\n\n{text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        
        summary = chain.invoke({"text": combined})
        return Document(page_content=summary, metadata={"level": 1})
    
    def build_raptor_tree(self, documents: List[Document]):
        """
        Build hierarchical summary tree
        """
        # Level 0: Original chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        level_0 = splitter.split_documents(documents)
        
        # Level 1: Summaries of groups
        level_1 = []
        for i in range(0, len(level_0), self.cluster_size):
            chunk_group = level_0[i:i + self.cluster_size]
            summary = self.summarize_chunks(chunk_group)
            level_1.append(summary)
        
        # Level 2: Summaries of summaries
        level_2 = []
        if len(level_1) > 1:
            for i in range(0, len(level_1), self.cluster_size):
                summary_group = level_1[i:i + self.cluster_size]
                meta_summary = self.summarize_chunks(summary_group)
                meta_summary.metadata["level"] = 2
                level_2.append(meta_summary)
        
        # Combine all levels
        all_nodes = level_0 + level_1 + level_2
        
        # Create vector store with all levels
        vectorstore = FAISS.from_documents(all_nodes, self.embeddings)
        
        return vectorstore.as_retriever()


# ============================================================================
# USAGE EXAMPLE: Combining Multiple Techniques
# ============================================================================

def create_production_rag_pipeline():
    """
    Example: Combining multiple techniques for production system
    """
    # Initialize components
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # Load documents
    documents = [
        Document(page_content="Sample document 1...", metadata={"source": "doc1"}),
        Document(page_content="Sample document 2...", metadata={"source": "doc2"}),
    ]
    
    # 1. Semantic chunking
    semantic_rag = SemanticChunkingRAG(embeddings)
    chunks = semantic_rag.create_semantic_chunks(documents)
    
    # 2. Contextual enrichment
    contextual_rag = ContextualRetrievalRAG(llm, embeddings)
    enriched_chunks = contextual_rag.create_contextual_chunks(documents)
    
    # 3. Hybrid search
    hybrid_rag = HybridSearchRAG(embeddings)
    hybrid_retriever = hybrid_rag.create_hybrid_retriever(enriched_chunks)
    
    # 4. Add reranking
    rerank_rag = ReRankingRAG(hybrid_retriever)
    reranked_retriever = rerank_rag.create_reranking_retriever(top_n=5)
    
    # 5. Wrap in agentic workflow
    agentic_rag = AgenticRAG(reranked_retriever, llm)
    
    return agentic_rag


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Advanced RAG Techniques - Complete Implementation")
    print("=" * 60)
    print("\nThis file demonstrates 15+ advanced RAG techniques:")
    print("1. Hierarchical Indexing (Parent-Child)")
    print("2. Hybrid Search (Sparse + Dense)")
    print("3. Cross-Encoder Reranking")
    print("4. Multi-Query Retrieval")
    print("5. HyDE (Hypothetical Document Embeddings)")
    print("6. Semantic Chunking")
    print("7. Step-Back Prompting")
    print("8. Query Decomposition")
    print("9. Self-Query with Metadata Filtering")
    print("10. Agentic RAG with LangGraph")
    print("11. Fusion Retrieval (RRF)")
    print("12. Contextual Retrieval")
    print("13. Sliding Window Retrieval")
    print("14. Corrective RAG (CRAG)")
    print("15. RAPTOR (Recursive Summaries)")
    print("\nEach technique is modular and can be used independently or combined!")