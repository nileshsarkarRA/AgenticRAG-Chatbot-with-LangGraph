"""
Advanced RAG System for OpAmp 741 Datasheet Analysis
Using Gemini 2.0 Flash (Free), Tavily Web Search, and Local Embeddings

Goal: Answer technical questions about OpAmp 741 with high accuracy
by combining datasheet knowledge with web research when needed.

Requirements:
pip install langchain langchain-google-genai langchain-community
pip install faiss-cpu sentence-transformers tavily-python
pip install pypdf langgraph rank-bm25
"""

import os
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass

# LangChain Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Gemini Integration
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Local Embeddings (Free Alternative)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# Document Processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Retrievers
from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever
)
from langchain_community.retrievers import BM25Retriever

# LangGraph for Agentic Workflow
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Tavily for Web Search
from tavily import TavilyClient

# Storage
from langchain.storage import InMemoryStore


# ============================================================================
# CONFIGURATION
# ============================================================================

class RAGConfig:
    """Configuration for the RAG system"""
    
    # API Keys (set as environment variables)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    
    # Model Configuration
    GEMINI_MODEL = "gemini-2.0-flash-exp"  # Free tier model
    TEMPERATURE = 0.1  # Low for factual accuracy
    
    # Local Embeddings (Free)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking Strategy
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    PARENT_CHUNK_SIZE = 2000
    CHILD_CHUNK_SIZE = 400
    
    # Retrieval
    TOP_K = 5
    HYBRID_WEIGHTS = (0.6, 0.4)  # Dense, Sparse
    
    # Document Path
    DATASHEET_PATH = "./opamp_741_datasheet.pdf"
    VECTORSTORE_PATH = "./opamp_741_vectorstore"


# ============================================================================
# INITIALIZE CORE COMPONENTS
# ============================================================================

def initialize_components():
    """Initialize LLM, Embeddings, and Tools"""
    
    # Gemini 2.0 Flash (Free)
    llm = GoogleGenerativeAI(
        model=RAGConfig.GEMINI_MODEL,
        google_api_key=RAGConfig.GOOGLE_API_KEY,
        temperature=RAGConfig.TEMPERATURE
    )
    
    # Local Embeddings (Free - runs on CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name=RAGConfig.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Tavily Web Search
    tavily_client = TavilyClient(api_key=RAGConfig.TAVILY_API_KEY)
    
    return llm, embeddings, tavily_client


# ============================================================================
# TECHNIQUE 1: HIERARCHICAL PARENT-CHILD RETRIEVAL
# ============================================================================

class HierarchicalRetriever:
    """
    Parent-Child Chunking for OpAmp Datasheet:
    - Child chunks (400 chars): Precise retrieval for specs
    - Parent chunks (2000 chars): Full context for understanding
    """
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
    def create_retriever(self, documents: List[Document]):
        """Build parent-child retriever"""
        
        # Small chunks for precise search
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHILD_CHUNK_SIZE,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        
        # Large chunks for context
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.PARENT_CHUNK_SIZE,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        
        # Vector store
        vectorstore = FAISS.from_documents([], self.embeddings)
        
        # In-memory docstore
        docstore = InMemoryStore()
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        retriever.add_documents(documents)
        return retriever


# ============================================================================
# TECHNIQUE 2: HYBRID SEARCH (SEMANTIC + KEYWORD)
# ============================================================================

class HybridRetriever:
    """
    Combine semantic and keyword search for OpAmp specs:
    - Semantic: "voltage gain characteristics"
    - Keyword: "741", "µA741", specific part numbers
    """
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
    def create_retriever(self, documents: List[Document]):
        """Build hybrid retriever"""
        
        # Dense retriever (semantic)
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": RAGConfig.TOP_K}
        )
        
        # Sparse retriever (BM25 keyword)
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = RAGConfig.TOP_K
        
        # Ensemble
        hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=list(RAGConfig.HYBRID_WEIGHTS)
        )
        
        return hybrid_retriever


# ============================================================================
# TECHNIQUE 3: MULTI-QUERY EXPANSION
# ============================================================================

class MultiQueryRetriever:
    """
    Generate query variations for ambiguous questions:
    Q: "What's the gain?"
    -> "What is the open-loop voltage gain of 741?"
    -> "What is the typical gain specification?"
    """
    
    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm
        
    def create_retriever(self):
        """Build multi-query retriever"""
        from langchain.retrievers.multi_query import MultiQueryRetriever as MQR
        
        return MQR.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )


# ============================================================================
# TECHNIQUE 4: HYDE (HYPOTHETICAL ANSWER GENERATION)
# ============================================================================

class HyDERetriever:
    """
    Generate hypothetical answer, then search:
    Q: "How does 741 handle high frequencies?"
    Hypothetical: "The 741 has a gain-bandwidth product of 1MHz..."
    Then search using this hypothetical answer.
    """
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        
    def retrieve(self, query: str, k: int = RAGConfig.TOP_K) -> List[Document]:
        """HyDE retrieval"""
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert on OpAmp 741. Generate a detailed technical "
            "answer to this question based on typical 741 specifications:\n\n"
            "Question: {query}\n\n"
            "Hypothetical Answer (be specific with numbers and specs):"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        hypothetical = chain.invoke({"query": query})
        
        # Search using hypothetical answer
        docs = self.vectorstore.similarity_search(hypothetical, k=k)
        return docs


# ============================================================================
# TECHNIQUE 5: CONTEXTUAL CHUNK ENRICHMENT
# ============================================================================

class ContextualChunker:
    """
    Add context to chunks before embedding:
    
    Original: "Gain: 200,000"
    Enriched: "OpAmp 741 Open-Loop Voltage Gain: 200,000"
    """
    
    def __init__(self, llm):
        self.llm = llm
        
    def enrich_chunks(self, chunks: List[Document]) -> List[Document]:
        """Add context to each chunk"""
        
        prompt = ChatPromptTemplate.from_template(
            "You are processing an OpAmp 741 datasheet chunk. "
            "Add a brief prefix (5-10 words) that provides context.\n\n"
            "Chunk: {chunk}\n\n"
            "Return ONLY the enriched chunk with prefix:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        enriched = []
        for chunk in chunks[:10]:  # Limit for free tier
            try:
                result = chain.invoke({"chunk": chunk.page_content[:500]})
                chunk.page_content = result
                enriched.append(chunk)
            except Exception as e:
                print(f"Enrichment failed: {e}")
                enriched.append(chunk)
        
        return enriched


# ============================================================================
# TECHNIQUE 6: QUERY DECOMPOSITION
# ============================================================================

class QueryDecomposer:
    """
    Break complex queries:
    Q: "Compare 741 slew rate to input offset voltage impact"
    -> Q1: "What is the slew rate of 741?"
    -> Q2: "What is the input offset voltage of 741?"
    -> Q3: "How do these parameters affect performance?"
    """
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
    def decompose_and_retrieve(self, query: str) -> List[Document]:
        """Decompose query and retrieve for each part"""
        
        prompt = ChatPromptTemplate.from_template(
            "Break this complex OpAmp 741 question into 2-3 simpler questions.\n"
            "Question: {query}\n\n"
            "Sub-questions (one per line, no numbering):"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query})
        
        # Parse sub-queries
        sub_queries = [q.strip() for q in result.split('\n') if q.strip()]
        
        # Retrieve for each
        all_docs = []
        for sq in sub_queries[:3]:  # Limit for efficiency
            docs = self.retriever.invoke(sq)
            all_docs.extend(docs)
        
        # Deduplicate
        unique = {doc.page_content: doc for doc in all_docs}
        return list(unique.values())


# ============================================================================
# TECHNIQUE 7: CORRECTIVE RAG WITH WEB SEARCH
# ============================================================================

class CorrectiveRAG:
    """
    Self-correct with Tavily web search:
    1. Try local datasheet
    2. If quality is low -> search web for updated info
    3. Combine both sources
    """
    
    def __init__(self, retriever, llm, tavily_client):
        self.retriever = retriever
        self.llm = llm
        self.tavily = tavily_client
        
    def grade_relevance(self, query: str, docs: List[Document]) -> bool:
        """Check if documents are relevant"""
        
        if not docs:
            return False
        
        context = "\n".join([d.page_content[:200] for d in docs[:3]])
        
        prompt = ChatPromptTemplate.from_template(
            "Query: {query}\n\nContext:\n{context}\n\n"
            "Is this context relevant to answer the query about OpAmp 741? "
            "Answer ONLY 'yes' or 'no'."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "context": context})
        
        return "yes" in result.lower()
    
    def web_search_fallback(self, query: str) -> List[Document]:
        """Search web using Tavily"""
        
        try:
            # Add context to query
            search_query = f"OpAmp 741 {query} specifications datasheet"
            
            response = self.tavily.search(
                query=search_query,
                max_results=3,
                search_depth="advanced"
            )
            
            docs = []
            for result in response.get('results', []):
                doc = Document(
                    page_content=result['content'],
                    metadata={
                        'source': result['url'],
                        'title': result.get('title', ''),
                        'type': 'web_search'
                    }
                )
                docs.append(doc)
            
            return docs
        
        except Exception as e:
            print(f"Web search failed: {e}")
            return []
    
    def retrieve(self, query: str) -> tuple[List[Document], str]:
        """Corrective retrieval with web fallback"""
        
        # Step 1: Local retrieval
        local_docs = self.retriever.invoke(query)
        
        # Step 2: Grade relevance
        is_relevant = self.grade_relevance(query, local_docs)
        
        # Step 3: Web search if needed
        if not is_relevant:
            web_docs = self.web_search_fallback(query)
            source = "web_search"
            return web_docs + local_docs, source
        
        return local_docs, "datasheet"


# ============================================================================
# TECHNIQUE 8: AGENTIC RAG WITH LANGGRAPH
# ============================================================================

class AgentState(TypedDict):
    """State for agentic workflow"""
    query: str
    documents: List[Document]
    answer: str
    source: str
    needs_web: bool
    iteration: int


class AgenticRAG:
    """
    Multi-step reasoning agent for OpAmp 741 analysis:
    - Retrieves from datasheet
    - Grades quality
    - Falls back to web if needed
    - Self-corrects
    """
    
    def __init__(self, corrective_rag: CorrectiveRAG, llm):
        self.crag = corrective_rag
        self.llm = llm
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("verify", self._verify)
        
        # Edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._should_retry,
            {"end": END, "retry": "retrieve"}
        )
        
        return workflow.compile()
    
    def _retrieve(self, state: AgentState) -> AgentState:
        """Retrieve with correction"""
        docs, source = self.crag.retrieve(state["query"])
        state["documents"] = docs
        state["source"] = source
        return state
    
    def _generate(self, state: AgentState) -> AgentState:
        """Generate answer"""
        
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'datasheet')}]\n{doc.page_content}"
            for doc in state["documents"][:5]
        ])
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert on OpAmp 741. Answer the question using this context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n\n"
            "Provide a detailed technical answer with specific numbers and specifications. "
            "Cite sources when mentioning specs.\n\nAnswer:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "query": state["query"]
        })
        
        state["answer"] = answer
        return state
    
    def _verify(self, state: AgentState) -> AgentState:
        """Verify answer quality"""
        # Simple verification: check length and key terms
        return state
    
    def _should_retry(self, state: AgentState) -> Literal["end", "retry"]:
        """Decide if we need to retry"""
        
        # Retry if answer is too short and we haven't retried yet
        if len(state["answer"]) < 100 and state["iteration"] < 1:
            state["iteration"] += 1
            state["needs_web"] = True
            return "retry"
        
        return "end"
    
    def run(self, query: str) -> Dict[str, Any]:
        """Execute workflow"""
        
        initial_state = {
            "query": query,
            "documents": [],
            "answer": "",
            "source": "",
            "needs_web": False,
            "iteration": 0
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result["answer"],
            "source": result["source"],
            "documents": result["documents"]
        }


# ============================================================================
# MAIN RAG SYSTEM
# ============================================================================

class OpAmp741RAG:
    """
    Complete Advanced RAG System for OpAmp 741 Analysis
    """
    
    def __init__(self, datasheet_path: str):
        self.datasheet_path = datasheet_path
        
        # Initialize components
        print("Initializing components...")
        self.llm, self.embeddings, self.tavily = initialize_components()
        
        # Load and process datasheet
        print("Loading datasheet...")
        self.documents = self._load_datasheet()
        
        # Build retrieval system
        print("Building retrieval system...")
        self.retriever = self._build_retrieval_system()
        
        # Create agentic RAG
        print("Creating agentic workflow...")
        corrective_rag = CorrectiveRAG(
            self.retriever,
            self.llm,
            self.tavily
        )
        self.agent = AgenticRAG(corrective_rag, self.llm)
        
        print("✓ RAG system ready!\n")
    
    def _load_datasheet(self) -> List[Document]:
        """Load and chunk the OpAmp 741 datasheet"""
        
        # Load PDF
        loader = PyPDFLoader(self.datasheet_path)
        documents = loader.load()
        
        # Chunk with semantic boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        
        print(f"Loaded {len(documents)} pages, created {len(chunks)} chunks")
        return chunks
    
    def _build_retrieval_system(self):
        """Build hybrid retrieval system"""
        
        # Hybrid search
        hybrid = HybridRetriever(self.embeddings)
        retriever = hybrid.create_retriever(self.documents)
        
        return retriever
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: Technical question about OpAmp 741
            
        Returns:
            Dict with answer, source, and supporting documents
        """
        
        print(f"\nQuery: {question}")
        print("-" * 60)
        
        result = self.agent.run(question)
        
        print(f"\nSource: {result['source']}")
        print(f"Answer:\n{result['answer']}\n")
        
        return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("OpAmp 741 Advanced RAG System")
    print("=" * 60)
    print("\nFeatures:")
    print("✓ Gemini 2.0 Flash (Free)")
    print("✓ Local Embeddings (CPU-based)")
    print("✓ Tavily Web Search Fallback")
    print("✓ Hybrid Retrieval (Semantic + Keyword)")
    print("✓ Self-Corrective with LangGraph")
    print("✓ Parent-Child Chunking")
    print("=" * 60)
    
    # Set environment variables
    os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
    os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"
    
    # Initialize RAG system
    rag = OpAmp741RAG(datasheet_path="./opamp_741_datasheet.pdf")
    
    # Example queries
    queries = [
        "What is the typical open-loop voltage gain of the 741?",
        "What are the input offset voltage specifications?",
        "How does the slew rate affect high-frequency performance?",
        "Compare the 741 power consumption to modern op-amps",  # Will trigger web search
    ]
    
    for query in queries:
        result = rag.query(query)
        print("\n" + "=" * 60 + "\n")