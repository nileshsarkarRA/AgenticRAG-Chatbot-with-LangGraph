Key Features
1. Free & Local Components:

Gemini 2.0 Flash (Free tier) for LLM
HuggingFace Embeddings (all-MiniLM-L6-v2) - runs locally on CPU
Tavily for web search fallback

2. Advanced Techniques Implemented:

✅ Hierarchical Parent-Child Retrieval - precise specs with full context
✅ Hybrid Search - semantic + BM25 keyword matching
✅ HyDE - hypothetical answer generation
✅ Query Decomposition - breaks complex questions
✅ Corrective RAG - automatic web search fallback
✅ Agentic Workflow (LangGraph) - self-correcting multi-step reasoning
✅ Contextual Enrichment - adds context to chunks

3. OpAmp-Specific Optimizations:

Tuned for technical specifications
Handles part numbers and variations (741, µA741, LM741)
Combines datasheet precision with web knowledge

📦 Installation
bashpip install langchain langchain-google-genai langchain-community
pip install faiss-cpu sentence-transformers tavily-python
pip install pypdf langgraph rank-bm25
🔑 Setup
Set your API keys:
pythonos.environ["GOOGLE_API_KEY"] = "your-key"
os.environ["TAVILY_API_KEY"] = "your-key"
🚀 Usage
python# Initialize system
rag = OpAmp741RAG(datasheet_path="./opamp_741_datasheet.pdf")

# Query
result = rag.query("What is the slew rate of the 741?")
print(result["answer"])
The system automatically:

Searches the local datasheet
Grades result quality
Falls back to Tavily web search if needed
Combines both sources for comprehensive answers

All techniques are modular - you can enable/disable any feature by modifying the _build_retrieval_system() method! 🎨