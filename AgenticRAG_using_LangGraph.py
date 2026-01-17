import streamlit as st
import os
import io
import sys
import json
import re
from typing import List, Any, TypedDict, Dict
import PyPDF2
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import asyncio
import nest_asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Environment Variable Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


class State(TypedDict):
    """
    Defines the shared state for the RAG workflow.
    This acts as a memory object that is passed between nodes in the graph.
    """
    file_content: bytes
    question: str
    llm: Any
    embeddings_model: Any
    full_document_text: str
    text_chunks: List[str]
    vector_store: Any
    retrieved_docs: List[str]
    confidence: float
    web_results: List[str]
    answer: str

def run_agentic_rag():
    """
    Constructs and compiles the agentic RAG graph using LangGraph.
    This function defines the complete workflow logic.
    """
    # Initialize the Tavily search tool for web-based fallback retrieval.
    web_search_tool = TavilySearchResults(max_results=4, tavily_api_key=TAVILY_API_KEY)

    # --- Graph Nodes ---
    # The following functions are nodes in the graph. Each performs a
    # specific action and returns a dictionary to update the state.

    def initialize_models_node(state: State) -> Dict[str, Any]:
        """Initializes the generative and embedding models required for the workflow."""
        print("--- Initializing Models ---")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        return {"llm": llm, "embeddings_model": embeddings_model}

    def load_pdf_text_node(state: State) -> Dict[str, str]:
        """Extracts text content from the uploaded PDF file."""
        print("--- Loading PDF Text ---")
        text = ""
        try:
            # Read the PDF from the in-memory bytes content.
            reader = PyPDF2.PdfReader(io.BytesIO(state["file_content"]))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"PDF read error: {e}")
        return {"full_document_text": text}

    def chunk_text_node(state: State) -> Dict[str, List[str]]:
        """Splits the full document text into smaller, overlapping chunks for processing."""
        print("--- Chunking Text ---")
        text = state["full_document_text"]
        chunks, chunk_size, overlap = [], 1000, 200
        # Overlapping chunks help maintain context across boundaries.
        for start in range(0, len(text), chunk_size - overlap):
            chunks.append(text[start:start + chunk_size])
        return {"text_chunks": chunks}

    def create_chroma_index_node(state: State) -> Dict[str, Any]:
        """Creates a ChromaDB vector store from text chunks for efficient similarity search."""
        print("--- Creating Chroma Index ---")
        try:
            # The vector store embeds each chunk for fast retrieval.
            vector_store = Chroma.from_texts(texts=state["text_chunks"], embedding=state["embeddings_model"])
            return {"vector_store": vector_store}
        except Exception as e:
            print(f"Chroma error: {e}")
            return {"vector_store": None}

    def retrieve_node(state: State) -> Dict[str, Any]:
        """Retrieves relevant document snippets from the vector store based on the user's query."""
        print("--- Retrieving Documents ---")
        if state.get("vector_store") is None:
             return {"retrieved_docs": []}
        # The retriever performs a similarity search on the vector store.
        retriever = state["vector_store"].as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(state["question"])
        snippets = [doc.page_content for doc in docs]
        return {"retrieved_docs": snippets}

    def evaluate_node(state: State) -> Dict[str, Any]:
        """Evaluates the relevance of retrieved snippets and generates a confidence score."""
        print("--- Evaluating Relevance ---")
        prompt = PromptTemplate(
            template="""You are a grader assessing the relevance of retrieved snippets to a user question.
            Aggregate the relevance of all snippets and provide a single confidence score from 0.0 to 1.0.
            Respond ONLY with a JSON object containing a single key "confidence".

            Question: {question}
            Snippets:
            {snippets}""",
            input_variables=["question", "snippets"],
        )

        evaluator_chain = prompt | state["llm"] | JsonOutputParser()
        snippets_text = "\n\n".join(state["retrieved_docs"])
        confidence = 0.0
        try:
            response_json = evaluator_chain.invoke({
                "question": state["question"],
                "snippets": snippets_text
            })
            confidence = float(response_json.get("confidence", 0.0))
        except Exception as e:
            print(f"Confidence parsing error: {e}. Defaulting to 0.0.")

        print(f"Confidence Score: {confidence}")
        return {"confidence": confidence}

    def web_search_node(state: State) -> Dict[str, Any]:
        """Performs a web search if document retrieval is insufficient."""
        print("--- Performing Web Search ---")
        hits = []
        try:
            # Use the Tavily tool as a fallback to search the web.
            results = web_search_tool.invoke({"query": state["question"]})
            hits = [r.get("content", "") for r in results if isinstance(r, dict) and r.get("content")]
        except Exception as e:
            print(f"Web search failed: {e}")
        return {"web_results": hits}

    def generate_node(state: State) -> Dict[str, Any]:
        """Generates the final answer using context from document snippets or web results."""
        print("--- Generating Final Answer ---")
        context = ""

        # Use document context if confidence is high; otherwise, use web results.
        if state.get("confidence", 0.0) >= 0.6 and state.get("retrieved_docs"):
            print("--- Using documents as context ---")
            context = "\n\n".join(state["retrieved_docs"])
        elif state.get("web_results"):
            print("--- Using web search results as context ---")
            context = "\n\n".join(state["web_results"])
        else:
            # Fallback if no relevant context is found.
            return {"answer": "I couldn't find any relevant information in the PDF or online."}

        prompt = (
            f"You are a helpful assistant. Answer the question using only the context below. "
            f"If no answer can be found, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\nAnswer:"
        )
        response = state["llm"].invoke(prompt)
        return {"answer": response.content}

    def summarize_node(state: State) -> Dict[str, Any]:
        """Generates a concise summary of the document."""
        print("--- Generating Summary ---")
        # Use the initial part of the document for summarization.
        doc_text = state["full_document_text"][:15000]
        prompt = f"Summarize the following document:\n\n{doc_text}"
        response = state["llm"].invoke(prompt)
        return {"answer": response.content}

    # --- Graph Construction ---
    # The following section constructs the graph by defining nodes and their connections.

    workflow = StateGraph(State)

    # Add each function as a node to the graph.
    workflow.add_node("initialize_models", initialize_models_node)
    workflow.add_node("load_pdf_text", load_pdf_text_node)
    workflow.add_node("chunk_text", chunk_text_node)
    workflow.add_node("create_chroma_index", create_chroma_index_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("summarize", summarize_node)

    # Define the primary sequence of operations.
    workflow.set_entry_point("initialize_models")
    workflow.add_edge("initialize_models", "load_pdf_text")
    workflow.add_edge("load_pdf_text", "chunk_text")
    workflow.add_edge("chunk_text", "create_chroma_index")
    workflow.add_edge("create_chroma_index", "retrieve")

    # --- Conditional Edges ---
    # Conditional edges direct the workflow based on the current state.

    def route_after_retrieval(state: State):
        """Routes to summarization if requested; otherwise, proceeds to evaluation."""
        return "summarize" if state["question"].lower().strip() == "summarize" else "evaluate"

    def route_after_evaluation(state: State):
        """Routes to answer generation if confidence is high; otherwise, falls back to web search."""
        return "generate" if state.get("confidence", 0.0) >= 0.6 else "web_search"

    # Add the conditional routing logic to the graph.
    workflow.add_conditional_edges("retrieve", route_after_retrieval, {
        "summarize": "summarize",
        "evaluate": "evaluate"
    })
    workflow.add_conditional_edges("evaluate", route_after_evaluation, {
        "generate": "generate",
        "web_search": "web_search"
    })

    # Define the final paths to the end state.
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("summarize", END)

    # Compile the graph into a runnable application.
    return workflow.compile()

async def get_rag_response(rag_app, initial_state):
    """Asynchronously invokes the RAG application and returns the final state."""
    return await rag_app.ainvoke(initial_state)

def main():
    """Main function to set up the Streamlit UI and execute the RAG workflow."""
    # Allow asyncio event loops to run within Streamlit's execution model.
    nest_asyncio.apply()

    st.set_page_config(page_title="Agentic RAG with Streamlit", layout="wide")
    st.title("Agentic RAG Chatbot")

    # Verify that the necessary API keys are configured as environment variables.
    if not GEMINI_API_KEY or not TAVILY_API_KEY:
        st.error("API keys for Gemini or Tavily are not set. Please set them as environment variables.")
        st.stop()

    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        # Read the uploaded file content into memory.
        file_content = uploaded_file.getvalue()
        st.sidebar.success("File uploaded successfully!")

        # Get the user's query from the text input.
        user_query = st.text_input("Ask a question about the document (or type 'summarize'):", "")

        if st.button("Get Answer"):
            if not user_query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("The agent is thinking... This may take a moment."):
                    # Build the RAG application graph.
                    rag_app = run_agentic_rag()
                    # Define the initial state for the workflow.
                    initial_state = {
                        "file_content": file_content,
                        "question": user_query,
                    }
                    
                    # Execute the RAG workflow.
                    final_state = asyncio.run(get_rag_response(rag_app, initial_state))
                    
                    # Display the generated answer in the Streamlit interface.
                    st.write("### Answer:")
                    st.write(final_state.get('answer', 'No answer was generated.'))
    else:
        st.info("Please upload a PDF file to begin.")

if __name__ == "__main__":
    main()
