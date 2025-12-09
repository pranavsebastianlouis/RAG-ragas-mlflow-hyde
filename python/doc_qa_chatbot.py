"""
FastAPI Document QA Server with RAG and Evaluation
A REST API for conversational AI that can answer questions about PDF documents using retrieval-augmented generation.
Includes RAGAS evaluation and MLflow tracking.
Supports HyDE (Hypothetical Document Embeddings) for enhanced retrieval.
"""

import os
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import weaviate
import tempfile
import shutil
import mlflow

# Import evaluation module - make it optional to prevent startup failures
try:
    from python.evaluation import RAGEvaluator
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import evaluation module: {e}")
    print("   Evaluation endpoints will not be available.")
    EVALUATION_AVAILABLE = False
    RAGEvaluator = None

# Import HyDE module
try:
    from python.hyde_retriever import create_hyde_retriever, create_hybrid_retriever
    HYDE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import HyDE module: {e}")
    print("   HyDE retrieval will not be available.")
    HYDE_AVAILABLE = False


# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"


class QueryResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    sources: List[Dict]


class EvaluationRequest(BaseModel):
    test_cases: List[Dict]  # List of {question, ground_truth (optional)}
    run_name: Optional[str] = None


class EvaluationResponse(BaseModel):
    scores: Dict[str, float]
    mlflow_run_id: str
    num_test_cases: int
    detailed_results: List[Dict]


class SingleEvalRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class PDFUploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    weaviate_connected: bool
    llm_ready: bool
    embeddings_ready: bool
    mlflow_tracking_uri: str
    hyde_available: bool
    using_hyde: bool


# Global chatbot instance
chatbot = None


class DocumentQAChatbot:
    """Main chatbot class for document-based question answering."""
    
    def __init__(self, weaviate_url: str = None, ollama_base_url: str = None):
        """
        Initialize the chatbot.
        
        Args:
            weaviate_url: URL for Weaviate vector database
            ollama_base_url: Base URL for Ollama server
        """
        # Get URLs from environment or use defaults
        self.weaviate_url = weaviate_url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embeddings = None
        self.db = None
        self.llm = None
        self.conv_rag_chain = None
        self.store = {}  # Session history storage
        self.client = None
        self.collection_name = "Newspaperchunks"
        self.evaluator = None
        self.use_hyde = False  # HyDE flag
        self.retriever = None  # Store retriever for reuse
        
    def initialize_embeddings(self):
        """Initialize the embedding model."""
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        print("✓ Embedding model loaded")
        
    def initialize_llm(self, model_name: str = "mistral", temperature: float = 0):
        """
        Initialize the language model.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature
        """
        print(f"Initializing LLM ({model_name})...")
        self.llm = OllamaLLM(
            model=model_name, 
            temperature=temperature,
            base_url=self.ollama_base_url
        )
        print("✓ LLM initialized")
        
    def initialize_evaluator(self, experiment_name: str = "document_qa_evaluation"):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            experiment_name: Name for the MLflow experiment
        """
        if not EVALUATION_AVAILABLE:
            print("⚠️  Evaluation module not available. Skipping evaluator initialization.")
            return False
            
        try:
            print("Initializing RAGAS evaluator...")
            # Set MLflow tracking URI
            mlflow.set_tracking_uri("file:./mlruns")
            # Initialize evaluator
            self.evaluator = RAGEvaluator(experiment_name=experiment_name)
            print("✓ RAGAS evaluator initialized")
            return True
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize evaluator: {e}")
            print("   Evaluation endpoints will not be available.")
            self.evaluator = None
            return False
        
    def connect_weaviate(self):
        """Connect to Weaviate database."""
        print(f"Connecting to Weaviate at {self.weaviate_url}...")
        # Parse host and port from URL
        if "localhost" in self.weaviate_url or "127.0.0.1" in self.weaviate_url:
            self.client = weaviate.connect_to_local()
        else:
            # For Docker or remote connections
            from weaviate.connect import ConnectionParams
            # Extract host from URL (remove http:// and port)
            host = self.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0]
            port = int(self.weaviate_url.split(":")[-1])
            self.client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=False
            )
        
        if self.client.is_ready():
            print("✓ Connected to Weaviate")
        else:
            raise ConnectionError("Failed to connect to Weaviate")
            
    def load_and_process_pdf(self, pdf_path: str, collection_name: Optional[str] = None) -> Dict:
        """
        Load PDF, split into chunks, and store in vector database.
        
        Args:
            pdf_path: Path to the PDF file
            collection_name: Name for the Weaviate collection
            
        Returns:
            Dictionary with processing stats
        """
        if collection_name:
            self.collection_name = collection_name
            
        # Load PDF
        print(f"Loading PDF from {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        pages = len(documents)
        print(f"✓ Loaded {pages} pages from PDF")
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        chunks = len(splits)
        print(f"✓ Split into {chunks} chunks")
        
        # Delete existing collection if it exists
        if self.client.collections.exists(self.collection_name):
            print(f"Deleting existing collection '{self.collection_name}'...")
            self.client.collections.delete(self.collection_name)
        
        # Store in Weaviate
        print("Storing embeddings in Weaviate...")
        self.db = WeaviateVectorStore.from_documents(
            splits, 
            self.embeddings, 
            client=self.client, 
            index_name=self.collection_name
        )
        print("✓ Documents stored in vector database")
        
        return {"pages": pages, "chunks": chunks}
        
    def setup_rag_chain(self, use_hyde: bool = False):
        """
        Set up the conversational RAG chain.
        
        Args:
            use_hyde: Whether to use HyDE for retrieval enhancement
        """
        if self.db is None:
            # Try to load existing collection
            if self.client.collections.exists(self.collection_name):
                print(f"Loading existing collection '{self.collection_name}'...")
                self.db = WeaviateVectorStore(
                    client=self.client,
                    index_name=self.collection_name,
                    text_key="text",
                    embedding=self.embeddings
                )
            else:
                raise ValueError("No documents loaded. Please upload a PDF first.")
        
        print(f"Setting up RAG chain (HyDE: {use_hyde})...")
        self.use_hyde = use_hyde
        
        # Create retriever based on HyDE flag
        if use_hyde and HYDE_AVAILABLE:
            print("  Using HyDE retriever...")
            self.retriever = create_hyde_retriever(
                llm=self.llm,
                embeddings=self.embeddings,
                vectorstore=self.db,
                k=5
            )
        else:
            print("  Using standard retriever...")
            self.retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        # Context-aware question reformulation prompt
        context_q_sys_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone "
            "question which can be understood without the chat history. Do NOT answer the "
            "question, just reformulate if required and otherwise return as is."
        )
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", context_q_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, context_prompt
        )
        
        # Question-answering prompt
        sys_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. Keep the answer concise.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Create QA chain
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Combine retrieval and QA
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        
        # Wrap with message history
        self.conv_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        retriever_type = "HyDE" if use_hyde else "Standard"
        print(f"✓ RAG chain ready with {retriever_type} retriever")
        
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieve or create chat history for a session.
        
        Args:
            session_id: Unique identifier for the conversation
            
        Returns:
            ChatMessageHistory object
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
        
    def ask(self, question: str, session_id: str = "default_session") -> Dict:
        """
        Ask a question to the chatbot.
        
        Args:
            question: The question to ask
            session_id: Session identifier for conversation context
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if self.conv_rag_chain is None:
            raise ValueError("RAG chain not initialized. Please upload a PDF first.")
            
        response = self.conv_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        return response
        
    def reset_conversation(self, session_id: str = "default_session"):
        """
        Clear chat history for a session.
        
        Args:
            session_id: Session to reset
        """
        if session_id in self.store:
            del self.store[session_id]
            return True
        return False
            
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            self.client.close()
            print("✓ Weaviate connection closed")
            
    def is_ready(self) -> Dict[str, bool]:
        """Check if all components are ready."""
        return {
            "weaviate_connected": self.client is not None and self.client.is_ready(),
            "llm_ready": self.llm is not None,
            "embeddings_ready": self.embeddings is not None,
            "rag_ready": self.conv_rag_chain is not None,
            "evaluator_ready": self.evaluator is not None
        }


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)."""
    global chatbot
    # Startup
    print("Starting FastAPI server...")
    chatbot = DocumentQAChatbot()
    chatbot.initialize_embeddings()
    # Use smaller model - change to "llama3.2:1b" or "qwen2.5:0.5b" for low memory
    chatbot.initialize_llm(model_name="mistral")  
    chatbot.connect_weaviate()
    chatbot.initialize_evaluator()
    print("✓ Server ready")
    
    yield
    
    # Shutdown
    if chatbot:
        chatbot.cleanup()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Document QA API",
    description="REST API for document-based question answering using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document QA API with RAGAS Evaluation & HyDE",
        "version": "3.0.0",
        "features": {
            "rag": "Retrieval Augmented Generation",
            "evaluation": "RAGAS metrics with MLflow",
            "hyde": "Hypothetical Document Embeddings",
            "conversation": "Session-based chat history"
        },
        "endpoints": {
            "POST /upload": "Upload a PDF document",
            "POST /query": "Ask a question about the document",
            "POST /evaluate": "Evaluate the RAG pipeline with test cases",
            "POST /evaluate/single": "Evaluate a single query",
            "POST /evaluate/compare-retrievers": "Compare baseline vs HyDE",
            "GET /evaluate/history": "Get evaluation history",
            "POST /toggle-hyde": "Toggle HyDE retrieval on/off",
            "POST /reset": "Reset conversation history",
            "GET /health": "Check API health status"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized - server may still be starting")
    
    status = chatbot.is_ready()
    return HealthResponse(
        status="healthy" if all(status.values()) else "degraded",
        weaviate_connected=status["weaviate_connected"],
        llm_ready=status["llm_ready"],
        embeddings_ready=status["embeddings_ready"],
        mlflow_tracking_uri=mlflow.get_tracking_uri(),
        hyde_available=HYDE_AVAILABLE,
        using_hyde=chatbot.use_hyde
    )


@app.get("/debug/status")
async def debug_status():
    """Debug endpoint to check initialization status."""
    global chatbot
    
    return {
        "chatbot_initialized": chatbot is not None,
        "evaluator_initialized": chatbot.evaluator is not None if chatbot else False,
        "evaluation_available": EVALUATION_AVAILABLE,
        "chatbot_ready": chatbot.is_ready() if chatbot else {},
        "rag_chain_ready": chatbot.conv_rag_chain is not None if chatbot else False,
        "mlflow_uri": mlflow.get_tracking_uri()
    }


@app.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload
        
    Returns:
        Upload confirmation with processing stats
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Process the PDF
        stats = chatbot.load_and_process_pdf(tmp_path)
        
        # Setup RAG chain
        chatbot.setup_rag_chain()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return PDFUploadResponse(
            message="PDF uploaded and processed successfully",
            filename=file.filename,
            pages=stats["pages"],
            chunks=stats["chunks"]
        )
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        file.file.close()


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Ask a question about the uploaded document.
    
    Args:
        request: Query request with question and optional session_id
        
    Returns:
        Answer with sources
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if chatbot.conv_rag_chain is None:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a PDF first.")
    
    try:
        # Get response from chatbot
        response = chatbot.ask(request.question, request.session_id)
        
        # Extract source documents
        sources = []
        if "context" in response:
            for doc in response["context"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        return QueryResponse(
            question=request.question,
            answer=response["answer"],
            session_id=request.session_id,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/reset")
async def reset_session(session_id: str = "default_session"):
    """
    Reset conversation history for a session.
    
    Args:
        session_id: Session to reset
        
    Returns:
        Confirmation message
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    success = chatbot.reset_conversation(session_id)
    
    if success:
        return {"message": f"Session '{session_id}' reset successfully"}
    else:
        return {"message": f"No active session found for '{session_id}'"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pipeline(request: EvaluationRequest):
    """
    Evaluate the RAG pipeline with multiple test cases.
    
    Args:
        request: Evaluation request with test cases
        
    Returns:
        Evaluation scores and MLflow run ID
    """
    global chatbot
    
    # Check if evaluation is available
    if not EVALUATION_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Evaluation module not available. Please ensure 'evaluation.py' is in the same directory and all dependencies are installed (pip install ragas datasets)."
        )
    
    # Check initialization
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized. Server may be starting up.")
    
    if chatbot.evaluator is None:
        raise HTTPException(
            status_code=503, 
            detail="Evaluator not initialized. Check server logs for initialization errors. You may need to install: pip install ragas datasets mlflow"
        )
    
    if chatbot.conv_rag_chain is None:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a PDF first using POST /upload.")
    
    try:
        # Generate answers for each test case
        print(f"Evaluating {len(request.test_cases)} test cases...")
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        
        for i, test_case in enumerate(request.test_cases):
            question = test_case["question"]
            questions.append(question)
            
            print(f"  Processing question {i+1}/{len(request.test_cases)}: {question[:50]}...")
            
            # Get answer from chatbot
            response = chatbot.ask(question, session_id=f"eval_{i}")
            answers.append(response["answer"])
            
            # Extract contexts
            contexts = [doc.page_content for doc in response.get("context", [])]
            contexts_list.append(contexts)
            
            # Get ground truth if provided
            ground_truths.append(test_case.get("ground_truth"))
        
        # Check if we have any ground truths
        has_ground_truth = any(gt is not None for gt in ground_truths)
        
        print("  Running RAGAS evaluation...")
        # Run evaluation using chatbot's evaluator
        result = chatbot.evaluator.evaluate_pipeline(
            questions=questions,
            answers=answers,
            contexts=contexts_list,
            ground_truths=ground_truths if has_ground_truth else None,
            run_name=request.run_name
        )
        
        print("✓ Evaluation complete!")
        
        return EvaluationResponse(
            scores=result["scores"],
            mlflow_run_id=result["mlflow_run_id"],
            num_test_cases=len(request.test_cases),
            detailed_results=result["detailed_results"]
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Evaluation error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")


@app.post("/evaluate/single")
async def evaluate_single_query(request: SingleEvalRequest):
    """
    Evaluate a single query-answer pair.
    
    Args:
        request: Single evaluation request
        
    Returns:
        Evaluation scores
    """
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        scores = evaluator.evaluate_single_query(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth
        )
        
        return {
            "question": request.question,
            "scores": scores,
            "message": "Single query evaluated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")


@app.get("/evaluate/history")
async def get_evaluation_history():
    """
    Get history of all evaluation runs.
    
    Returns:
        List of evaluation runs with scores
    """
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        history = RAGEvaluator.get_experiment_history()
        
        if history.empty:
            return {
                "message": "No evaluation history found",
                "runs": []
            }
        
        # Convert to dict for JSON serialization
        runs = history.to_dict('records')
        
        return {
            "message": f"Found {len(runs)} evaluation runs",
            "runs": runs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.get("/evaluate/compare")
async def compare_evaluation_runs(run_ids: str):
    """
    Compare multiple evaluation runs.
    
    Args:
        run_ids: Comma-separated list of MLflow run IDs
        
    Returns:
        Comparison of runs
    """
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        run_id_list = [rid.strip() for rid in run_ids.split(",")]
        comparison = RAGEvaluator.compare_runs(run_id_list)
        
        return {
            "message": f"Comparing {len(run_id_list)} runs",
            "comparison": comparison.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing runs: {str(e)}")


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "doc_qa_chatbot:app",  # Fixed module name
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()