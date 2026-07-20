import os
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Modern LangChain core and community imports (v0.2+ compatible)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

# In-memory session store for conversational follow-ups
session_store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieves or initializes chat memory for a specific user session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


class AgronomyRAGPipeline:
    def __init__(
        self,
        knowledge_base_dir: str = "./knowledge_base",
        persist_dir: str = "./data/chroma_db",
        # 1. Use a tiny, free 80MB local model for embeddings (perfect for laptop RAM!)
        embedding_model: str = "all-MiniLM-L6-v2",
        # 2. Use Groq's free Llama 3.3 70B model for conversational reasoning
        llm_model: str = "llama-3.3-70b-versatile",
    ):
        self.kb_dir = Path(knowledge_base_dir)
        self.persist_dir = persist_dir
        
        # FREE EMBEDDINGS: Runs locally on your CPU (zero API cost, zero network lag)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # FREE LLM: Redirects the standard OpenAI wrapper to Groq's free servers
        self.llm = ChatOpenAI(
            model=llm_model, 
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Initialize vector store
        self.vectorstore = self._build_or_load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        self.chain = None
    def _load_documents(self) -> List[Any]:
        """Scans the knowledge base directory and loads PDFs and Markdown files."""
        if not self.kb_dir.exists():
            self.kb_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Warning] Knowledge base directory created at {self.kb_dir}. Add files before querying.")
            return []

        docs = []
        # Robust PDF loading
        pdf_loader = DirectoryLoader(
            str(self.kb_dir), glob="**/*.pdf", loader_cls=PyPDFLoader
        )
        # Robust Markdown/Text loading with UTF-8 encoding
        md_loader = DirectoryLoader(
            str(self.kb_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )

        try:
            docs.extend(pdf_loader.load())
            docs.extend(md_loader.load())
            print(f"[Ingestion] Loaded {len(docs)} source documents from {self.kb_dir}")
        except Exception as e:
            print(f"[Error] Failed during document loading: {str(e)}")

        return docs

    def _build_or_load_vectorstore(self) -> Chroma:
        """Indexes documents into a local ChromaDB instance or loads existing weights."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"[VectorDB] Loading existing Chroma index from {self.persist_dir}")
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="agronomy_extension_docs",
            )

        print("[VectorDB] No index found. Building new vector database...")
        raw_docs = self._load_documents()
        
        if not raw_docs:
            # Return an empty initialized database if no documents are present yet
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="agronomy_extension_docs",
            )

        # Chunk agronomy texts by semantic paragraphs and structural headers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        chunks = text_splitter.split_documents(raw_docs)
        print(f"[VectorDB] Split into {len(chunks)} searchable chunks.")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="agronomy_extension_docs",
        )
        return vectorstore

    def setup_pipeline(self, prompt_yaml_path: str = None) -> None:
        """Compiles the LCEL RAG chain with dynamic metadata injection and memory."""
        if prompt_yaml_path is None:
            prompt_yaml_path = os.path.join(os.path.dirname(__file__), "prompt_config.yaml")
            
        with open(prompt_yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        system_template = config["template"]

        # Build prompt schema supporting system context, chat history, and user input
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        def format_retrieved_docs(docs: List[Any]) -> str:
            if not docs:
                return "No relevant extension literature found for this query."
            formatted = []
            for d in docs:
                source = d.metadata.get("source", "Unknown Manual")
                page = d.metadata.get("page", "N/A")
                formatted.append(f"--- Source: {source} (Page {page}) ---\n{d.page_content}")
            return "\n\n".join(formatted)

        # Build composite query string to ensure retriever gets crop and disease context
        def construct_search_query(inputs: dict) -> str:
            return f"{inputs['crop_name']} {inputs['detected_disease']} {inputs['question']}"

        retrieval_chain = (
            RunnableLambda(construct_search_query) 
            | self.retriever 
            | format_retrieved_docs
        )

        # Assemble core LCEL chain
        rag_chain = (
            RunnablePassthrough.assign(context=retrieval_chain)
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Wrap with stateful session memory
        self.chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        print("[Pipeline] RAG chain compiled successfully with conversation memory.")

    def query(
        self,
        session_id: str,
        user_question: str,
        crop_name: str,
        detected_disease: str,
        confidence_score: float,
    ) -> str:
        """Executes the pipeline for a given user session and visual diagnosis."""
        if not self.chain:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")

        invocation_payload = {
            "question": user_question,
            "crop_name": crop_name,
            "detected_disease": detected_disease,
            "confidence_score": f"{confidence_score:.2f}",
        }

        response = self.chain.invoke(
            invocation_payload,
            config={"configurable": {"session_id": session_id}}
        )
        return response


# =====================================================================
# Verification & Execution Simulation
# =====================================================================
if __name__ == "__main__":
    # Ensure your free Groq API key is set in your environment:
    # os.environ["GROQ_API_KEY"] = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    # 1. Initialize Pipeline (No changes needed here!)
    bot = AgronomyRAGPipeline(
        knowledge_base_dir="./knowledge_base",
        persist_dir="/data/chroma_db"
    )
    
    # Create dummy prompt config file for verification run
    dummy_yaml = """
name: "Agronomy Pathologist System Prompt"
version: "2.0.0"
template: |
  You are a certified plant pathologist. Answer based ONLY on context.
  Crop: {crop_name} | Disease: {detected_disease} | Confidence: {confidence_score}
  
  If Confidence < 0.70, refuse treatment advice and ask for a better photo.
  Otherwise, provide Cultural, Organic, and Chemical controls from context.
  
  [Verified Extension Context]
  {context}
    """
    with open("prompt_config.yaml", "w", encoding="utf-8") as f:
        f.write(dummy_yaml)

    bot.setup_pipeline("prompt_config.yaml")

    # 2. Simulate Scenario A: Low Confidence Guardrail Trigger (< 0.70)
    print("\n--- Scenario A: Low Confidence (< 0.70) ---")
    ans_low = bot.query(
        session_id="farmer_user_01",
        user_question="What should I spray to fix this immediately?",
        crop_name="Tomato",
        detected_disease="Early_Blight",
        confidence_score=0.58,  # Triggers guardrail
    )
    print(ans_low)

    # 3. Simulate Scenario B: High Confidence Diagnosis (>= 0.70)
    print("\n--- Scenario B: High Confidence (>= 0.70) ---")
    ans_high = bot.query(
        session_id="farmer_user_02",
        user_question="How should I treat this issue?",
        crop_name="Apple",
        detected_disease="Apple_Scab",
        confidence_score=0.94,  # Passes threshold
    )
    print(ans_high)

    # 4. Simulate Scenario C: Conversational Follow-up (Testing Memory)
    print("\n--- Scenario C: Follow-up Question (Memory Test) ---")
    ans_followup = bot.query(
        session_id="farmer_user_02", # Same session ID as Scenario B
        user_question="How many days should I wait between those sprays after it rains?",
        crop_name="Apple",
        detected_disease="Apple_Scab",
        confidence_score=0.94,
    )
    print(ans_followup)