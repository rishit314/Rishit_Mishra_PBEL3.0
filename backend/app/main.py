import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, PredictResponse, DiseaseContext
# 1. FIXED IMPORT: Point to the working CNNVisionService we verified!
from app.services.vision import CNNVisionService
from app.services.rag import AgronomyRAGPipeline

session_context_cache = {}
global_services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 2. FIXED PATH: Point directly to crop_disease_cnn.onnx
    default_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/data/models/crop_disease_cnn.onnx"))
    model_path = os.getenv("MODEL_PATH", default_model_path)
    
    # Check container path fallback
    if not os.path.exists(model_path) and os.path.exists("/app/data/models/crop_disease_cnn.onnx"):
        model_path = "/app/data/models/crop_disease_cnn.onnx"
    elif not os.path.exists(model_path) and os.path.exists("data/models/crop_disease_cnn.onnx"):
        model_path = "data/models/crop_disease_cnn.onnx"
        
    if not os.path.exists(model_path):
        print(f"[Warning] ONNX model missing at {model_path}. Please check file placement.")
        
    # 3. INITIALIZE: Instantiate your working CNNVisionService
    global_services["vision"] = CNNVisionService(model_path=model_path)
    
    rag = AgronomyRAGPipeline()
    rag.setup_pipeline()
    global_services["rag"] = rag
    yield
    global_services.clear()
    session_context_cache.clear()

app = FastAPI(title="AI Agronomy Production Core", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_vision_service() -> CNNVisionService:
    return global_services["vision"]

def get_rag_service() -> AgronomyRAGPipeline:
    return global_services["rag"]

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_and_summarize(
    file: UploadFile = File(...),
    session_id: str = "default_session",
    vision: CNNVisionService = Depends(get_vision_service),
    rag: AgronomyRAGPipeline = Depends(get_rag_service)
):
    """
    Ingests leaf photos, generates classification metrics via ONNX, caches context,
    and returns a localized immediate agronomic overview.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Provided document stream must be an image.")
        
    try:
        contents = await file.read()
        
        # 4. FIXED INFERENCE CALL: Use your working .predict() dictionary output!
        prediction = vision.predict(contents)
        disease = prediction["disease_id"]
        crop = prediction["crop_name"]
        score = prediction["confidence"]
        
        # Build structured container schema
        context_data = DiseaseContext(
            disease_id=disease,
            crop_name=crop,
            confidence=score
        )
        
        # Cache findings
        session_context_cache[session_id] = context_data
        
        # Generate RAG overview
        initial_prompt = f"Provide a brief one-sentence explanation of what {disease} is on {crop}."
        summary = rag.query(
            session_id=session_id,
            user_question=initial_prompt,
            crop_name=crop,
            detected_disease=disease,
            confidence_score=score
        )
        
        return PredictResponse(
            disease_id=disease,
            crop_name=crop,
            confidence=score,
            initial_summary=summary
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Pipeline Failure: {str(e)}")

@app.post("/chat")
async def stream_chat_advice(
    request: ChatRequest,
    rag: AgronomyRAGPipeline = Depends(get_rag_service)
):
    active_context = request.disease_context or session_context_cache.get(request.session_id)
    
    if not active_context:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED, 
            detail="Missing disease context metadata. Run diagnosis check endpoint `/predict` first."
        )

    async def token_generator():
        try:
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, 
                lambda: rag.query(
                    session_id=request.session_id,
                    user_question=request.user_message,
                    crop_name=active_context.crop_name,
                    detected_disease=active_context.disease_id,
                    confidence_score=active_context.confidence
                )
            )
            
            for word in response_text.split(" "):
                yield f"{word} "
                await asyncio.sleep(0.02)
                
        except asyncio.TimeoutError:
            yield "Error: Upstream AI provider API connection timeout encountered."
        except Exception as streaming_err:
            yield f"Error occurred inside processing execution layers: {str(streaming_err)}"

    return StreamingResponse(token_generator(), media_type="text/event-stream")