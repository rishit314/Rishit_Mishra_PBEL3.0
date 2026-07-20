# 🌱 AgroVision AI — Crop Disease Detection & RAG Consult bot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://rishit-mishra-pbel-3-0.vercel.app)
[![Backend](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render)](https://agrovision-ai-api.onrender.com/docs)

> **🚀 [Try the Live Interactive Demo Here!](https://your-app-name.vercel.app)**
>
> *💡 **DevOps Note:** The FastAPI microservice backend is hosted on a serverless free tier to optimize infrastructure costs. If the app has been idle, **the initial diagnostic request may take ~30 seconds** to wake up the Docker container from cold sleep. All subsequent queries respond instantly!*

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/rishit314/Rishit_Mishra_PBEL3.0)
![Docker](https://img.shields.io/badge/Docker-Microservices-2496ED?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-009688?style=flat&logo=fastapi&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Edge%20Optimized-005CED?style=flat&logo=onnx&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat&logo=next.dot.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-Strict-3178C6?style=flat&logo=typescript&logoColor=white)

An end-to-end, Dockerized machine learning platform that combines **low-latency edge computer vision** with a **context-aware agronomy RAG (Retrieval-Augmented Generation) pipeline** to diagnose plant pathologies and stream actionable, localized treatment protocols in real time.

---

## ⚡ System Architecture & "Two-Brain" Design

AgroVision AI decouples visual diagnosis from clinical reasoning by splitting inference into a synchronized two-step pipeline:

```text
[ React / Next.js UI ]
        │  (Multipart Image Payload)
        ▼
[ FastAPI API Gateway ] ──► [ ONNX Vision Engine ] ──► Predicts Class (e.g., Cherry__Healthy : 91.0%)
        │                                                     │
        │  (Decouples Crop & Disease ID)                      │
        ▼                                                     ▼
[ LangChain RAG Core ] ◄── Context Cache & Vector Search ◄────┘
        │
        ▼  (Server-Sent Events / SSE Streaming)
[ Real-Time Treatment Protocol Display ]
```

- **The Visual Brain (ONNX Edge Engine):** Instead of running heavy PyTorch dependencies in production, models are exported to lightweight ONNX (Open Neural Network Exchange) graphs. Using custom float32 tensor mathematical preprocessing (CHW transposition and ImageNet normalization), the CPU execution provider achieves sub-50ms inference on low-cost compute nodes.
- **The Agronomy Brain (LangChain RAG):** Once a pathology is classified across our 29-class taxonomy, the backend queries a localized vector knowledge base. It synthesizes an immediate diagnostic summary and initializes a stateful, streaming Server-Sent Events (SSE) chat interface for deep-dive agronomic consultation.

---

## Key Features

- **29-Class Pathology Classification:** Custom-trained EfficientNet-B0 network capable of distinguishing complex fungal, bacterial, and viral plant diseases across crops like tomatoes, apples, potatoes, corn, and grapes.
- **Zero-Blocking SSE Streaming:** The `/chat` endpoint bypasses I/O blocking by streaming markdown treatment advice token-by-token using asynchronous Python generators and LangChain runnables.
- **Resilient Docker Microservices:** Completely containerized frontend, backend, and persistence layers with isolated volume mounting for hot-swapping model weights (`.onnx`) without rebuilding containers.
- **Strict TypeScript / UI Reliability:** A responsive Tailwind CSS frontend utilizing polymorphic TypeScript interfaces (`onPredict`, `onResult`) to prevent rendering crashes and state loss during asynchronous fetches.

---

## 🛠️ Technology Stack

| Layer | Technologies Used | Key Responsibilities |
|---|---|---|
| **Frontend** | Next.js, React, TypeScript, Tailwind CSS | Drag-and-drop ingestion, state management, real-time SSE chat rendering |
| **API Gateway** | Python 3.10+, FastAPI, Pydantic v2, Uvicorn | Request validation, asynchronous I/O, CORS handling, session context caching |
| **Vision Engine** | ONNX Runtime, NumPy, Pillow (PIL) | Tensor normalization, softmax probability decoding, argmax classification |
| **RAG / AI Core** | LangChain, Vector Storage (ChromaDB/SQLite) | Contextual information retrieval, prompt engineering, streaming LLM wrappers |
| **DevOps** | Docker, Docker Compose, Git | Multi-stage container builds, volume persistence, environment isolation |

---

## 🚀 Quickstart & Local Deployment

Because the entire infrastructure is containerized, you can launch the production stack on any OS with just three commands—no local Python or Node modules required.

### Prerequisites

- Docker Desktop installed and running.
- Git installed.

### 1. Clone the Repository

```bash
git clone https://github.com/rishit314/Rishit_Mishra_PBEL3.0.git
cd Rishit_Mishra_PBEL3.0
```

### 2. Configure Environment Secrets

Copy the template environment file to create your local secrets configuration:

```bash
# On Windows PowerShell:
Copy-Item .env.example .env

# On macOS/Linux:
cp .env.example .env
```

Open `.env` and insert your LLM API keys if utilizing the active live RAG chat streaming.

### 3. Build and Launch the Microservices

```bash
docker compose up -d --build
```

- 🌐 **Frontend Application:** [http://localhost:3000](http://localhost:3000)
- 🔌 **FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Model & Dataset Taxonomy

The ONNX vision engine is trained to recognize 29 distinct classes, ensuring precision across common agricultural disease vectors:

- **Apple:** Scab, Black Rot, Cedar Rust, Healthy
- **Bell Pepper:** Bacterial Spot, Healthy
- **Cherry:** Powdery Mildew, Healthy
- **Corn (Maize):** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- **Grape:** Black Rot, Esca (Black Measles), Leaf Blight, Healthy
- **Peach:** Bacterial Spot, Healthy
- **Potato:** Early Blight, Late Blight, Healthy
- **Strawberry:** Leaf Scorch, Healthy
- **Tomato:** Bacterial Spot, Early Blight, Late Blight, Septoria Leaf Spot, Yellow Leaf Curl Virus, Healthy

---

## 💻 API Endpoints Summary

### `POST /predict`

Accepts multipart image form data. Returns pathology classification, exact confidence metrics, and a synthesized agronomic RAG summary.

```json
{
  "disease_id": "Healthy",
  "crop_name": "Cherry",
  "confidence": 0.9097,
  "initial_summary": "Detected Healthy on Cherry with 91.0% confidence."
}
```

### `POST /chat`

Accepts JSON session context and conversational queries. Returns an asynchronous Text/Event-Stream (SSE) delivering real-time guidance based on the prior prediction state.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.