# 🌿 AgroVision AI — Plant Pathology & Diagnostic RAG System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://rishit-mishra-pbel-3-0.vercel.app)
[![Backend API](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render)](https://agrovision-ai-api.onrender.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
> **🚀 [Try the Live Interactive Demo Here!](https://rishit-mishra-pbel-3-0.vercel.app)**
> *💡 **DevOps Note:** The FastAPI backend is hosted on a serverless free tier to optimize infrastructure costs. If the app has been idle, **the initial diagnostic request may take ~30 seconds** to wake up the Docker container from cold sleep. All subsequent queries respond instantly!*

## 📖 Overview

AgroVision AI is a full-stack, production-ready machine learning application designed to identify crop diseases from leaf images and instantly provide actionable, context-aware treatment plans. 

By combining a **Convolutional Neural Network (CNN)** for rapid image classification with a **Retrieval-Augmented Generation (RAG)** pipeline, the system bridges the gap between raw computer vision and practical agronomy advice. The architecture is explicitly designed for low-memory environments, successfully containerizing complex ML models into a footprint of less than 512MB.

## ✨ Key Features

* **CNN Image Classification:** Utilizes a custom-trained Convolutional Neural Network (CNN) exported to ONNX format to accurately diagnose plant diseases directly from uploaded leaf images.
* **RAG-Powered Treatment Plans:** LangChain and ChromaDB index vetted agricultural knowledge, streaming real-time, highly relevant mitigation strategies using the Groq LLM API.
* **Memory-Optimized Inference:** Eliminated 2.4GB of heavy PyTorch dependencies by switching to CPU-native ONNX Runtime and FastEmbed, ensuring the entire microservice runs on a lightweight free-tier cloud container.
* **Decoupled Architecture:** A responsive Next.js frontend securely communicates with an isolated FastAPI Docker backend via REST APIs.

## 🛠️ Technology Stack

**Frontend (Client)**
* **Framework:** Next.js, React
* **Styling:** Tailwind CSS
* **Deployment:** Vercel

**Backend (Microservice)**
* **Framework:** FastAPI, Uvicorn
* **Containerization:** Docker, Docker Compose
* **Deployment:** Render (Free Tier)

**Machine Learning & AI**
* **Computer Vision:** Convolutional Neural Network (CNN), ONNX Runtime, OpenCV
* **RAG Pipeline:** LangChain, ChromaDB, FastEmbed (ONNX-native embeddings)
* **LLM Engine:** Groq API

## 🚀 Local Development Setup

To run this project locally on your machine, you will need [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/Rishit_Mishra_PBEL3.0.git](https://github.com/yourusername/Rishit_Mishra_PBEL3.0.git)
cd Rishit_Mishra_PBEL3.0
```

### 2. Set up Environment Variables
Create a `.env` file in the root directory (or inside the `backend/` folder depending on your compose setup) and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

### 3. Build and run with Docker Compose
The `docker-compose.yml` is configured to build the backend from the `./backend` context automatically.
```bash
docker compose up -d --build backend
```

### 4. Access the Local App
* **Backend API / Swagger UI:** `http://localhost:8000/docs`
* **Frontend:** If running the Next.js app locally, run `npm run dev` in the `frontend` folder and visit `http://localhost:3000`.

## 🧠 System Architecture

1. **User Upload:** User submits a leaf photo via the Vercel-hosted Next.js frontend.
2. **REST API:** The image is sent as a `multipart/form-data` POST request to the FastAPI backend.
3. **CNN Inference:** The image is preprocessed using OpenCV and passed through the ONNX-optimized CNN model to extract the disease classification.
4. **Vector Search:** The classification string triggers a similarity search in ChromaDB (using FastEmbed) to retrieve relevant agricultural treatments.
5. **LLM Generation:** The retrieved context and classification are passed to Groq to generate a final, readable treatment plan.
6. **Client Render:** The JSON response is returned to the frontend and rendered in dynamic UI cards.
