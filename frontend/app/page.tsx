"use client";

import React, { useState } from "react";
// Ensure this import path matches where your component sits!
import ImageUploader, { PredictResponse } from "@/components/ImageUploader";

export default function Home() {
  // 1. DEFINE THE STATE: This holds the diagnosis once the backend returns it
  const [diagnosis, setDiagnosis] = useState<PredictResponse | null>(null);

  // 2. DEFINE THE FUNCTION: This is what runs when the uploader finishes fetch()
  const handlePredictionResult = (result: PredictResponse) => {
    console.log("✅ Diagnosis received from backend:", result);
    setDiagnosis(result); // This updates the screen!
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-8 flex flex-col items-center">
      <div className="max-w-2xl w-full">
        <h1 className="text-3xl font-extrabold text-emerald-400 mb-6 text-center">
          AgroVision AI Diagnosis
        </h1>

        {/* 3. PASS THE FUNCTION TO THE COMPONENT AS A PROP */}
        {/* Without 'onPredict={handlePredictionResult}', the uploader throws 'undefined' */}
        <ImageUploader onPredict={handlePredictionResult} />

        {/* 4. RENDER THE RESULTS: This section only appears AFTER diagnosis is set */}
        {diagnosis && (
          <div className="mt-8 p-6 bg-slate-900 rounded-2xl border border-emerald-500/30 shadow-2xl animate-fade-in">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400">
                  Detected Pathology
                </span>
                <h3 className="text-2xl font-bold text-white mt-1">
                  {diagnosis.crop_name} — {diagnosis.disease_id}
                </h3>
              </div>
              <div className="bg-emerald-950/80 border border-emerald-500/50 px-4 py-2 rounded-xl text-center">
                <span className="block text-xs text-slate-400">Confidence</span>
                <span className="text-lg font-bold text-emerald-400">
                  {(diagnosis.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-slate-300 mb-2">
                📋 Initial Agronomy RAG Summary
              </h4>
              <p className="text-slate-300 text-sm leading-relaxed bg-slate-950/60 p-4 rounded-xl border border-slate-800/80">
                {diagnosis.initial_summary}
              </p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}