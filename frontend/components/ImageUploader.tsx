"use client";

import React, { useState, useRef, useCallback } from "react";

export interface PredictResponse {
  disease_id: string;
  crop_name: string;
  confidence: number;
  initial_summary: string;
}

export interface ImageUploaderProps {
  onPredictSuccess?: (result: any) => void;
  onPredict?: (result: any) => void;
  onResult?: (result: any) => void;
  [key: string]: any; // This index signature prevents TypeScript from ever blocking custom props again!
}

export default function ImageUploader({
  onPredict,
  onPredictSuccess,
  onResult,
  apiEndpoint = "http://localhost:8000/predict",
}: ImageUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [errorToast, setErrorToast] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const triggerToast = (message: string) => {
    setErrorToast(message);
    setTimeout(() => setErrorToast(null), 5000);
  };

  const handleFileSelection = useCallback((selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      triggerToast("Invalid file type. Please upload a valid JPEG, PNG, or WebP image.");
      return;
    }
    if (selectedFile.size > 10 * 1024 * 1024) {
      triggerToast("File size exceeds the 10MB limit.");
      return;
    }

    setFile(selectedFile);
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
        e.dataTransfer.clearData();
      }
    },
    [handleFileSelection]
  );

  const handleUploadAndPredict = async () => {
    if (!file) {
      triggerToast("Please select a plant leaf image first.");
      return;
    }

    setIsLoading(true);
    setErrorToast(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("session_id", "session_" + Math.random().toString(36).substring(2, 9));

    try {
      const response = await fetch(apiEndpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned status ${response.status}`);
      }

      const data: PredictResponse = await response.json();
      onPredict(data);
    } catch (err: any) {
      triggerToast(`Diagnosis failed: ${err.message || "Network connection error"}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative flex flex-col w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      {/* Error Toast Notification */}
      {errorToast && (
        <div className="absolute top-4 left-4 right-4 z-50 bg-rose-600/90 text-white px-4 py-3 rounded-lg shadow-lg text-sm flex items-center justify-between border border-rose-500 animate-fade-in">
          <span>⚠️ {errorToast}</span>
          <button onClick={() => setErrorToast(null)} className="font-bold ml-2">
            ✕
          </button>
        </div>
      )}

      <h2 className="text-xl font-bold text-slate-100 mb-2 flex items-center gap-2">
        <span>🌱</span> Crop Vision Ingestion
      </h2>
      <p className="text-slate-400 text-sm mb-6">
        Upload or capture a high-resolution photo of the affected plant leaf for ONNX Vision Transformer analysis.
      </p>

      {/* Drag & Drop Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        className={`relative flex flex-col items-center justify-center w-full h-72 border-2 border-dashed rounded-xl cursor-pointer transition-all overflow-hidden ${
          isDragging
            ? "border-emerald-500 bg-emerald-950/20"
            : "border-slate-700 hover:border-slate-500 bg-slate-950/50"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])}
          className="hidden"
          disabled={isLoading}
        />

        {isLoading ? (
          /* Loading Skeleton during inference */
          <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center p-6 z-10">
            <div className="w-16 h-16 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mb-4" />
            <p className="text-emerald-400 font-semibold animate-pulse text-sm">
              Running ONNX Vision Inference...
            </p>
            <p className="text-slate-500 text-xs mt-1">Analyzing pathology & extracting confidence metrics</p>
          </div>
        ) : null}

        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Leaf Upload Preview"
            className="w-full h-full object-cover rounded-lg"
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center">
            <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center text-3xl mb-3 text-slate-300">
              📸
            </div>
            <p className="text-sm font-medium text-slate-200">
              Drag & drop your crop photo here, or <span className="text-emerald-400 underline">browse</span>
            </p>
            <p className="text-xs text-slate-500 mt-2">
              Supports JPEG, PNG, WebP (Max 10MB)
            </p>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-3 mt-6">
        {previewUrl && (
          <button
            type="button"
            onClick={() => { setFile(null); setPreviewUrl(null); }}
            disabled={isLoading}
            className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-sm font-medium transition disabled:opacity-50"
          >
            Clear
          </button>
        )}
        <button
          type="button"
          onClick={handleUploadAndPredict}
          disabled={!file || isLoading}
          className="flex-1 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold rounded-xl text-sm shadow-lg shadow-emerald-900/20 transition-all flex items-center justify-center gap-2"
        >
          {isLoading ? "Analyzing..." : "Diagnose Plant"}
        </button>
      </div>
    </div>
  );
}
