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
  apiEndpoint = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
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
      const response = await fetch('${apiEndpoint}/predict', {
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
    <div
      className="relative flex flex-col w-full rounded-sm p-6"
      style={{ background: "#DFE8E1", border: "1px solid #B7C6BC" }}
    >
      <style>{`
        @keyframes avfScanSweep {
          0% { top: -8%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 104%; opacity: 0; }
        }
        .avf-uploader-scanline {
          position: absolute;
          left: 0; right: 0;
          height: 2px;
          background: linear-gradient(90deg, transparent, #6FBFA0, transparent);
          box-shadow: 0 0 12px 2px #6FBFA0;
          animation: avfScanSweep 1.8s ease-in-out infinite;
        }
      `}</style>

      {/* Error Toast Notification */}
      {errorToast && (
        <div
          className="absolute top-4 left-4 right-4 z-50 px-4 py-3 rounded-sm shadow-lg text-sm flex items-center justify-between"
          style={{ background: "rgba(178,58,46,0.92)", color: "#F5EDE9", border: "1px solid #8A2E24" }}
        >
          <span>⚠️ {errorToast}</span>
          <button onClick={() => setErrorToast(null)} className="font-bold ml-2">
            ✕
          </button>
        </div>
      )}

      <h2 className="text-xl font-bold mb-2 flex items-center gap-2" style={{ color: "#182420" }}>
        <span>🌱</span> Crop Vision Ingestion
      </h2>
      <p className="text-sm mb-6" style={{ color: "#4A5C50" }}>
        Upload or capture a high-resolution photo of the affected plant leaf for ONNX Vision Transformer analysis.
      </p>

      {/* Drag & Drop Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        className="relative flex flex-col items-center justify-center w-full h-72 border-2 border-dashed rounded-sm cursor-pointer transition-all overflow-hidden"
        style={{
          borderColor: isDragging ? "#2F7A5C" : "#B7C6BC",
          background: isDragging ? "rgba(111,191,160,0.12)" : "rgba(223,232,225,0.5)",
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])}
          className="hidden"
          disabled={isLoading}
        />

        {isLoading ? (
          /* Loading Skeleton during inference */
          <div
            className="absolute inset-0 backdrop-blur-sm flex flex-col items-center justify-center p-6 z-10"
            style={{ background: "rgba(235,241,236,0.85)" }}
          >
            <div
              className="w-16 h-16 border-4 rounded-full animate-spin mb-4"
              style={{ borderColor: "rgba(47,122,92,0.3)", borderTopColor: "#2F7A5C" }}
            />
            <p className="font-semibold animate-pulse text-sm" style={{ color: "#2F7A5C" }}>
              Running ONNX Vision Inference...
            </p>
            <p className="text-xs mt-1" style={{ color: "#4A5C50" }}>
              Analyzing pathology &amp; extracting confidence metrics
            </p>
          </div>
        ) : null}

        {previewUrl ? (
          <>
            <img
              src={previewUrl}
              alt="Leaf Upload Preview"
              className="w-full h-full object-contain rounded-sm"
            />
            {isLoading && <div className="avf-uploader-scanline" />}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center">
            <div
              className="w-16 h-16 rounded-full flex items-center justify-center text-3xl mb-3"
              style={{ background: "#EBF1EC", color: "#182420" }}
            >
              📸
            </div>
            <p className="text-sm font-medium" style={{ color: "#182420" }}>
              Drag &amp; drop your crop photo here, or{" "}
              <span className="underline" style={{ color: "#2F7A5C" }}>browse</span>
            </p>
            <p className="text-xs mt-2" style={{ color: "#4A5C50" }}>
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
            className="px-4 py-2.5 rounded-sm text-sm font-medium transition disabled:opacity-50"
            style={{ background: "#EBF1EC", color: "#182420" }}
          >
            Clear
          </button>
        )}
        <button
          type="button"
          onClick={handleUploadAndPredict}
          disabled={!file || isLoading}
          className="flex-1 py-3 font-semibold rounded-sm text-sm shadow-lg transition-all flex items-center justify-center gap-2 disabled:cursor-not-allowed"
          style={{
            background: !file || isLoading ? "#B7C6BC" : "#2F7A5C",
            color: !file || isLoading ? "#6B8578" : "#0B1712",
          }}
        >
          {isLoading ? "Analyzing..." : "Diagnose Plant"}
        </button>
      </div>
    </div>
  );
}