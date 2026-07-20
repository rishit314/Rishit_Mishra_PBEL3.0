"use client";

import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import ImageUploader, { PredictResponse } from "@/components/ImageUploader";

// --- 1. THE INTERACTIVE RAG CHAT COMPONENT ---
interface Message {
  sender: "user" | "ai";
  text: string;
}

function AgronomyChatBox({ diagnosis }: { diagnosis: PredictResponse }) {
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "ai",
      text: `Hello! I am your AI Agronomist. Ask me anything about treating ${diagnosis.disease_id} on your ${diagnosis.crop_name} crops!`,
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    
    // Add user message to screen immediately
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setIsLoading(true);

    try {
      // Connect to your FastAPI /chat streaming endpoint
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_message: userMessage,
          session_id: "default_session",
          disease_context: {
            disease_id: diagnosis.disease_id,
            crop_name: diagnosis.crop_name,
            confidence: diagnosis.confidence,
          },
        }),
      });

      if (!response.body) throw new Error("No streaming stream available");

      // Set up a blank AI message that we will stream tokens into
      setMessages((prev) => [...prev, { sender: "ai", text: "" }]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let aiResponseText = "";

      // Read the token stream chunk-by-chunk
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        aiResponseText += chunk;

        // Dynamically update the last message on screen with new words
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { sender: "ai", text: aiResponseText };
          return updated;
        });
      }
    } catch (error) {
      console.error("Chat streaming failed:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: "⚠️ Error connecting to RAG knowledge base. Please check your backend connection." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="mt-6 p-6 bg-slate-900/90 rounded-2xl border border-emerald-500/30 shadow-2xl animate-fade-in">
      <h3 className="text-lg font-bold text-emerald-400 mb-4 flex items-center gap-2">
        💬 Interactive Agronomy Consultation
      </h3>

      {/* Message History Window */}
      <div className="h-64 overflow-y-auto pr-2 space-y-3 mb-4 scrollbar-thin scrollbar-thumb-slate-700">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.sender === "user"
                  ? "bg-emerald-600 text-white rounded-br-none font-medium"
                  : "bg-slate-950 text-slate-300 border border-slate-800 rounded-bl-none"
              }`}
            >
              <ReactMarkdown>
                {msg.text || (isLoading && index === messages.length - 1 ? "💭 Thinking..." : "")}
              </ReactMarkdown>
            </div>
          </div>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSend} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g., What organic fungicide should I apply?"
          disabled={isLoading}
          className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500 transition-colors disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-800 text-slate-950 font-bold px-5 py-2.5 rounded-xl text-sm transition-all shadow-lg hover:shadow-emerald-500/20 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}

// --- 2. YOUR MAIN PAGE LAYOUT ---
export default function Home() {
  const [diagnosis, setDiagnosis] = useState<PredictResponse | null>(null);

  const handlePredictionResult = (result: PredictResponse) => {
    console.log("✅ Diagnosis received from backend:", result);
    setDiagnosis(result);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-8 flex flex-col items-center">
      <div className="max-w-2xl w-full">
        <h1 className="text-3xl font-extrabold text-emerald-400 mb-6 text-center">
          AgroVision AI Diagnosis
        </h1>

        <ImageUploader onPredict={handlePredictionResult} />

        {/* RENDER DIAGNOSIS & CHAT BOX TOGETHER */}
        {diagnosis && (
          <div className="space-y-6">
            {/* Pathology Overview Card */}
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

            {/* 5. NEW: The Interactive RAG Chatbox renders automatically! */}
            <AgronomyChatBox diagnosis={diagnosis} />
          </div>
        )}
      </div>
    </main>
  );
}