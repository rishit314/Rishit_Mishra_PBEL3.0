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
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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
      const response = await fetch("${API_URL}/chat", {
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
    <div className="flex-1 flex flex-col min-h-0">
      {/* Message History Window */}
      <div className="flex-1 overflow-y-auto pr-1 mb-4" style={{ maxHeight: "300px" }}>
        {messages.map((msg, index) => (
          <div key={index} className="mb-3 text-sm leading-relaxed font-mono">
            <span style={{ color: msg.sender === "user" ? "#2F7A5C" : "#6FBFA0" }}>
              {msg.sender === "user" ? "you > " : "agronomist > "}
            </span>
            <span style={{ color: msg.sender === "user" ? "#EDF3EE" : "#D8E4D2" }}>
              <ReactMarkdown>
                {msg.text || (isLoading && index === messages.length - 1 ? "💭 Thinking..." : "")}
              </ReactMarkdown>
            </span>
          </div>
        ))}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSend} className="flex items-center gap-2 pt-3" style={{ borderTop: "1px solid #274238" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g., What organic fungicide should I apply?"
          disabled={isLoading}
          className="flex-1 bg-transparent outline-none text-sm font-mono px-2 py-2 rounded-sm disabled:opacity-50"
          style={{ color: "#EDF3EE", border: "1px solid #274238" }}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-5 py-2.5 rounded-sm text-sm font-bold font-mono transition-all disabled:cursor-not-allowed disabled:opacity-30"
          style={{ background: "#2F7A5C", color: "#0B1712" }}
        >
          Send
        </button>
      </form>
    </div>
  );
}

// --- 2. YOUR MAIN PAGE LAYOUT ---
export default function Home() {
  const TAXONOMY = [
  { crop: "Apple", diseases: ["Scab", "Black Rot", "Cedar Rust", "Healthy"] },
  { crop: "Bell Pepper", diseases: ["Bacterial Spot", "Healthy"] },
  { crop: "Cherry", diseases: ["Powdery Mildew", "Healthy"] },
  { crop: "Corn", diseases: ["Cercospora Leaf Spot", "Common Rust", "N. Leaf Blight", "Healthy"] },
  { crop: "Grape", diseases: ["Black Rot", "Esca", "Leaf Blight", "Healthy"] },
  { crop: "Peach", diseases: ["Bacterial Spot", "Healthy"] },
  { crop: "Potato", diseases: ["Early Blight", "Late Blight", "Healthy"] },
  { crop: "Strawberry", diseases: ["Leaf Scorch", "Healthy"] },
  { crop: "Tomato", diseases: ["Bacterial Spot", "Early Blight", "Late Blight", "Septoria Leaf Spot", "Yellow Leaf Curl", "Healthy"] },
];
  const [diagnosis, setDiagnosis] = useState<PredictResponse | null>(null);

  const handlePredictionResult = (result: PredictResponse) => {
    console.log("✅ Diagnosis received from backend:", result);
    setDiagnosis(result);
  };

  return (
    <main className="min-h-screen px-6 md:px-10 py-8" style={{ background: "#DFE8E1", color: "#182420" }}>
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="pb-6">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold tracking-wide">🌱 AgroVision AI</span>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
            <span
              className="text-xs uppercase font-mono font-medium"
              style={{ color: "#2F7A5C", letterSpacing: "0.12em" }}
            >
              Two-Brain Diagnostics
            </span>
            <span style={{ color: "#B7C6BC" }}>—</span>
            <span className="text-sm opacity-70">
              visual classification, then agronomy consult, in one pass
            </span>
          </div>
          <div className="mt-5 h-px w-full" style={{ background: "#B7C6BC" }} />
        </header>

        {/* Main split */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* LEFT: Specimen Intake */}
          <section
            className="rounded-sm p-5 md:p-6"
            style={{ background: "#EBF1EC", border: "1px solid #B7C6BC" }}
          >
            <div className="flex items-center gap-2 mb-4">
              <span
                className="text-xs uppercase font-semibold font-mono"
                style={{ letterSpacing: "0.14em" }}
              >
                Specimen Intake
              </span>
            </div>

            <ImageUploader onPredict={handlePredictionResult} />

            {diagnosis && (
              <div className="mt-4 p-4 rounded-sm" style={{ border: "1px solid #B7C6BC", background: "#DFE8E1" }}>
                <div className="flex items-center justify-between">
                  <div>
                    <div
                      className="text-xs uppercase font-mono opacity-55"
                      style={{ letterSpacing: "0.1em" }}
                    >
                      {diagnosis.crop_name}
                    </div>
                    <div className="text-lg font-bold">{diagnosis.disease_id}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-semibold font-mono" style={{ color: "#2F7A5C" }}>
                      {(diagnosis.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs opacity-55">confidence</div>
                  </div>
                </div>
                <div className="mt-3 h-1.5 w-full rounded-full" style={{ background: "#B7C6BC" }}>
                  <div
                    className="h-1.5 rounded-full"
                    style={{ width: `${diagnosis.confidence * 100}%`, background: "#6FBFA0" }}
                  />
                </div>

                <h4 className="mt-4 text-sm font-semibold font-mono opacity-70">
                  📋 Initial Agronomy RAG Summary
                </h4>
                <p
                  className="mt-2 text-sm leading-relaxed p-3 rounded-sm"
                  style={{ background: "rgba(11,23,18,0.05)", border: "1px solid #B7C6BC" }}
                >
                  {diagnosis.initial_summary}
                </p>
              </div>
            )}
          </section>

          {/* RIGHT: Field Consult Log */}
          <section
            className="rounded-sm p-5 md:p-6 flex flex-col"
            style={{ background: "#0F211B", border: "1px solid #274238", minHeight: "420px" }}
          >
            <div className="flex items-center gap-2 mb-4">
              <span
                className="text-xs uppercase font-semibold font-mono"
                style={{ color: "#6FBFA0", letterSpacing: "0.14em" }}
              >
                Field Consult Log
              </span>
            </div>

            {diagnosis ? (
              <AgronomyChatBox diagnosis={diagnosis} />
            ) : (
              <p className="text-sm font-mono opacity-60" style={{ color: "#9AAE93" }}>
                &gt; awaiting diagnosis — scan a specimen to open a consult
              </p>
            )}
          </section>
        </div>
      </div>
      {/* Taxonomy strip */}
      <footer className="mt-8">
        <div className="flex items-center gap-2 mb-3">
          <span
            className="text-xs uppercase font-mono"
            style={{ color: "#182420", opacity: 0.55, letterSpacing: "0.12em" }}
          >
            Supported Specimens — 29 Classes
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {TAXONOMY.map((t) => (
            <div
              key={t.crop}
              className="rounded-sm px-3 py-1.5"
              style={{ background: "#EBF1EC", border: "1px solid #B7C6BC" }}
            >
              <span className="text-xs font-medium" style={{ color: "#182420" }}>
                {t.crop}
              </span>
              <span className="text-xs font-mono opacity-50" style={{ color: "#182420" }}>
                {" "}
                · {t.diseases.length}
              </span>
            </div>
          ))}
        </div>
      </footer>
    </main>
  );
}