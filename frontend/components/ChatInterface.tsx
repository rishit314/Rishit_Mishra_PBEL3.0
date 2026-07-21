"use client";

import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { PredictResponse } from "./ImageUploader";

interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  isStreaming?: boolean;
}

export interface ChatInterfaceProps {
  initialContext?: any;
  diagnosisContext?: any;
  [key: string]: any;
}

export default function ChatInterface({
  diagnosis,
  apiEndpoint = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-populate initial summary when a new diagnosis arrives
  useEffect(() => {
    if (diagnosis) {
      setMessages([
        {
          id: "init_0",
          sender: "assistant",
          text: `### Diagnosis Verified: **${diagnosis.disease_id.replace(/_/g, " ")}**\n\n${diagnosis.initial_summary}\n\n*Feel free to ask for specific organic, cultural, or chemical treatment intervals!*`,
        },
      ]);
    } else {
      setMessages([]);
    }
  }, [diagnosis]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || isStreaming || !diagnosis) return;

    const userText = inputQuery.trim();
    setInputQuery("");

    const userMsgId = "user_" + Date.now();
    const botMsgId = "bot_" + Date.now();

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, sender: "user", text: userText },
      { id: botMsgId, sender: "assistant", text: "", isStreaming: true },
    ]);
    setIsStreaming(true);

    try {
      const response = await fetch(`${apiEndpoint}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: "default_session",
          user_message: userText,
          disease_context: {
            disease_id: diagnosis.disease_id,
            crop_name: diagnosis.crop_name,
            confidence: diagnosis.confidence,
          },
        }),
      });

      if (!response.ok) throw new Error("Failed to connect to RAG backend");

      const reader = response.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let accumulatedText = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          accumulatedText += chunk;

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMsgId ? { ...msg, text: accumulatedText } : msg
            )
          );
        }
      }
    } catch (error: any) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === botMsgId
            ? { ...msg, text: "⚠️ **Error:** Unable to retrieve extension guidance. Please check server logs." }
            : msg
        )
      );
    } finally {
      setIsStreaming(false);
      setMessages((prev) =>
        prev.map((msg) => (msg.id === botMsgId ? { ...msg, isStreaming: false } : msg))
      );
    }
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.85) return "bg-emerald-950 text-emerald-300 border-emerald-700/50";
    if (conf >= 0.70) return "bg-amber-950 text-amber-300 border-amber-700/50";
    return "bg-rose-950 text-rose-300 border-rose-700/50";
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 border border-slate-800 rounded-2xl shadow-xl overflow-hidden">
      {/* Diagnostic Results Header Card */}
      {diagnosis ? (
        <div className="bg-slate-950 p-5 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Target Crop: <strong className="text-slate-200">{diagnosis.crop_name}</strong>
            </span>
            <h3 className="text-xl font-bold text-slate-100 mt-0.5">
              {diagnosis.disease_id.replace(/_/g, " ")}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 rounded-full text-xs font-mono font-bold border ${getConfidenceColor(diagnosis.confidence)}`}>
              {(diagnosis.confidence * 100).toFixed(1)}% Confidence
            </span>
          </div>
        </div>
      ) : (
        <div className="bg-slate-950 p-5 border-b border-slate-800 text-slate-500 text-sm flex items-center justify-between">
          <span>No active diagnosis context</span>
          <span className="text-xs bg-slate-900 px-2.5 py-1 rounded border border-slate-800">Awaiting Ingestion</span>
        </div>
      )}

      {/* Chat Message History */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {!diagnosis && (
          <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 space-y-3">
            <div className="text-4xl">💬</div>
            <p className="text-sm max-w-sm">
              Upload a plant leaf on the left to initialize the conversational RAG agronomy assistant.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.sender === "user" ? "items-end" : "items-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-5 py-4 text-sm leading-relaxed shadow-md ${
                msg.sender === "user"
                  ? "bg-emerald-600 text-white rounded-br-none"
                  : "bg-slate-800/80 text-slate-200 border border-slate-700/70 rounded-bl-none"
              }`}
            >
              {msg.sender === "assistant" ? (
                <div className="prose prose-invert prose-emerald max-w-none text-sm space-y-2">
                  <ReactMarkdown
                    components={{
                      ul: ({ children }) => <ul className="list-disc pl-5 space-y-1 my-2">{children}</ul>,
                      ol: ({ children }) => <ol className="list-decimal pl-5 space-y-1 my-2">{children}</ol>,
                      li: ({ children }) => <li className="text-slate-300">{children}</li>,
                      strong: ({ children }) => <strong className="text-emerald-400 font-semibold">{children}</strong>,
                      h3: ({ children }) => <h3 className="text-base font-bold text-slate-100 border-b border-slate-700 pb-1 mt-3">{children}</h3>,
                    }}
                  >
                    {msg.text || (msg.isStreaming ? "⌛ *Consulting extension databases...*" : "")}
                  </ReactMarkdown>
                </div>
              ) : (
                <p>{msg.text}</p>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Message Input Box */}
      <form onSubmit={handleSendMessage} className="p-4 bg-slate-950 border-t border-slate-800 flex items-center gap-3">
        <input
          type="text"
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          disabled={!diagnosis || isStreaming}
          placeholder={
            diagnosis
              ? "Ask a follow-up (e.g., When should I spray copper fungicide?)"
              : "Diagnose an image to start chatting..."
          }
          className="flex-1 bg-slate-900 border border-slate-800 focus:border-emerald-500 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none transition disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!diagnosis || isStreaming || !inputQuery.trim()}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold px-6 py-3 rounded-xl text-sm transition shadow-lg shadow-emerald-900/20"
        >
          Send
        </button>
      </form>
    </div>
  );
}
