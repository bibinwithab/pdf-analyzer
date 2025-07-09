import React from "react";
import ReactMarkdown from "react-markdown";
import { motion } from "framer-motion";
import { RotateCcw } from "lucide-react";

interface ChatSectionProps {
  chat: { type: "user" | "bot"; text: string }[];
  question: string;
  setQuestion: (q: string) => void;
  askQuestion: () => void;
  asking: boolean;
}

export default function ChatSection({
  chat,
  question,
  setQuestion,
  askQuestion,
  asking,
}: ChatSectionProps) {
  return (
    <section className="mb-8">
      <div className="bg-[#23272f] rounded-lg shadow p-4 mb-2">
        <div className="h-64 overflow-y-auto space-y-2 mb-2">
          {chat.length === 0 && (
            <div className="text-gray-500 text-center pt-16">
              Ask a question about your PDF!
            </div>
          )}
          {chat.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex ${
                msg.type === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`px-4 py-2 rounded-lg max-w-[80%] ${
                  msg.type === "user"
                    ? "bg-indigo-600 text-white"
                    : "bg-[#181a20] text-gray-100"
                }`}
              >
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            </motion.div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 px-3 py-2 rounded border bg-[#181a20] border-neutral-700"
            placeholder="Ask a question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && askQuestion()}
            disabled={asking}
          />
          <button
            className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition"
            onClick={askQuestion}
            disabled={asking || !question.trim()}
          >
            {asking ? <RotateCcw className="animate-spin" size={18} /> : "Ask"}
          </button>
        </div>
      </div>
    </section>
  );
}
