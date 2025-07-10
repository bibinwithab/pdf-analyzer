import ReactMarkdown from "react-markdown";
import { motion } from "framer-motion";
import { RotateCcw, Send } from "lucide-react";

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
         <div className="bg-[#23272f] rounded-lg shadow p-1 flex flex-col items-center">
        <h2 className="font-semibold mb-2">Chat with PDF</h2>
         </div>
        <div className="h-128 overflow-y-auto flex flex-col-reverse space-y-2 space-y-reverse mb-2">
          {chat.length === 0 && (
            <div className="text-gray-500 text-center mb-64">
              Ask a question about your PDF!
            </div>
          )}
          {chat
            .slice()
            .reverse()
            .map((msg, i) => (
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
            className="flex-1 px-3 py-2 rounded border bg-[#181a20] border-neutral-700 focus:outline-none focus:border-indigo-500"
            placeholder="Ask a question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && askQuestion()}
            disabled={asking}
          />
          <button
            className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition flex items-center justify-center"
            onClick={askQuestion}
            disabled={asking || !question.trim()}
          >
            {asking ? (
              <RotateCcw className="animate-spin" size={18} />
            ) : (
              <Send className="h-6 w-6" />
            )}
          </button>
        </div>
      </div>
    </section>
  );
}
