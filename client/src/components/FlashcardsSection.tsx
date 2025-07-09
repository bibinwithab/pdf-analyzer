import React from "react";
import ReactMarkdown from "react-markdown";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight } from "lucide-react";

type Flashcard = { question: string; answer: string };

interface FlashcardsSectionProps {
  flashcards: Flashcard[];
  flashIndex: number;
  flashFlipped: boolean;
  setFlashFlipped: (f: boolean) => void;
  handleFlashPrev: () => void;
  handleFlashNext: () => void;
  topic: string;
  setTopic: (t: string) => void;
  generateFlashcards: () => void;
}

export default function FlashcardsSection({
  flashcards,
  flashIndex,
  flashFlipped,
  setFlashFlipped,
  handleFlashPrev,
  handleFlashNext,
  topic,
  setTopic,
  generateFlashcards,
}: FlashcardsSectionProps) {
  return (
    <div className="bg-[#23272f] rounded-lg shadow p-4 flex flex-col items-center">
      <h2 className="font-semibold mb-2">Flashcards</h2>
      <div className="flex gap-2 mb-2 w-full">
        <input
          className="flex-1 px-3 py-2 rounded border bg-[#181a20] border-neutral-700"
          placeholder="Topic for flashcards"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
        <button
          className="bg-indigo-600 text-white px-3 py-2 rounded hover:bg-indigo-700 transition"
          onClick={generateFlashcards}
          disabled={!topic.trim()}
        >
          Generate
        </button>
      </div>
      {flashcards.length > 0 && (
        <div className="flex flex-col items-center w-full">
          <motion.div
            className={`flip-card mb-4 ${flashFlipped ? "flipped" : ""}`}
            onClick={() => setFlashFlipped((f) => !f)}
            whileTap={{ scale: 0.97, rotate: 1 }}
          >
            <div className="flip-card-inner">
              <div className="flip-card-front flex items-center justify-center">
                <span className="font-medium text-lg text-indigo-300">
                  <ReactMarkdown>
                    {flashcards[flashIndex].question}
                  </ReactMarkdown>
                </span>
              </div>
              <div className="flip-card-back flex items-center justify-center">
                <span className="font-medium text-lg text-green-400">
                  <ReactMarkdown>{flashcards[flashIndex].answer}</ReactMarkdown>
                </span>
              </div>
            </div>
          </motion.div>
          <div className="flex gap-2">
            <button
              className="px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600"
              onClick={handleFlashPrev}
              disabled={flashIndex === 0}
            >
              <ArrowLeft size={16} />
            </button>
            <span className="text-sm text-gray-300">
              {flashIndex + 1} / {flashcards.length}
            </span>
            <button
              className="px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600"
              onClick={handleFlashNext}
              disabled={flashIndex === flashcards.length - 1}
            >
              <ArrowRight size={16} />
            </button>
          </div>
          <div className="text-xs text-gray-400 mt-1">Click card to flip</div>
        </div>
      )}
    </div>
  );
}
