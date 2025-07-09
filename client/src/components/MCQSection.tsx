import React from "react";
import ReactMarkdown from "react-markdown";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight } from "lucide-react";

type MCQ = {
  question: string;
  options: Record<string, string>;
  correct_answer: string;
};

interface MCQSectionProps {
  mcqs: MCQ[];
  mcqIndex: number;
  selectedOption: string | null;
  showAnswer: boolean;
  handleOptionSelect: (opt: string) => void;
  handleMcqPrev: () => void;
  handleMcqNext: () => void;
  topic: string;
  setTopic: (t: string) => void;
  generateMCQs: () => void;
}

export default function MCQSection({
  mcqs,
  mcqIndex,
  selectedOption,
  showAnswer,
  handleOptionSelect,
  handleMcqPrev,
  handleMcqNext,
  topic,
  setTopic,
  generateMCQs,
}: MCQSectionProps) {
  return (
    <div className="bg-[#23272f] rounded-lg shadow p-4 flex flex-col items-center">
      <h2 className="font-semibold mb-2">MCQs</h2>
      <div className="flex gap-2 mb-2 w-full">
        <input
          className="flex-1 px-3 py-2 rounded border bg-[#181a20] border-neutral-700"
          placeholder="Topic for MCQs"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
        <button
          className="bg-indigo-600 text-white px-3 py-2 rounded hover:bg-indigo-700 transition"
          onClick={generateMCQs}
          disabled={!topic.trim()}
        >
          Generate
        </button>
      </div>
      {mcqs.length > 0 && (
        <div className="flex flex-col items-center w-full">
          <div className="w-full mb-4">
            <div className="font-medium text-lg text-indigo-300 mb-2">
              <ReactMarkdown>{mcqs[mcqIndex].question}</ReactMarkdown>
            </div>
            <ul className="space-y-2">
              {Object.entries(mcqs[mcqIndex].options).map(([key, value]) => {
                let optionClass =
                  "border px-3 py-2 rounded cursor-pointer transition flex items-center";
                if (showAnswer) {
                  if (key === mcqs[mcqIndex].correct_answer) {
                    optionClass +=
                      " border-green-600 bg-green-900 text-green-300 font-semibold";
                  } else if (selectedOption === key) {
                    optionClass +=
                      " border-red-600 bg-red-900 text-red-300 font-semibold";
                  } else {
                    optionClass += " border-neutral-700 bg-[#23272f]";
                  }
                } else {
                  optionClass +=
                    " border-neutral-700 bg-[#23272f] hover:bg-neutral-700";
                }
                return (
                  <motion.li
                    key={key}
                    className={optionClass}
                    onClick={() => handleOptionSelect(key)}
                    whileTap={{ scale: 0.97 }}
                  >
                    <span className="font-semibold mr-2">{key}.</span>
                    <ReactMarkdown>{value}</ReactMarkdown>
                    {showAnswer && key === mcqs[mcqIndex].correct_answer && (
                      <span className="ml-2 text-green-400">
                        <ArrowRight size={16} />
                      </span>
                    )}
                    {showAnswer &&
                      selectedOption === key &&
                      selectedOption !== mcqs[mcqIndex].correct_answer && (
                        <span className="ml-2 text-red-400">
                          <ArrowRight size={16} />
                        </span>
                      )}
                  </motion.li>
                );
              })}
            </ul>
            {showAnswer && (
              <div className="mt-2 text-sm">
                {selectedOption === mcqs[mcqIndex].correct_answer ? (
                  <span className="text-green-400 font-semibold">Correct!</span>
                ) : (
                  <span className="text-red-400 font-semibold">
                    Incorrect. Correct answer: {mcqs[mcqIndex].correct_answer}
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <button
              className="px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600"
              onClick={handleMcqPrev}
              disabled={mcqIndex === 0}
            >
              <ArrowLeft size={16} />
            </button>
            <span className="text-sm text-gray-300">
              {mcqIndex + 1} / {mcqs.length}
            </span>
            <button
              className="px-3 py-1 rounded bg-neutral-700 hover:bg-neutral-600"
              onClick={handleMcqNext}
              disabled={mcqIndex === mcqs.length - 1}
            >
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
