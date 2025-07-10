import React, { useRef, useState, useEffect } from "react";
import Header from "./components/Header";
import PdfSelector from "./components/PdfSelector";
import ChatSection from "./components/ChatSection";
import FlashcardsSection from "./components/FlashcardsSection";
import MCQSection from "./components/MCQSection";
import "./App.css";

const API_BASE_URL = "http://localhost:8000";

type IndexItem = { id: string; name: string };
type Flashcard = { question: string; answer: string };
type MCQ = {
  question: string;
  options: Record<string, string>;
  correct_answer: string;
};

export default function App() {
  // Indexes and selection
  const [indexes, setIndexes] = useState<IndexItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState<string | null>(null);
  const [loadingIndexes, setLoadingIndexes] = useState(false);

  // Upload
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  // Chat
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState<{ type: "user" | "bot"; text: string }[]>(
    []
  );
  const [asking, setAsking] = useState(false);

  // Flashcards/MCQs
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [flashIndex, setFlashIndex] = useState(0);
  const [flashFlipped, setFlashFlipped] = useState(false);

  const [mcqs, setMcqs] = useState<MCQ[]>([]);
  const [mcqIndex, setMcqIndex] = useState(0);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [showAnswer, setShowAnswer] = useState(false);

  const [topic, setTopic] = useState("");

  // Fetch indexes
  useEffect(() => {
    setLoadingIndexes(true);
    fetch(`${API_BASE_URL}/list-indexes/`)
      .then((r) => r.json())
      .then((data) => setIndexes(data.indexes || []))
      .finally(() => setLoadingIndexes(false));
  }, []);

  // Upload PDF
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    setUploading(true);
    const form = new FormData();
    form.append("file", e.target.files[0]);
    const res = await fetch(`${API_BASE_URL}/upload-pdf/`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    setUploading(false);
    if (data.index_id) {
      setIndexes((prev) => [
        ...prev,
        { id: data.index_id, name: data.file_name },
      ]);
      setCurrentIndex(data.index_id);
    }
  };

  // Ask question
  const askQuestion = async () => {
    if (!question.trim() || !currentIndex) return;
    setChat((c) => [...c, { type: "user", text: question }]);
    setAsking(true);
    setQuestion("");
    const res = await fetch(`${API_BASE_URL}/ask-question/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, index_id: currentIndex }),
    });
    const data = await res.json();
    setChat((c) => [
      ...c,
      {
        type: "bot",
        text: data.answer?.content || data.answer || "No answer.",
      },
    ]);
    setAsking(false);
  };

  // Generate flashcards
  const generateFlashcards = async () => {
    if (!topic.trim() || !currentIndex) return;
    setFlashcards([]);
    setFlashIndex(0);
    setFlashFlipped(false);
    const res = await fetch(`${API_BASE_URL}/generate-flashcards/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: topic, index_id: currentIndex }),
    });
    const data = await res.json();
    setFlashcards(data.flashcards || []);
  };

  // Generate MCQs
  const generateMCQs = async () => {
    if (!topic.trim() || !currentIndex) return;
    setMcqs([]);
    setMcqIndex(0);
    setSelectedOption(null);
    setShowAnswer(false);
    const res = await fetch(`${API_BASE_URL}/generate-mcqs/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: topic, index_id: currentIndex }),
    });
    const data = await res.json();
    setMcqs(data.mcqs || []);
  };

  // Flashcard navigation
  const handleFlashPrev = () => {
    setFlashIndex((i) => (i > 0 ? i - 1 : i));
    setFlashFlipped(false);
  };
  const handleFlashNext = () => {
    setFlashIndex((i) => (i < flashcards.length - 1 ? i + 1 : i));
    setFlashFlipped(false);
  };

  // MCQ navigation
  const handleMcqPrev = () => {
    setMcqIndex((i) => (i > 0 ? i - 1 : i));
    setSelectedOption(null);
    setShowAnswer(false);
  };
  const handleMcqNext = () => {
    setMcqIndex((i) => (i < mcqs.length - 1 ? i + 1 : i));
    setSelectedOption(null);
    setShowAnswer(false);
  };

  // MCQ option select
  const handleOptionSelect = (opt: string) => {
    if (showAnswer) return;
    setSelectedOption(opt);
    setShowAnswer(true);
  };

  // Delete index
  const handleDeleteIndex = async (id: string) => {
    if (!window.confirm("Delete this PDF?")) return;
    await fetch(`${API_BASE_URL}/delete-index/${id}`, { method: "DELETE" });
    setIndexes((prev) => prev.filter((idx) => idx.id !== id));
    if (currentIndex === id) setCurrentIndex(null);
  };

  return (
    <div className="min-h-screen bg-[#181a20] text-gray-100">
      <Header />
      <main className="max-w-3xl mx-auto py-8 px-4">
        <PdfSelector
          indexes={indexes}
          currentIndex={currentIndex}
          setCurrentIndex={setCurrentIndex}
          handleDeleteIndex={handleDeleteIndex}
          fileInput={fileInput}
          handleUpload={handleUpload}
          uploading={uploading}
          loadingIndexes={loadingIndexes}
        />
        {currentIndex && (
          <>
            <ChatSection
              chat={chat}
              question={question}
              setQuestion={setQuestion}
              askQuestion={askQuestion}
              asking={asking}
            />
            <section className="grid md:grid-cols-2 gap-6">
              <FlashcardsSection
                flashcards={flashcards}
                flashIndex={flashIndex}
                flashFlipped={flashFlipped}
                setFlashFlipped={setFlashFlipped}
                handleFlashPrev={handleFlashPrev}
                handleFlashNext={handleFlashNext}
                topic={topic}
                setTopic={setTopic}
                generateFlashcards={generateFlashcards}
              />
              <MCQSection
                mcqs={mcqs}
                mcqIndex={mcqIndex}
                selectedOption={selectedOption}
                showAnswer={showAnswer}
                handleOptionSelect={handleOptionSelect}
                handleMcqPrev={handleMcqPrev}
                handleMcqNext={handleMcqNext}
                topic={topic}
                setTopic={setTopic}
                generateMCQs={generateMCQs}
              />
            </section>
          </>
        )}
      </main>
    </div>
  );
}
