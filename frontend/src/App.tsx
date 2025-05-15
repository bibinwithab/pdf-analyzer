import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import PDFChat from "./components/PDFChat";
import Flashcards from "./components/Flashcards";
import MCQs from "./components/MCQs";
import { DataProvider } from "./components/DataContext";
import { Send } from "lucide-react";

function App() {
  return (
    <Router>
      <DataProvider>
        <div className="min-h-screen bg-gray-100">
          <Navbar />
          <Routes>
            <Route path="/" element={<PDFChat />} />
            <Route path="/flashcards" element={<Flashcards />} />
            <Route path="/mcqs" element={<MCQs />} />
          </Routes>
          <a
            href="https://forms.gle/YCTi9UzzbuP9e9tr7"
            target="_blank"
            rel="noopener noreferrer"
            className="fixed text-l bottom-10 right-10 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-full shadow-lg transition-transform transform hover:scale-105 z-50 inline-flex"
          >
            Give Us Your Feedback
            <Send className="ml-2 w-6 h-6"/>
          </a>
        </div>
      </DataProvider>
    </Router>
  );
}

export default App;
