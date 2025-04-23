import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import PDFChat from './components/PDFChat';
import Flashcards from './components/Flashcards';
import MCQs from './components/MCQs';
import { DataProvider } from './components/DataContext'; // Import DataProvider

function App() {
  return (
    <Router>
      <DataProvider> {/* Wrap with DataProvider */}
        <div className="min-h-screen bg-gray-100">
          <Navbar />
          <Routes>
            <Route path="/" element={<PDFChat />} />
            <Route path="/flashcards" element={<Flashcards />} />
            <Route path="/mcqs" element={<MCQs />} />
          </Routes>
        </div>
      </DataProvider>
    </Router>
  );
}

export default App;
