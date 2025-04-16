import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import PDFChat from './components/PDFChat';
import Flashcards from './components/Flashcards';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Navbar />
        <Routes>
          <Route path="/" element={<PDFChat />} />
          <Route path="/flashcards" element={<Flashcards />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;