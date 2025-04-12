import React, { useState } from 'react';
import axios from 'axios';

export default function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      await axios.post('http://localhost:8000/upload-pdf/', formData);
      alert('PDF uploaded and processed!');
    } catch (err) {
      alert('Failed to upload PDF.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAsk = async () => {
    if (!question) return;
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('question', question);
      const response = await axios.post('http://localhost:8000/ask-question/', formData);
      setAnswer(response.data.answer);
    } catch (err) {
      alert('Failed to get answer.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6 bg-gray-100 text-center">
      <h1 className="text-3xl font-bold mb-6">PDF Q&A App</h1>

      <div className="mb-4">
        <input type="file" accept="application/pdf" onChange={handleFileChange} />
        <button
          className="ml-4 px-4 py-2 bg-blue-500 text-white rounded"
          onClick={handleUpload}
        >
          Upload PDF
        </button>
      </div>

      <div className="mb-4">
        <input
          type="text"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="px-3 py-2 border rounded w-1/2"
        />
        <button
          className="ml-4 px-4 py-2 bg-green-600 text-white rounded"
          onClick={handleAsk}
        >
          Ask
        </button>
      </div>

      {isLoading && <p>Loading...</p>}

      {answer && (
        <div className="mt-6 p-4 bg-white rounded shadow-md">
          <h2 className="text-xl font-semibold mb-2">Answer:</h2>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}
