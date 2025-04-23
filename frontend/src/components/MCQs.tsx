import React, { useState } from 'react';
import axios from 'axios';
import { Search, RotateCw } from 'lucide-react';
import { useData } from './DataContext';


export default function MCQs() {
    const { mcqs, setMcqs } = useData(); 
  const [topic, setTopic] = useState('');
//   const [mcqs, setMcqs] = useState<MCQ[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [loading, setLoading] = useState(false);

  const URL = 'http://127.0.0.1:8000/';

  const handleSearch = async () => {
    if (!topic.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post(URL + 'generate-mcqs/', {
        question: topic,
      });

      if (response.data && Array.isArray(response.data.mcqs)) {
        setMcqs(response.data.mcqs);
      } else {
        console.error('Invalid MCQ format received');
        setMcqs([]);
      }
      setCurrentIndex(0);
      setShowAnswer(false);
    } catch (error) {
      console.error('Error fetching MCQs:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOptionClick = (optionKey: string) => {
    setShowAnswer(true);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-3xl mx-auto">
        {/* Search Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex gap-4">
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Enter a topic..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button
              onClick={handleSearch}
              disabled={loading}
              className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {loading ? <RotateCw className="animate-spin" /> : <Search />}
              Search
            </button>
          </div>
        </div>

        {/* MCQ Display */}
        {mcqs.length > 0 && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-8 space-y-4">
              <h2 className="text-2xl font-semibold text-gray-800">
                {mcqs[currentIndex].question}
              </h2>
              <div className="space-y-3">
                {Object.entries(mcqs[currentIndex].options).map(
                  ([key, value]) => (
                    <button
                      key={key}
                      onClick={() => handleOptionClick(key)}
                      className={`block w-full text-left px-4 py-2 rounded-lg border ${
                        showAnswer
                          ? key === mcqs[currentIndex].correct_answer
                            ? 'border-green-600 bg-green-100 text-green-800 font-semibold'
                            : 'border-gray-300 bg-white'
                          : 'border-gray-300 bg-white hover:bg-indigo-50'
                      }`}
                    >
                      <strong>{key}.</strong> {value}
                    </button>
                  )
                )}
              </div>
              {showAnswer && (
                <p className="text-sm text-green-600 mt-2">
                  Correct Answer: {mcqs[currentIndex].correct_answer}
                </p>
              )}
            </div>

            {/* Navigation Controls */}
            <div className="flex justify-between items-center">
              <button
                onClick={() =>
                  setCurrentIndex((prev) => Math.max(prev - 1, 0))
                }
                disabled={currentIndex === 0}
                className="px-4 py-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors disabled:opacity-50"
              >
                Previous
              </button>
              <span className="text-gray-600">
                {currentIndex + 1} of {mcqs.length}
              </span>
              <button
                onClick={() => {
                  setCurrentIndex((prev) =>
                    Math.min(prev + 1, mcqs.length - 1)
                  );
                  setShowAnswer(false);
                }}
                disabled={currentIndex === mcqs.length - 1}
                className="px-4 py-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}

        {/* Empty State */}
        {mcqs.length === 0 && !loading && (
          <div className="text-center text-gray-500 mt-8">
            Enter a topic to generate MCQs
          </div>
        )}
      </div>
    </div>
  );
}
