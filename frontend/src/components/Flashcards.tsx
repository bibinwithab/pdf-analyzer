import React, { useState } from 'react';
import axios from 'axios';
import { Search, RotateCw, ChevronLeft, ChevronRight } from 'lucide-react';

interface Flashcard {
  question: string;
  answer: string;
}

export default function Flashcards() {
  const [topic, setTopic] = useState('');
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [loading, setLoading] = useState(false);

  const URL = 'http://127.0.0.1:8000/';

  const handleSearch = async () => {
    if (!topic.trim()) return;
    
    setLoading(true);
    try {
      const response = await axios.post(URL + 'generate-flashcards/', { 
        question: topic,
      });

      if (response.data && Array.isArray(response.data.flashcards)) {
        setFlashcards(response.data.flashcards);
      } else {
        console.error('Invalid flashcard format received');
        setFlashcards([]);
      }
      setCurrentIndex(0);
      setIsFlipped(false);
    } catch (error) {
      console.error('Error fetching flashcards:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (currentIndex < flashcards.length - 1) {
      setCurrentIndex(prev => prev + 1);
      setIsFlipped(false);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
      setIsFlipped(false);
    }
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

        {/* Flashcard Display */}
        {flashcards.length > 0 && (
          <div className="space-y-6">
            <div
              className="bg-white rounded-xl shadow-lg p-8 cursor-pointer min-h-[300px] relative"
              onClick={() => setIsFlipped(!isFlipped)}
              style={{
                transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0)',
                transformStyle: 'preserve-3d',
                transition: 'transform 0.6s',
              }}
            >
              <div
                className="absolute inset-0 backface-hidden p-8 flex items-center justify-center text-center"
                style={{ backfaceVisibility: 'hidden' }}
              >
                <h2 className="text-2xl font-semibold text-gray-800">
                  {flashcards[currentIndex].question}
                </h2>
              </div>
              <div
                className="absolute inset-0 backface-hidden p-8 flex items-center justify-center text-center"
                style={{
                  backfaceVisibility: 'hidden',
                  transform: 'rotateY(180deg)',
                }}
              >
                <p className="text-xl text-gray-700">
                  {flashcards[currentIndex].answer}
                </p>
              </div>
            </div>

            {/* Navigation Controls */}
            <div className="flex justify-between items-center">
              <button
                onClick={handlePrevious}
                disabled={currentIndex === 0}
                className="flex items-center gap-2 px-4 py-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors disabled:opacity-50"
              >
                <ChevronLeft />
                Previous
              </button>
              <span className="text-gray-600">
                {currentIndex + 1} of {flashcards.length}
              </span>
              <button
                onClick={handleNext}
                disabled={currentIndex === flashcards.length - 1}
                className="flex items-center gap-2 px-4 py-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors disabled:opacity-50"
              >
                Next
                <ChevronRight />
              </button>
            </div>
          </div>
        )}

        {/* Empty State */}
        {flashcards.length === 0 && !loading && (
          <div className="text-center text-gray-500 mt-8">
            Enter a topic to generate flashcards
          </div>
        )}
      </div>
    </div>
  );
}
