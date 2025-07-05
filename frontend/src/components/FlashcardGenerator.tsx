// src/components/FlashcardGenerator.tsx
import React, { useState } from 'react';

interface Flashcard {
    question: string;
    answer: string;
}

interface FlashcardGeneratorProps {
    indexId: string;
    apiBaseUrl: string;
}

const FlashcardGenerator: React.FC<FlashcardGeneratorProps> = ({ indexId, apiBaseUrl }) => {
    const [topic, setTopic] = useState('');
    const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleGenerateFlashcards = async () => {
        if (!topic.trim()) {
            setError("Please enter a topic for flashcards.");
            return;
        }

        setLoading(true);
        setFlashcards([]);
        setError(null);

        try {
            const response = await fetch(`${apiBaseUrl}/generate-flashcards/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: topic, index_id: indexId }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(`Backend error: ${data.error}. Raw: ${data.raw_response}`);
            }
            setFlashcards(data.flashcards);
        } catch (err: any) {
            setError(`Failed to generate flashcards: ${err.message}`);
            console.error('Error generating flashcards:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flashcard-generator">
            <h3>Generate Flashcards</h3>
            <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Topic for flashcards (e.g., 'Introduction to AI')"
                disabled={loading}
            />
            <button onClick={handleGenerateFlashcards} disabled={loading || !topic.trim()}>
                {loading ? 'Generating...' : 'Generate Flashcards'}
            </button>
            {error && <p className="error-message">{error}</p>}
            {flashcards.length > 0 && (
                <div className="flashcard-list">
                    {flashcards.map((card, index) => (
                        <div key={index} className="flashcard">
                            <h4>Q: {card.question}</h4>
                            <p>A: {card.answer}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default FlashcardGenerator;