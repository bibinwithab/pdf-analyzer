// src/components/QuestionInput.tsx
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

interface QuestionInputProps {
    indexId: string;
    apiBaseUrl: string;
}

const QuestionInput: React.FC<QuestionInputProps> = ({ indexId, apiBaseUrl }) => {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        if (!question.trim()) {
            setError("Please enter a question.");
            return;
        }

        setLoading(true);
        setAnswer(null);
        setError(null);

        try {
            const response = await fetch(`${apiBaseUrl}/ask-question/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question, index_id: indexId }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setAnswer(data.answer);
        } catch (err: any) {
            setError(`Failed to get answer: ${err.message}`);
            console.error('Error asking question:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="question-input">
            <form onSubmit={handleSubmit}>
                <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question about the PDF..."
                    rows={4}
                    disabled={loading}
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Thinking...' : 'Ask'}
                </button>
            </form>
            {error && <p className="error-message">{error}</p>}
            {answer && (
                <div className="answer-section">
                    <h3>Answer:</h3>
                    <ReactMarkdown>{answer}</ReactMarkdown>
                </div>
            )}
        </div>
    );
};

export default QuestionInput;