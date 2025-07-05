// src/components/McqGenerator.tsx
import React, { useState } from 'react';

interface MCQ {
    question: string;
    options: { A: string; B: string; C: string; D: string };
    correct_answer: string;
}

interface McqGeneratorProps {
    indexId: string;
    apiBaseUrl: string;
}

const McqGenerator: React.FC<McqGeneratorProps> = ({ indexId, apiBaseUrl }) => {
    const [topic, setTopic] = useState('');
    const [mcqs, setMcqs] = useState<MCQ[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleGenerateMcqs = async () => {
        if (!topic.trim()) {
            setError("Please enter a topic for MCQs.");
            return;
        }

        setLoading(true);
        setMcqs([]);
        setError(null);

        try {
            const response = await fetch(`${apiBaseUrl}/generate-mcqs/`, {
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
            setMcqs(data.mcqs);
        } catch (err: any) {
            setError(`Failed to generate MCQs: ${err.message}`);
            console.error('Error generating MCQs:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="mcq-generator">
            <h3>Generate MCQs</h3>
            <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Topic for MCQs (e.g., 'Chapter 1 Concepts')"
                disabled={loading}
            />
            <button onClick={handleGenerateMcqs} disabled={loading || !topic.trim()}>
                {loading ? 'Generating...' : 'Generate MCQs'}
            </button>
            {error && <p className="error-message">{error}</p>}
            {mcqs.length > 0 && (
                <div className="mcq-list">
                    {mcqs.map((mcq, index) => (
                        <div key={index} className="mcq-item">
                            <h4>{index + 1}. {mcq.question}</h4>
                            <ul>
                                {Object.entries(mcq.options).map(([key, value]) => (
                                    <li key={key}>
                                        <strong>{key}:</strong> {value}
                                    </li>
                                ))}
                            </ul>
                            <p className="correct-answer">Correct Answer: <strong>{mcq.correct_answer}</strong></p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default McqGenerator;