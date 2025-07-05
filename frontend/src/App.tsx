// src/App.tsx
import React, { useState, useEffect } from 'react';
import PdfUpload from './components/PdfUpload';
import QuestionInput from './components/QuestionInput';
import FlashcardGenerator from './components/FlashcardGenerator';
import McqGenerator from './components/McqGenerator';
import IndexSelector from './components/IndexSelector';
import './App.css'; // For basic styling

const API_BASE_URL = 'http://localhost:8000'; // Make sure this matches your FastAPI port

// Define the type for an index item
interface IndexItem {
    id: string;
    name: string;
}

function App() {
    const [currentIndexId, setCurrentIndexId] = useState<string | null>(null);
    // Change type to array of IndexItem
    const [indexes, setIndexes] = useState<IndexItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchIndexes();
    }, []);

    const fetchIndexes = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${API_BASE_URL}/list-indexes/`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            // Assuming data.indexes is now an array of { id: string, name: string }
            setIndexes(data.indexes);
            // If there's an existing index_id and it's no longer valid in the new list, clear it
            if (currentIndexId && !data.indexes.some((item: IndexItem) => item.id === currentIndexId)) {
                setCurrentIndexId(null);
            } else if (!currentIndexId && data.indexes.length > 0) {
                // Optionally auto-select the first index if none is selected
                // setCurrentIndexId(data.indexes[0].id);
            }
        } catch (err: any) {
            setError(`Failed to fetch indexes: ${err.message}`);
            console.error('Error fetching indexes:', err);
        } finally {
            setLoading(false);
        }
    };

    // When a new index is created, immediately set it as active and refresh the list
    const handleIndexCreated = (newIndexId: string, newFileName: string) => {
        setCurrentIndexId(newIndexId);
        // Optimistically add the new index to the list
        setIndexes(prev => [...prev, { id: newIndexId, name: newFileName }]);
        // Or re-fetch the entire list for robustness
        // fetchIndexes();
    };

    return (
        <div className="App">
            <h1>PDF Analyzer</h1>

            <section className="upload-section">
                <h2>Upload PDF</h2>
                <PdfUpload onIndexCreated={handleIndexCreated} apiBaseUrl={API_BASE_URL} />
                {loading && <p>Loading indexes...</p>}
                {error && <p className="error">{error}</p>}
            </section>

            <section className="index-selection-section">
                <h2>Select PDF</h2>
                <IndexSelector
                    indexes={indexes}
                    currentIndexId={currentIndexId}
                    onSelectIndex={setCurrentIndexId}
                    onDeleteIndex={fetchIndexes} // Pass fetchIndexes to refresh after deletion
                    apiBaseUrl={API_BASE_URL}
                />
            </section>

            {currentIndexId && (
                <>
                    <section className="chat-section">
                        <h2>Ask Questions</h2>
                        <QuestionInput indexId={currentIndexId} apiBaseUrl={API_BASE_URL} />
                    </section>

                    <section className="tools-section">
                        <h2>Generate Learning Materials</h2>
                        <FlashcardGenerator indexId={currentIndexId} apiBaseUrl={API_BASE_URL} />
                        <McqGenerator indexId={currentIndexId} apiBaseUrl={API_BASE_URL} />
                    </section>
                </>
            )}

            {!currentIndexId && indexes.length > 0 && (
                <p>Please select an existing index to proceed.</p>
            )}
            {!currentIndexId && indexes.length === 0 && (
                <p>Please upload a PDF to create an index and proceed.</p>
            )}
        </div>
    );
}

export default App;