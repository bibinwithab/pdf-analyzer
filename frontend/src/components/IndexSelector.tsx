// src/components/IndexSelector.tsx
import React, { useState } from 'react';

// Define the type for an index item as received from the backend
interface IndexItem {
    id: string;
    name: string;
}

interface IndexSelectorProps {
    indexes: IndexItem[]; // Now an array of objects
    currentIndexId: string | null;
    onSelectIndex: (indexId: string) => void;
    onDeleteIndex: () => void; // Callback to refresh indexes after deletion
    apiBaseUrl: string;
}

const IndexSelector: React.FC<IndexSelectorProps> = ({
    indexes,
    currentIndexId,
    onSelectIndex,
    onDeleteIndex,
    apiBaseUrl
}) => {
    const [deletingId, setDeletingId] = useState<string | null>(null);
    const [deleteError, setDeleteError] = useState<string | null>(null);
    const [deleteMessage, setDeleteMessage] = useState<string | null>(null);

    const handleDelete = async (indexId: string) => {
        if (!window.confirm(`Are you sure you want to delete the PDF "${indexes.find(i => i.id === indexId)?.name}"?`)) {
            return;
        }

        setDeletingId(indexId);
        setDeleteError(null);
        setDeleteMessage(null);

        try {
            const response = await fetch(`${apiBaseUrl}/delete-index/${indexId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setDeleteMessage(data.message);
            onSelectIndex(null); // Deselect the current index if it was deleted
            onDeleteIndex(); // Refresh the list of indexes
        } catch (err: any) {
            setDeleteError(`Failed to delete PDF: ${err.message}`);
            console.error('Error deleting PDF:', err);
        } finally {
            setDeletingId(null);
        }
    };

    return (
        <div className="index-selector">
            <label htmlFor="index-select">Choose an existing PDF:</label>
            <select
                id="index-select"
                value={currentIndexId || ""}
                onChange={(e) => onSelectIndex(e.target.value)}
            >
                <option value="">--Select an PDF--</option>
                {indexes.map((index) => (
                    <option key={index.id} value={index.id}>
                        {index.name}
                    </option> // Display name, but use ID as value
                ))}
            </select>
            {currentIndexId && (
                <p>
                    Current active PDF: <strong>{indexes.find(i => i.id === currentIndexId)?.name || currentIndexId}</strong>
                    <button
                        onClick={() => handleDelete(currentIndexId)}
                        disabled={deletingId === currentIndexId}
                        style={{ marginLeft: '10px', backgroundColor: '#dc3545' }}
                    >
                        {deletingId === currentIndexId ? 'Deleting...' : 'Delete PDF'}
                    </button>
                </p>
            )}
            {deleteMessage && <p className="success-message">{deleteMessage}</p>}
            {deleteError && <p className="error-message">{deleteError}</p>}
            {indexes.length === 0 && <p>No PDFs available. Please upload a PDF.</p>}
        </div>
    );
};

export default IndexSelector;