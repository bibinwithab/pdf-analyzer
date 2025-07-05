// src/components/PdfUpload.tsx
import React, { useState } from 'react';

interface PdfUploadProps {
    // Now pass the file name as well
    onIndexCreated: (indexId: string, fileName: string) => void;
    apiBaseUrl: string;
}

const PdfUpload: React.FC<PdfUploadProps> = ({ onIndexCreated, apiBaseUrl }) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
            setMessage(null);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError("Please select a file first.");
            return;
        }

        setUploading(true);
        setMessage(null);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${apiBaseUrl}/upload-pdf/`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setMessage(data.message);
            // Pass both id and name
            onIndexCreated(data.index_id, data.file_name);
            setSelectedFile(null); // Clear the selected file after successful upload
        } catch (err: any) {
            setError(`Upload failed: ${err.message}`);
            console.error('Error uploading PDF:', err);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="pdf-upload">
            <input type="file" accept=".pdf" onChange={handleFileChange} disabled={uploading} />
            <button onClick={handleUpload} disabled={!selectedFile || uploading}>
                {uploading ? 'Uploading...' : 'Upload PDF'}
            </button>
            {message && <p className="success-message">{message}</p>}
            {error && <p className="error-message">{error}</p>}
            {selectedFile && <p>Selected file: {selectedFile.name}</p>}
        </div>
    );
};

export default PdfUpload;