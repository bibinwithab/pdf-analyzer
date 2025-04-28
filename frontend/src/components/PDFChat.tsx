import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Send, Loader2, X} from 'lucide-react';
import { useData } from './DataContext'; 

export default function PDFChat() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { file, setFile, messages, setMessages, loading, setLoading } = useData();

  const URL = 'https://pdf-analyzer-backend-lcz2.onrender.com/'

  // const generateAudioSummary = async () => {
  //   setLoading(true);
  //   setMessages(prev => [...prev, { type: 'bot', content: 'Generating audio summary...' }]);
  //   try {
  //     const response = await axios.post(URL + 'pdf-summary-audio/', {}, { responseType: 'blob' });
  //     const blob = new Blob([response.data], { type: 'audio/mpeg' });
  //     const url = URL.createObjectURL(blob);
  //     setAudioUrl(url);
  //     setMessages(prev => [...prev, { type: 'bot', content: 'Audio summary generated. Click play below.' }]);
  //   } catch (error) {
  //     setMessages(prev => [...prev, { type: 'bot', content: 'Failed to generate audio summary.' }]);
  //     console.error(error);
  //   }
  //   setLoading(false);
  // };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setMessages([{ type: 'bot', content: 'Processing...' }]);
      handleUpload(selectedFile);
    } else {
      setMessages([{ type: 'bot', content: 'Please upload a PDF file.' }]);
    }
  };

  const handleUpload = async (selectedFile: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(URL+'upload-pdf/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setMessages(prev => [...prev, { type: 'bot', content: response.data.message }]);
    } catch (error) {
      setMessages(prev => [...prev, { type: 'bot', content: 'Error processing the PDF. Please try again.' }]);
    } finally {
      setLoading(false);
    }
  };

  const askQuestion = async (question: string) => {
    setLoading(true);
    setMessages(prev => [...prev, { type: 'user', content: question }]);
    try {
      const response = await axios.post(URL+'ask-question/', {
        question,
      });

      const answer = response.data.answer || 'No answer returned.';
      setMessages(prev => [...prev, { type: 'bot', content: answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { type: 'bot', content: 'Error getting answer. Please try again.' }]);
    }
    setLoading(false);
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    setMessages(prev => [...prev, { type: 'bot', content: 'PDF file removed. You can upload a new one.' }]);
  };

  return (
    <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden mt-8">
      {/* Header */}
      <div className="bg-indigo-600 p-4">
        <h1 className="text-2xl font-bold text-white text-center">PDF Chat Assistant</h1>
      </div>

      {/* Current File Display */}
      {file && (
        <div className="bg-indigo-50 p-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Upload size={20} className="text-indigo-600" />
            <span className="text-indigo-900 font-medium">{file.name}</span>
            <span className="text-indigo-600 text-sm">
              ({(file.size / (1024 * 1024)).toFixed(2)} MB)
            </span>
          </div>
          <button
            onClick={handleRemoveFile}
            className="text-indigo-600 hover:text-indigo-800 p-1 rounded-full hover:bg-indigo-100 transition-colors"
            title="Remove PDF"
          >
            <X size={20} />
          </button>
        </div>
      )}

      {/* Chat Messages */}
      <div className="h-[500px] overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.type === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[70%] rounded-lg p-3 ${
                message.type === 'user'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-center">
            <Loader2 className="animate-spin text-indigo-600" />
          </div>
        )}
      </div>

      {/* Upload Section */}
      <div className="p-4 border-t">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".pdf"
          className="hidden"
        />
        <div className="flex pb-4 justify-center">
          <button
            onClick={triggerFileInput}
            disabled={loading}
            className="flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
          >
            <Upload size={20} />
            {file ? 'Upload Another PDF' : 'Upload PDF'}
          </button>
        </div>
        {/* Ask Question Section */}
        <div className="p-4 border-t">
          <input
            type="text"
            placeholder="Ask a question about the PDF..."
            className="w-full p-3 border rounded-lg"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.target as HTMLInputElement).value) {
                askQuestion((e.target as HTMLInputElement).value);
                (e.target as HTMLInputElement).value = '';
              }
            }}
          />
          <div className="flex justify-end mt-2">
            <button
              onClick={() => askQuestion('What is the summary of the PDF?')}
              disabled={loading}
              className="flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50"
            >
              <Send size={20} />
              Ask Question / Click for Summaray
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
