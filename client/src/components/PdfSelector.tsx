import { Trash2, Upload, RotateCcw } from "lucide-react";
import React from "react";

type IndexItem = { id: string; name: string };

interface PdfSelectorProps {
  indexes: IndexItem[];
  currentIndex: string | null;
  setCurrentIndex: (id: string) => void;
  handleDeleteIndex: (id: string) => void;
  fileInput: React.RefObject<HTMLInputElement>;
  handleUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  uploading: boolean;
  loadingIndexes: boolean;
}

export default function PdfSelector({
  indexes,
  currentIndex,
  setCurrentIndex,
  handleDeleteIndex,
  fileInput,
  handleUpload,
  uploading,
  loadingIndexes,
}: PdfSelectorProps) {
  return (
    <section className="mb-8">
      <div className="flex flex-col md:flex-row gap-4 items-center">
        <button
          className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 transition flex items-center gap-2"
          onClick={() => fileInput.current?.click()}
          disabled={uploading}
        >
          <Upload size={18} />
          {uploading ? "Uploading..." : "Upload PDF"}
        </button>
        <input
          type="file"
          accept="application/pdf"
          className="hidden"
          ref={fileInput}
          onChange={handleUpload}
        />
        <div className="flex-1">
          <div className="relative">
            <select
              className="w-full px-3 py-2 rounded border bg-[#23272f] border-neutral-700 pr-10  text-white focus:outline-none focus:border-indigo-500 "
              value={currentIndex || ""}
              onChange={(e) => setCurrentIndex(e.target.value)}
            >
              <option value="">Select PDF</option>
              {indexes.map((idx) => (
                <option key={idx.id} value={idx.id}>
                  {idx.name}
                </option>
              ))}
            </select>
            {currentIndex && (
              <button
                className="absolute right-7 top-1/2 -translate-y-1/2 text-red-400 hover:text-red-600"
                title="Delete PDF"
                onClick={() => handleDeleteIndex(currentIndex)}
              >
                <Trash2 size={18} />
              </button>
            )}
          </div>
        </div>
      </div>
      {loadingIndexes && (
        <div className="text-sm text-gray-400 mt-2 flex items-center gap-2">
          <RotateCcw className="animate-spin" size={16} /> Loading PDFs...
        </div>
      )}
    </section>
  );
}
