import React, { createContext, useState, useContext } from 'react';

interface Flashcard {
  question: string;
  answer: string;
}

interface MCQ {
  question: string;
  options: {
    A: string;
    B: string;
    C: string;
    D: string;
  };
  correct_answer: string;
}

interface Message {
  type: 'user' | 'bot';
  content: string;
}

interface DataContextType {
  flashcards: Flashcard[];
  setFlashcards: React.Dispatch<React.SetStateAction<Flashcard[]>>;
  mcqs: MCQ[];
  setMcqs: React.Dispatch<React.SetStateAction<MCQ[]>>;
  
  file: File | null;
  setFile: React.Dispatch<React.SetStateAction<File | null>>;

  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;

  loading: boolean;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const DataProvider: React.FC = ({ children }) => {
  // State for Flashcards
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  
  // State for MCQs
  const [mcqs, setMcqs] = useState<MCQ[]>([]);

  // State for PDF Chat
  const [file, setFile] = useState<File | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  return (
    <DataContext.Provider
      value={{
        flashcards,
        setFlashcards,
        mcqs,
        setMcqs,
        file,
        setFile,
        messages,
        setMessages,
        loading,
        setLoading,
      }}
    >
      {children}
    </DataContext.Provider>
  );
};

export const useData = (): DataContextType => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
};
