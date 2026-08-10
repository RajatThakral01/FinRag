import React, { useState, useRef, useEffect } from 'react';
import { CornerDownLeft, Terminal } from 'lucide-react';
import { SAMPLE_QUERIES } from '../../constants';
import styles from './ChatInput.module.css';

interface ChatInputProps {
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSubmit, isLoading }) => {
  const [query, setQuery] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [query]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSubmit(query.trim());
    setQuery('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChipClick = (sample: string) => {
    if (isLoading) return;
    onSubmit(sample);
  };

  return (
    <div className={styles.inputWrapper}>
      {/* Sample Query Chips */}
      <div className={styles.chipsRow}>
        {SAMPLE_QUERIES.map((sample, idx) => (
          <button
            key={idx}
            className={styles.chip}
            onClick={() => handleChipClick(sample)}
            disabled={isLoading}
          >
            <Terminal size={11} className={styles.chipIcon} />
            <span>{sample}</span>
          </button>
        ))}
      </div>

      {/* Input Form */}
      <form className={styles.inputForm} onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          placeholder="Enter financial research query (e.g. Apple FY24 Revenue, Tesla risk factors)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          rows={1}
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={!query.trim() || isLoading}
          title="Execute Query (Enter)"
        >
          <CornerDownLeft size={15} />
          <span className={styles.btnText}>EXECUTE</span>
        </button>
      </form>

      <div className={styles.inputFooterText}>
        Filing Data Verified against SEC 10-K Reports • Automatic Follow-up Resolution
      </div>
    </div>
  );
};
