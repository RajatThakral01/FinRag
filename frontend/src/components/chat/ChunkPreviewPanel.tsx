import React, { useEffect, useState } from 'react';
import { X, FileText, AlertCircle, RefreshCw } from 'lucide-react';
import { api } from '../../api/client';
import type { ChunkDetailResponse } from '../../api/types';
import styles from './ChunkPreviewPanel.module.css';

interface ChunkPreviewPanelProps {
  chunkId: string | null;
  onClose: () => void;
}

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export const ChunkPreviewPanel: React.FC<ChunkPreviewPanelProps> = ({
  chunkId,
  onClose,
}) => {
  const [data, setData] = useState<ChunkDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchChunk = async (id: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.getChunk(id);
      setData(res);
    } catch (err: any) {
      setError(err?.message || 'Could not load this excerpt');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (chunkId) {
      fetchChunk(chunkId);
    } else {
      setData(null);
      setError(null);
    }
  }, [chunkId]);

  // Close on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && chunkId) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [chunkId, onClose]);

  if (!chunkId) return null;

  const meta = data?.metadata || {};
  const isTable = meta.chunk_type?.toUpperCase() === 'TABLE';

  return (
    <aside className={styles.panel} aria-label="Chunk Preview Panel">
      <header className={styles.header}>
        <div className={styles.topRow}>
          <div className={styles.titleGroup}>
            <FileText size={16} style={{ color: 'var(--accent-sole)' }} />
            <span className={styles.headerTitle}>SEC 10-K Document Excerpt</span>
          </div>
          <button
            className={styles.closeBtn}
            onClick={onClose}
            title="Close Preview Panel (Esc)"
            aria-label="Close Preview Panel"
          >
            <X size={15} />
          </button>
        </div>

        <div className={styles.badgeRow}>
          {meta.company && (
            <span className={styles.companyBadge}>
              {meta.company}
              {meta.ticker && <span className={styles.tickerTag}>({meta.ticker})</span>}
            </span>
          )}
          {meta.year && <span className={styles.tag}>FY{meta.year}</span>}
          {meta.chunk_type && (
            <span className={`${styles.tag} ${styles.accentTag}`}>
              [{meta.chunk_type.toUpperCase()}]
            </span>
          )}
        </div>

        <div className={styles.chunkId}>
          ID: {chunkId}
        </div>
      </header>

      <div className={styles.body}>
        {isLoading ? (
          <div className={styles.loadingContainer}>
            <div className={styles.spinner} />
            <span>Fetching chunk excerpt from Chroma store...</span>
          </div>
        ) : error ? (
          <div className={styles.errorContainer}>
            <div className={styles.errorTitle}>
              <AlertCircle size={16} />
              <span>Could not load this excerpt</span>
            </div>
            <p className={styles.errorMessage}>{error}</p>
            <button
              className={styles.retryBtn}
              onClick={() => fetchChunk(chunkId)}
            >
              <RefreshCw size={12} style={{ display: 'inline', marginRight: '4px' }} />
              Retry Request
            </button>
          </div>
        ) : data ? (
          <>
            {meta.section_name && (
              <div className={styles.sectionHeader}>
                {meta.section_name} {meta.table_name ? `— ${meta.table_name}` : ''}
              </div>
            )}
            {/* Use react-markdown to safely and beautifully render both tables and prose */}
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              className={styles.markdownContent}
            >
              {data.text}
            </ReactMarkdown>
          </>
        ) : null}
      </div>
    </aside>
  );
};
