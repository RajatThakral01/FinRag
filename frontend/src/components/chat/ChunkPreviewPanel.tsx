import React, { useEffect, useState } from 'react';
import { X, FileText, AlertCircle, RefreshCw } from 'lucide-react';
import { api } from '../../api/client';
import type { ChunkDetailResponse } from '../../api/types';
import styles from './ChunkPreviewPanel.module.css';

interface ChunkPreviewPanelProps {
  chunkId: string | null;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Pipe-table parser — converts markdown table syntax into <table> JSX.
// Detects tables by checking if ≥50% of non-empty lines start with '|'.
// ---------------------------------------------------------------------------
function isPipeTable(text: string): boolean {
  const lines = text.split('\n').filter((l) => l.trim().length > 0);
  if (lines.length < 2) return false;
  const pipeLines = lines.filter((l) => l.trim().startsWith('|'));
  return pipeLines.length / lines.length >= 0.5;
}

function parsePipeTable(text: string): React.ReactNode {
  const lines = text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  const rows: string[][] = lines
    .filter((l) => !l.match(/^\|[-| :]+\|?$/)) // skip separator rows (|---|---|)
    .map((l) =>
      l
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((cell) => cell.trim())
    );

  if (rows.length === 0) return <pre className={styles.proseContent}>{text}</pre>;

  const [header, ...body] = rows;

  return (
    <div className={styles.tableWrapper}>
      <table className={styles.dataTable}>
        <thead>
          <tr>
            {header.map((cell, i) => (
              <th key={i} className={styles.th}>
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, ri) => (
            <tr key={ri} className={ri % 2 === 0 ? styles.trEven : styles.trOdd}>
              {row.map((cell, ci) => (
                <td key={ci} className={`${styles.td} ${ci > 0 ? styles.tdNum : styles.tdLabel}`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

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
            {/* Render as a proper table for TABLE chunks, prose otherwise */}
            {isTable && isPipeTable(data.text)
              ? parsePipeTable(data.text)
              : <div className={styles.proseContent}>{data.text}</div>
            }
          </>
        ) : null}
      </div>
    </aside>
  );
};
