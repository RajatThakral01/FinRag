import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileCode, Building, Table, AlignLeft, Info, Eye } from 'lucide-react';
import type { ChunkSource } from '../../api/types';
import styles from './ChatComponents.module.css';

interface SourcesPanelProps {
  sources: ChunkSource[];
  onSelectChunk?: (chunkId: string) => void;
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({ sources, onSelectChunk }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (!sources || sources.length === 0) return null;

  const toggleCardExpand = (idx: number) => {
    setExpandedIdx((prev) => (prev === idx ? null : idx));
  };

  return (
    <div className={styles.sourcesContainer}>
      <button
        className={styles.sourcesHeader}
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
      >
        <div className={styles.sourcesTitle}>
          <FileCode size={13} className={styles.titleIcon} />
          <span>
            SEC 10-K Citation Index ({sources.length} excerpt{sources.length > 1 ? 's' : ''})
          </span>
        </div>
        {isOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>

      {isOpen && (
        <div className={styles.sourcesList}>
          {sources.map((src, idx) => {
            const company = src.company || 'Unknown Company';
            const ticker = src.ticker ? ` (${src.ticker})` : '';
            const year = src.year || '2024';
            const section = src.section_name || src.section || 'Item 8';
            const chunkId = src.chunk_id || `chunk_${idx + 1}`;
            const chunkType = (src.chunk_type || 'excerpt').toUpperCase();
            const tableName = src.table_name || '';
            const isExpanded = expandedIdx === idx;

            // Distinct display label: use table_name if present, otherwise section name
            const displayTitle = tableName ? tableName : section;

            return (
              <div
                key={chunkId || idx}
                className={`${styles.sourceCard} ${isExpanded ? styles.sourceCardExpanded : ''}`}
                onClick={() => toggleCardExpand(idx)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleCardExpand(idx);
                  }
                }}
                title="Click to expand chunk audit details"
              >
                {/* Header Row */}
                <div className={styles.sourceCompanyRow}>
                  <Building size={12} className={styles.companyIcon} />
                  <span className={styles.sourceCompany}>
                    {company}
                    <span className={styles.tickerTag}>{ticker}</span>
                  </span>
                  <span className={styles.chunkTypeBadge}>
                    {chunkType === 'TABLE' ? <Table size={10} /> : <AlignLeft size={10} />}
                    {chunkType}
                  </span>
                  <span className={styles.sourceTag}>FY{year}</span>
                </div>

                {/* Distinct Title & Table/Section Name */}
                <div className={styles.sourceTitleRow}>
                  <span className={styles.displayTitle}>{displayTitle}</span>
                </div>

                {/* Monospace Chunk ID */}
                <div className={styles.sourceMeta}>
                  <span className={styles.chunkIdText}>ID: {chunkId}</span>
                </div>

                {/* Action Bar: Show full chunk button & Metadata Expand */}
                <div className={styles.cardFooterRow}>
                  <button
                    className={styles.fullChunkBtn}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onSelectChunk && chunkId) {
                        onSelectChunk(chunkId);
                      }
                    }}
                    title="Open full chunk slide-in preview panel"
                  >
                    <Eye size={12} />
                    <span>Show full chunk</span>
                  </button>

                  <div className={styles.cardFooterHint}>
                    <span>{isExpanded ? 'Hide metadata' : 'Inspect metadata'}</span>
                    {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  </div>
                </div>

                {/* Expanded Excerpt Metadata Drawer */}
                {isExpanded && (
                  <div
                    className={styles.expandedDrawer}
                    onClick={(e) => e.stopPropagation()} // Prevent collapse when clicking inside
                  >
                    <div className={styles.drawerHeader}>
                      <Info size={12} className={styles.drawerIcon} />
                      <span>Audited Chunk Parameters</span>
                    </div>

                    <div className={styles.drawerGrid}>
                      <div className={styles.drawerItem}>
                        <span className={styles.drawerLabel}>Chunk ID:</span>
                        <code className={styles.drawerCode}>{chunkId}</code>
                      </div>
                      <div className={styles.drawerItem}>
                        <span className={styles.drawerLabel}>Company:</span>
                        <span className={styles.drawerVal}>{company} {ticker}</span>
                      </div>
                      <div className={styles.drawerItem}>
                        <span className={styles.drawerLabel}>Filing Section:</span>
                        <span className={styles.drawerVal}>{section}</span>
                      </div>
                      {tableName && (
                        <div className={styles.drawerItem}>
                          <span className={styles.drawerLabel}>Table Name:</span>
                          <span className={styles.drawerVal}>{tableName}</span>
                        </div>
                      )}
                      <div className={styles.drawerItem}>
                        <span className={styles.drawerLabel}>Data Type:</span>
                        <span className={styles.drawerVal}>{chunkType}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
