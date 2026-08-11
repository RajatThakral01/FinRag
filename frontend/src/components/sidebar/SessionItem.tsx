import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare } from 'lucide-react';
import type { SessionSummary } from '../../api/types';
import { api } from '../../api/client';
import styles from './SessionSidebar.module.css';

interface SessionItemProps {
  session: SessionSummary;
  isActive: boolean;
  onSelect: (id: string) => void;
  onRename: () => void; // triggers a session list refresh after rename
}

export const SessionItem: React.FC<SessionItemProps> = ({ session, isActive, onSelect, onRename }) => {
  const formattedTitle = session.title || `Session ${session.session_id.slice(0, 8)}`;
  const formattedDate = session.last_active
    ? new Date(session.last_active).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '';

  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(formattedTitle);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.select();
    }
  }, [isEditing]);

  const startEditing = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDraft(formattedTitle);
    setIsEditing(true);
  };

  const commitEdit = async () => {
    setIsEditing(false);
    const trimmed = draft.trim();
    if (!trimmed || trimmed === formattedTitle) return;
    try {
      await api.updateSessionTitle(session.session_id, trimmed);
      onRename();
    } catch {
      // silently ignore rename errors — title stays as-is on next refresh
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') commitEdit();
    if (e.key === 'Escape') setIsEditing(false);
  };

  return (
    <button
      className={`${styles.sessionItem} ${isActive ? styles.active : ''}`}
      onClick={() => onSelect(session.session_id)}
      onDoubleClick={startEditing}
      title="Click to open · Double-click to rename"
    >
      <MessageSquare className={styles.itemIcon} size={16} />
      <div className={styles.itemContent}>
        {isEditing ? (
          <input
            ref={inputRef}
            className={styles.itemTitleInput}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={handleKeyDown}
            onClick={(e) => e.stopPropagation()}
            maxLength={80}
          />
        ) : (
          <span className={styles.itemTitle}>{formattedTitle}</span>
        )}
        <span className={styles.itemDate}>{formattedDate}</span>
      </div>
    </button>
  );
};
