import React from 'react';
import { MessageSquare } from 'lucide-react';
import type { SessionSummary } from '../../api/types';
import styles from './SessionSidebar.module.css';

interface SessionItemProps {
  session: SessionSummary;
  isActive: boolean;
  onSelect: (id: string) => void;
}

export const SessionItem: React.FC<SessionItemProps> = ({ session, isActive, onSelect }) => {
  const formattedTitle = session.title || `Session ${session.session_id.slice(0, 8)}`;
  const formattedDate = session.last_active
    ? new Date(session.last_active).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '';

  return (
    <button
      className={`${styles.sessionItem} ${isActive ? styles.active : ''}`}
      onClick={() => onSelect(session.session_id)}
      title={session.session_id}
    >
      <MessageSquare className={styles.itemIcon} size={16} />
      <div className={styles.itemContent}>
        <span className={styles.itemTitle}>{formattedTitle}</span>
        <span className={styles.itemDate}>{formattedDate}</span>
      </div>
    </button>
  );
};
