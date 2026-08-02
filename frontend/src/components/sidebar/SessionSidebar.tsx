import React from 'react';
import { Plus, FileBarChart, Layers } from 'lucide-react';
import { useSessionContext } from '../../SessionContext';
import { SessionItem } from './SessionItem';
import { HealthBadge } from '../common/HealthBadge';
import styles from './SessionSidebar.module.css';

export const SessionSidebar: React.FC = () => {
  const { sessions, activeSessionId, createSession, selectSession, isLoadingSessions } =
    useSessionContext();

  const handleNewSession = async () => {
    try {
      await createSession();
    } catch (err) {
      console.error('Failed to create new session:', err);
    }
  };

  return (
    <aside className={styles.sidebar}>
      {/* Sidebar Header */}
      <div className={styles.header}>
        <div className={styles.logoRow}>
          <div className={styles.logoIcon}>
            <FileBarChart size={18} style={{ color: 'var(--accent-sole)' }} />
          </div>
          <div>
            <h1 className={styles.logoTitle}>FinRAG Terminal</h1>
            <p className={styles.logoSubtitle}>SEC 10-K Research Dossier</p>
          </div>
        </div>
        <button className={styles.newSessionBtn} onClick={handleNewSession}>
          <Plus size={15} />
          <span>New Research Session</span>
        </button>
      </div>

      {/* Session List */}
      <div className={styles.sessionListContainer}>
        <div className={styles.sectionHeader}>
          <Layers size={13} />
          <span>Filing History Registry</span>
        </div>

        {isLoadingSessions ? (
          <div className={styles.loadingState}>Loading session registry...</div>
        ) : sessions.length === 0 ? (
          <div className={styles.emptyState}>No research sessions recorded.</div>
        ) : (
          <div className={styles.sessionList}>
            {sessions.map((session) => (
              <SessionItem
                key={session.session_id}
                session={session}
                isActive={session.session_id === activeSessionId}
                onSelect={selectSession}
              />
            ))}
          </div>
        )}
      </div>

      {/* Sidebar Footer */}
      <div className={styles.footer}>
        <HealthBadge />
      </div>
    </aside>
  );
};
