import React from 'react';
import styles from './WelcomeScreen.module.css';
import { EXAMPLE_QUERIES } from '../../constants';

interface WelcomeScreenProps {
  onQueryClick: (query: string) => void;
  isSubmitting: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onQueryClick, isSubmitting }) => {
  return (
    <div className={styles.welcomeHero}>
      <div className={styles.heroBadge}>SEC 10-K RESEARCH TERMINAL</div>
      <h2 className={styles.heroTitle}>Audited Financial Intelligence</h2>
      <p className={styles.heroSubtitle}>
        Extract verified figures directly from official 10-K filings. Multi-turn
        follow-ups resolve automatically with numerical groundedness verification.
      </p>
      <div className={styles.queryCardGrid}>
        {EXAMPLE_QUERIES.map((q, i) => (
          <button
            key={i}
            className={styles.queryCard}
            onClick={() => onQueryClick(q.text)}
            disabled={isSubmitting}
          >
            <span className={styles.queryCardLabel}>{q.label}</span>
            <span className={styles.queryCardText}>{q.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
