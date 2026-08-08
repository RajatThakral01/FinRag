import React from 'react';
import styles from './ChatContainer.module.css'; // Re-use the existing styles for now

const EXAMPLE_QUERIES = [
  { label: 'FACTUAL', text: "What was Apple's total revenue in fiscal 2024?" },
  { label: 'FACTUAL', text: "What was NVIDIA's R&D expense in 2024?" },
  { label: 'COMPARATIVE', text: 'Compare Apple and Microsoft operating income' },
  { label: 'COMPARATIVE', text: 'Which company had the highest R&D spend in 2024?' },
  { label: 'CALCULATION', text: "What was Apple's gross margin percentage?" },
  { label: 'CALCULATION', text: "What was Tesla's revenue growth from 2023 to 2024?" },
];

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
