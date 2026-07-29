import React from 'react';
import { AlertCircle } from 'lucide-react';
import styles from './ChatComponents.module.css';

interface HonestFailureCardProps {
  answerSource?: string;
  finalAnswer: string;
}

export const HonestFailureCard: React.FC<HonestFailureCardProps> = ({
  answerSource,
  finalAnswer,
}) => {
  const isHonestFailure = answerSource === 'hallucination_exhausted';

  if (!isHonestFailure) {
    return null;
  }

  return (
    <div className={styles.honestFailureCard}>
      <div className={styles.honestFailureHeader}>
        <AlertCircle size={15} className={styles.failureIcon} />
        <span className={styles.honestFailureLabel}>Unverified</span>
        <span className={styles.todoBadge}>TODO: Backend field</span>
      </div>
      <p className={styles.honestFailureBody}>{finalAnswer}</p>
    </div>
  );
};
