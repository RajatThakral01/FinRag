import React from 'react';
import { AlertTriangle } from 'lucide-react';
import styles from './ChatComponents.module.css';

interface WarningBannerProps {
  errorMessage: string | null;
}

export const WarningBanner: React.FC<WarningBannerProps> = ({ errorMessage }) => {
  if (!errorMessage) return null;

  return (
    <div className={styles.warningCard}>
      <div className={styles.warningHeader}>
        <AlertTriangle size={15} className={styles.warningIcon} />
        <span className={styles.warningLabel}>Low confidence</span>
      </div>
      <p className={styles.warningText}>{errorMessage}</p>
    </div>
  );
};
