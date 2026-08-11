import React from 'react';
import { FileText } from 'lucide-react';
import styles from './TypingBubble.module.css';

interface TypingBubbleProps {
  stage: string;
}

export const TypingBubble: React.FC<TypingBubbleProps> = ({ stage }) => {
  return (
    <div className={styles.row}>
      {/* Avatar — mirrors assistant bubble */}
      <div className={styles.avatar}>
        <FileText size={15} />
      </div>

      {/* Bubble Container */}
      <div className={styles.bubbleContainer}>
        {/* Meta row */}
        <div className={styles.metaRow}>
          <span className={styles.senderName}>SEC 10-K VERIFICATION DOSSIER</span>
        </div>

        {/* Ghost bubble */}
        <div className={styles.bubble}>
          {/* Bouncing dots */}
          <div className={styles.dotsRow}>
            <span className={styles.dot} />
            <span className={styles.dot} />
            <span className={styles.dot} />
          </div>
          {/* Stage label */}
          <span className={styles.stageLabel}>{stage}</span>
        </div>
      </div>
    </div>
  );
};
