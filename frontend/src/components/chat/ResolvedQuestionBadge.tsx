import React from 'react';
import { Shield } from 'lucide-react';
import styles from './ChatComponents.module.css';

interface ResolvedQuestionBadgeProps {
  resolvedQuestion: string;
  questionWasResolved: boolean;
}

export const ResolvedQuestionBadge: React.FC<ResolvedQuestionBadgeProps> = ({
  resolvedQuestion,
  questionWasResolved,
}) => {
  if (!questionWasResolved || !resolvedQuestion) return null;

  return (
    <div className={styles.resolvedBadge} title="Context Resolver expanded follow-up question">
      <Shield size={13} className={styles.resolvedIcon} />
      <span>
        Context Resolved: <strong className={styles.resolvedText}>"{resolvedQuestion}"</strong>
      </span>
    </div>
  );
};
