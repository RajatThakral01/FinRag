import React, { useState, useEffect } from 'react';
import { Loader2, Search, Database, ShieldCheck, Cpu, Filter } from 'lucide-react';
import styles from './ChatComponents.module.css';

export type TaskStage = 'idle' | 'resolving' | 'retrieving' | 'grading' | 'generating' | 'checking';

interface StatusIndicatorProps {
  isLoading: boolean;
  onStageChange?: (label: string) => void;
}

const STAGES: { stage: TaskStage; label: string; icon: React.FC<{ size?: number; className?: string }> }[] = [
  { stage: 'resolving',  label: 'Resolving query context...',              icon: Search },
  { stage: 'retrieving', label: 'Retrieving SEC document excerpts...',      icon: Database },
  { stage: 'grading',    label: 'Grading retrieved context relevance...',   icon: Filter },
  { stage: 'generating', label: 'Synthesizing verified answer...',          icon: Cpu },
  { stage: 'checking',   label: 'Verifying numerical groundedness...',      icon: ShieldCheck },
];

const STAGE_COUNT = STAGES.length;
// Stage advance timers: 0ms → 1500ms → 3000ms → 5000ms → 8000ms
const TIMERS = [1500, 3000, 5000, 8000];
// Show slow-query note after this many ms on the final stage
const SLOW_QUERY_MS = 10000;

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ isLoading, onStageChange }) => {
  const [currentStageIdx, setCurrentStageIdx] = useState(0);
  const [showSlowNote, setShowSlowNote] = useState(false);

  useEffect(() => {
    if (!isLoading) {
      setCurrentStageIdx(0);
      setShowSlowNote(false);
      return;
    }

    // Advance through stages on fixed timers
    const timers = TIMERS.map((delay, idx) =>
      setTimeout(() => setCurrentStageIdx(idx + 1), delay)
    );

    // Show slow-query note after SLOW_QUERY_MS
    const slowTimer = setTimeout(() => setShowSlowNote(true), SLOW_QUERY_MS);

    return () => {
      timers.forEach(clearTimeout);
      clearTimeout(slowTimer);
    };
  }, [isLoading]);

  const activeStage = STAGES[Math.min(currentStageIdx, STAGE_COUNT - 1)];

  // Notify parent about stage label changes
  useEffect(() => {
    if (isLoading && onStageChange) {
      onStageChange(activeStage.label);
    }
  }, [activeStage.label, isLoading, onStageChange]);

  if (!isLoading) return null;
  const StageIcon = activeStage.icon;

  return (
    <div className={styles.statusPill} aria-live="polite">
      <Loader2 className={styles.spinner} size={14} />
      <StageIcon size={14} className={styles.stageIcon} />
      <span className={styles.statusText}>
        {activeStage.label}
        {showSlowNote && currentStageIdx >= STAGE_COUNT - 1 && (
          <span className={styles.slowNote}> — this may take up to 30s</span>
        )}
      </span>
    </div>
  );
};
