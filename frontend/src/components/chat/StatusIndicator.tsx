import React, { useState, useEffect } from 'react';
import { Loader2, Search, Database, ShieldCheck } from 'lucide-react';
import styles from './ChatComponents.module.css';

export type TaskStage = 'idle' | 'resolving' | 'retrieving' | 'checking' | 'complete';

interface StatusIndicatorProps {
  isLoading: boolean;
}

const STAGES: { stage: TaskStage; label: string; icon: React.FC<{ size?: number; className?: string }> }[] = [
  { stage: 'resolving', label: 'Analyzing & resolving question context...', icon: Search },
  { stage: 'retrieving', label: 'Retrieving SEC 10-K document chunks...', icon: Database },
  { stage: 'checking', label: 'Verifying numerical groundedness...', icon: ShieldCheck },
];

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ isLoading }) => {
  const [currentStageIdx, setCurrentStageIdx] = useState(0);

  useEffect(() => {
    if (!isLoading) {
      setCurrentStageIdx(0);
      return;
    }

    // Advance stages over time: 0ms -> 1200ms -> 2500ms
    const timer1 = setTimeout(() => setCurrentStageIdx(1), 1200);
    const timer2 = setTimeout(() => setCurrentStageIdx(2), 2500);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [isLoading]);

  if (!isLoading) return null;

  // GRACEFUL DEGRADATION:
  // If the query takes longer than 2.5 seconds (e.g. LLM generation latency or rate limit backoff),
  // currentStageIdx stays at index 2 ('checking') with an active pulsing shimmer.
  // It holds indefinitely on 'checking' until the API call completes, never overshooting or hiding.
  const activeStage = STAGES[Math.min(currentStageIdx, STAGE_COUNT - 1)];
  const StageIcon = activeStage.icon;

  return (
    <div className={styles.statusPill} aria-live="polite">
      <Loader2 className={styles.spinner} size={14} />
      <StageIcon size={14} className={styles.stageIcon} />
      <span className={styles.statusText}>{activeStage.label}</span>
    </div>
  );
};

const STAGE_COUNT = STAGES.length;
