import React from 'react';
import { User, FileText, CheckCircle2 } from 'lucide-react';
import type { ChunkSource } from '../../api/types';
import { ResolvedQuestionBadge } from './ResolvedQuestionBadge';
import { SourcesPanel } from './SourcesPanel';
import { WarningBanner } from './WarningBanner';
import { HonestFailureCard } from './HonestFailureCard';
import styles from './MessageBubble.module.css';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  rawQuestion?: string;
  resolvedQuestion?: string;
  questionWasResolved?: boolean;
  content: string;
  sources?: ChunkSource[];
  errorMessage?: string | null;
  answerSource?: string;
  cacheHit?: boolean;
  timestamp?: string;
}

interface MessageBubbleProps {
  message: ChatMessage;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.sender === 'user';

  return (
    <div className={`${styles.row} ${isUser ? styles.userRow : styles.assistantRow}`}>
      {/* Avatar */}
      <div className={`${styles.avatar} ${isUser ? styles.userAvatar : styles.assistantAvatar}`}>
        {isUser ? <User size={15} /> : <FileText size={15} />}
      </div>

      {/* Message Content Container */}
      <div className={styles.bubbleContainer}>
        {/* Header Metadata */}
        <div className={styles.metaRow}>
          <span className={styles.senderName}>
            {isUser ? 'ANALYST QUERY' : 'SEC 10-K VERIFICATION DOSSIER'}
          </span>
          {!isUser && message.cacheHit && (
            <span className={styles.cacheHitBadge} title="Answer retrieved from verified semantic cache">
              <CheckCircle2 size={11} /> Cache Verified
            </span>
          )}
        </div>

        {/* Audit Card Bubble */}
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
          {!isUser && (
            <>
              {/* Section 16 Requirement: resolved_question displayed ONLY when question_was_resolved is true */}
              <ResolvedQuestionBadge
                resolvedQuestion={message.resolvedQuestion || ''}
                questionWasResolved={!!message.questionWasResolved}
              />

              {/* Low-confidence warning banner */}
              <WarningBanner errorMessage={message.errorMessage || null} />

              {/* Honest-failure placeholder */}
              <HonestFailureCard
                answerSource={message.answerSource}
                finalAnswer={message.content}
              />
            </>
          )}

          {/* Main Answer Text Body */}
          <div className={styles.textBody}>{message.content}</div>

          {/* Sources Accordion */}
          {!isUser && message.sources && message.sources.length > 0 && (
            <SourcesPanel sources={message.sources} />
          )}
        </div>
      </div>
    </div>
  );
};
