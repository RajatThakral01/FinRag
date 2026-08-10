import React, { useState } from 'react';
import { User, FileText, CheckCircle2, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
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
  onSelectChunk?: (chunkId: string) => void;
}


// Format ISO timestamp as HH:MM for today, or MMM D · HH:MM for older dates
function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const isToday =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  if (isToday) return time;
  return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} · ${time}`;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onSelectChunk }) => {
  const isUser = message.sender === 'user';
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

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
          <div className={styles.metaLeft}>
            <span className={styles.senderName}>
              {isUser ? 'ANALYST QUERY' : 'SEC 10-K VERIFICATION DOSSIER'}
            </span>
            {!isUser && message.cacheHit && (
              <span className={styles.cacheHitBadge} title="Answer retrieved from verified semantic cache">
                <CheckCircle2 size={11} /> Cache Verified
              </span>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {message.timestamp && (
              <span className={styles.timestamp}>{formatTimestamp(message.timestamp)}</span>
            )}
            {!isUser && (
              <button className={styles.copyBtn} onClick={handleCopy} title="Copy to clipboard">
                {copied ? <Check size={14} color="var(--accent-success)" /> : <Copy size={14} />}
              </button>
            )}
          </div>
        </div>

        {/* Audit Card Bubble */}
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
          {!isUser && (
            <>
              {/* Show resolved question badge only when question was rewritten */}
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

          {/* Main Answer Text Body — markdown rendered for assistant, plain for user */}
          <div className={styles.textBody}>
            {isUser ? (
              message.content
            ) : (
              <div className={styles.markdownContent}>
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({node, inline, className, children, ...props}: any) {
                      const match = /language-(\w+)/.exec(className || '')
                      return !inline && match ? (
                        <SyntaxHighlighter
                          {...props}
                          children={String(children).replace(/\n$/, '')}
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                        />
                      ) : (
                        <code {...props} className={className}>
                          {children}
                        </code>
                      )
                    }
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            )}
          </div>

          {/* Sources Accordion */}
          {!isUser && message.sources && message.sources.length > 0 && (
            <SourcesPanel sources={message.sources} onSelectChunk={onSelectChunk} />
          )}
        </div>
      </div>
    </div>
  );
};
