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
  onSelectChunk?: (chunkId: string) => void;
}

// ---------------------------------------------------------------------------
// Lightweight markdown-to-JSX renderer (no dependencies)
// Handles: **bold**, *italic*, bullet lists (- / *), numbered lists, paragraphs
// ---------------------------------------------------------------------------
function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split('\n');
  const nodes: React.ReactNode[] = [];
  let i = 0;

  const inlineFormat = (line: string, key: string): React.ReactNode => {
    // Split by bold (**text**) and italic (*text*) markers
    const parts = line.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
    return (
      <React.Fragment key={key}>
        {parts.map((part, idx) => {
          if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={idx}>{part.slice(2, -2)}</strong>;
          }
          if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
            return <em key={idx}>{part.slice(1, -1)}</em>;
          }
          return part;
        })}
      </React.Fragment>
    );
  };

  while (i < lines.length) {
    const line = lines[i];

    // Bullet list block
    if (/^[-*]\s/.test(line)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^[-*]\s/.test(lines[i])) {
        items.push(<li key={i}>{inlineFormat(lines[i].replace(/^[-*]\s/, ''), `li-${i}`)}</li>);
        i++;
      }
      nodes.push(<ul key={`ul-${i}`} className={styles.mdList}>{items}</ul>);
      continue;
    }

    // Numbered list block
    if (/^\d+\.\s/.test(line)) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        items.push(<li key={i}>{inlineFormat(lines[i].replace(/^\d+\.\s/, ''), `oli-${i}`)}</li>);
        i++;
      }
      nodes.push(<ol key={`ol-${i}`} className={styles.mdList}>{items}</ol>);
      continue;
    }

    // Empty line → paragraph spacer (only if not the first node)
    if (line.trim() === '') {
      if (nodes.length > 0) {
        nodes.push(<div key={`sp-${i}`} className={styles.mdSpacer} />);
      }
      i++;
      continue;
    }

    // Plain text line with inline formatting
    nodes.push(<p key={`p-${i}`} className={styles.mdParagraph}>{inlineFormat(line, `p-${i}`)}</p>);
    i++;
  }

  return nodes;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onSelectChunk }) => {
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
            {isUser ? message.content : renderMarkdown(message.content)}
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
