import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { useSessionContext } from '../../SessionContext';
import { api } from '../../api/client';
import { MessageBubble, type ChatMessage } from './MessageBubble';
import { StatusIndicator } from './StatusIndicator';
import { ChatInput } from './ChatInput';
import { ChunkPreviewPanel } from './ChunkPreviewPanel';
import styles from './ChatContainer.module.css';

const EXAMPLE_QUERIES = [
  { label: 'FACTUAL', text: "What was Apple's total revenue in fiscal 2024?" },
  { label: 'FACTUAL', text: "What was NVIDIA's R&D expense in 2024?" },
  { label: 'COMPARATIVE', text: 'Compare Apple and Microsoft operating income' },
  { label: 'COMPARATIVE', text: 'Which company had the highest R&D spend in 2024?' },
  { label: 'CALCULATION', text: "What was Apple's gross margin percentage?" },
  { label: 'CALCULATION', text: "What was Tesla's revenue growth from 2023 to 2024?" },
];

export const ChatContainer: React.FC = () => {
  const { activeSessionId, createSession, refreshSessions } = useSessionContext();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingTurns, setIsLoadingTurns] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSubmitting]);

  const loadTurns = useCallback(async (sessionId: string, showLoadingIndicator = true) => {
    try {
      if (showLoadingIndicator) setIsLoadingTurns(true);
      setQueryError(null);
      const turns = await api.getSessionTurns(sessionId);

      const loadedMessages: ChatMessage[] = [];
      turns.forEach((turn) => {
        // User message
        loadedMessages.push({
          id: `turn-${turn.turn_number}-user`,
          sender: 'user',
          content: turn.raw_question,
          timestamp: turn.created_at,
        });

        // Assistant message
        loadedMessages.push({
          id: `turn-${turn.turn_number}-assistant`,
          sender: 'assistant',
          rawQuestion: turn.raw_question,
          resolvedQuestion: turn.resolved_question,
          questionWasResolved: (turn.raw_question || '').trim() !== (turn.resolved_question || '').trim(),
          content: turn.final_answer || 'No answer generated.',
          errorMessage: turn.error_message,
          timestamp: turn.created_at,
        });
      });

      setMessages(loadedMessages);
    } catch (err) {
      console.error('Failed to load turns:', err);
    } finally {
      if (showLoadingIndicator) setIsLoadingTurns(false);
    }
  }, []);

  // Load turn history when activeSessionId changes
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      setActiveChunkId(null);
      return;
    }
    
    loadTurns(activeSessionId);
  }, [activeSessionId, loadTurns]);

  const handleQuerySubmit = useCallback(
    async (questionText: string) => {
      let sessionId = activeSessionId;

      if (!sessionId) {
        try {
          sessionId = await createSession();
        } catch (err) {
          setQueryError(`Failed to initialize session: ${err instanceof Error ? err.message : String(err)}`);
          return;
        }
      }

      const tempUserMsgId = `user-${Date.now()}`;
      const userMessage: ChatMessage = {
        id: tempUserMsgId,
        sender: 'user',
        content: questionText,
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsSubmitting(true);
      setQueryError(null);

      try {
        // Re-load the session turns from the backend so we get the canonical
        // turn numbers and IDs instead of keeping optimistic ones. We fetch
        // silently to avoid blinking the UI.
        await loadTurns(sessionId, false);
        refreshSessions();
      } catch (err) {
        const errStr = err instanceof Error ? err.message : String(err);
        setQueryError(errStr);
      } finally {
        setIsSubmitting(false);
      }
    },
    [activeSessionId, createSession, refreshSessions]
  );

  return (
    <div className={styles.container}>
      {/* Messages Scroll Area */}
      <div className={styles.scrollArea}>
        {isLoadingTurns ? (
          <div className={styles.centerNotice}>Accessing session audit registry...</div>
        ) : messages.length === 0 ? (
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
                  onClick={() => handleQuerySubmit(q.text)}
                  disabled={isSubmitting}
                >
                  <span className={styles.queryCardLabel}>{q.label}</span>
                  <span className={styles.queryCardText}>{q.text}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onSelectChunk={(chunkId) => setActiveChunkId(chunkId)}
            />
          ))
        )}

        {/* Pending Query Status Indicator */}
        {isSubmitting && (
          <div className={styles.statusRow}>
            <StatusIndicator isLoading={isSubmitting} />
          </div>
        )}

        {/* API Error Notification */}
        {queryError && (
          <div className={styles.errorAlert}>
            <AlertCircle size={18} className={styles.errorIcon} />
            <div className={styles.errorContent}>
              <span className={styles.errorTitle}>Filing Query Execution Error</span>
              <span className={styles.errorText}>{queryError}</span>
            </div>
            <button className={styles.retryBtn} onClick={() => setQueryError(null)}>
              <RefreshCw size={13} /> Dismiss
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Fixed Bottom Input Area */}
      <div className={styles.inputContainer}>
        <ChatInput onSubmit={handleQuerySubmit} isLoading={isSubmitting} />
      </div>

      {/* Full Chunk Slide-in Preview Panel */}
      <ChunkPreviewPanel
        chunkId={activeChunkId}
        onClose={() => setActiveChunkId(null)}
      />
    </div>
  );
};
