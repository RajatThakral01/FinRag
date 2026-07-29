import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { useSessionContext } from '../../context/SessionContext';
import { api } from '../../api/client';
import { MessageBubble, type ChatMessage } from './MessageBubble';
import { StatusIndicator } from './StatusIndicator';
import { ChatInput } from './ChatInput';
import styles from './ChatContainer.module.css';

export const ChatContainer: React.FC = () => {
  const { activeSessionId, createSession, refreshSessions } = useSessionContext();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingTurns, setIsLoadingTurns] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isSubmitting]);

  // Load turn history when activeSessionId changes
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      return;
    }

    let isMounted = true;
    const loadTurns = async () => {
      try {
        setIsLoadingTurns(true);
        setQueryError(null);
        const turns = await api.getSessionTurns(activeSessionId);
        if (!isMounted) return;

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
        if (isMounted) {
          console.error('Failed to load turns:', err);
        }
      } finally {
        if (isMounted) setIsLoadingTurns(false);
      }
    };

    loadTurns();
    return () => {
      isMounted = false;
    };
  }, [activeSessionId]);

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
        const response = await api.submitQuery(sessionId, questionText);

        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          sender: 'assistant',
          rawQuestion: response.raw_question,
          resolvedQuestion: response.resolved_question,
          questionWasResolved: response.question_was_resolved,
          content: response.final_answer,
          sources: response.chunk_sources,
          errorMessage: response.error_message,
          answerSource: response.answer_source,
          cacheHit: response.cache_hit,
        };

        setMessages((prev) => [...prev, assistantMessage]);
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
              Extract verified figures directly from official 10-K filings for Apple, Microsoft, NVIDIA, and Tesla.
              Multi-turn follow-ups resolve automatically with numerical groundedness verification.
            </p>
          </div>
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
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
    </div>
  );
};
