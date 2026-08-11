import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AlertCircle, RefreshCw, ChevronDown } from 'lucide-react';
import { useSessionContext } from '../../SessionContext';
import { useToast } from '../../ToastContext';
import { api } from '../../api/client';
import { MessageBubble, type ChatMessage } from './MessageBubble';
import { StatusIndicator } from './StatusIndicator';
import { TypingBubble } from './TypingBubble';
import { ChatInput } from './ChatInput';
import { ChunkPreviewPanel } from './ChunkPreviewPanel';
import { WelcomeScreen } from './WelcomeScreen';
import styles from './ChatContainer.module.css';

export const ChatContainer: React.FC = () => {
  const { activeSessionId, createSession, refreshSessions } = useSessionContext();
  const { showToast } = useToast();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingTurns, setIsLoadingTurns] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const [typingStage, setTypingStage] = useState<string>('Processing query...');
  const contentRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Detect whether we are near the bottom of the scroll area
  const handleScroll = useCallback(() => {
    const el = scrollAreaRef.current;
    if (!el) return;
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setShowScrollBtn(distFromBottom > 120);
  }, []);

  useEffect(() => {
    if (!contentRef.current) return;
    const observer = new ResizeObserver(() => {
      scrollToBottom();
    });
    observer.observe(contentRef.current);
    return () => observer.disconnect();
  }, []);

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
          showToast('New research session started', 'info');
        } catch (err) {
          const msg = `Failed to initialize session: ${err instanceof Error ? err.message : String(err)}`;
          setQueryError(msg);
          showToast(msg, 'error');
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
        showToast(errStr, 'error');
      } finally {
        setIsSubmitting(false);
      }
    },
    [activeSessionId, createSession, refreshSessions]
  );

  return (
    <div className={styles.container}>
      {/* Messages Scroll Area */}
      <div
        className={styles.scrollArea}
        ref={scrollAreaRef}
        onScroll={handleScroll}
      >
        <div ref={contentRef} style={{ display: 'flex', flexDirection: 'column', minHeight: '100%' }}>
          {isLoadingTurns ? (
            <div className={styles.centerNotice}>Accessing session audit registry...</div>
          ) : messages.length === 0 ? (
            <WelcomeScreen onQueryClick={handleQuerySubmit} isSubmitting={isSubmitting} />
          ) : (
            messages.map((msg, idx) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                index={idx}
                onSelectChunk={(chunkId) => setActiveChunkId(chunkId)}
              />
            ))
          )}

          {/* Typing bubble — shown in-place where the answer will appear */}
          {isSubmitting && (
            <TypingBubble stage={typingStage} />
          )}

          {/* Hidden StatusIndicator drives the stage label via callback */}
          <StatusIndicator isLoading={isSubmitting} onStageChange={setTypingStage} />

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

          {/* Invisible div to anchor the scroll */}
          <div ref={messagesEndRef} style={{ height: 1 }} />
        </div>
      </div>

      {/* Fixed Bottom Input Area */}
      <div className={styles.inputContainer}>
        <ChatInput onSubmit={handleQuerySubmit} isLoading={isSubmitting} />
      </div>

      {/* Scroll-to-bottom button — visible when user has scrolled up */}
      {showScrollBtn && messages.length > 0 && (
        <button
          className={styles.scrollToBottomBtn}
          onClick={scrollToBottom}
          title="Jump to latest message"
        >
          <ChevronDown size={13} />
          JUMP TO LATEST
        </button>
      )}

      {/* Full Chunk Slide-in Preview Panel */}
      <ChunkPreviewPanel
        chunkId={activeChunkId}
        onClose={() => setActiveChunkId(null)}
      />
    </div>
  );
};
