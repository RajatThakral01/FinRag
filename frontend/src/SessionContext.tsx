import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { api } from './api/client';
import type { SessionSummary } from './api/types';

interface SessionContextType {
  sessions: SessionSummary[];
  activeSessionId: string | null;
  isLoadingSessions: boolean;
  createSession: (title?: string) => Promise<string>;
  selectSession: (sessionId: string) => void;
  refreshSessions: () => Promise<void>;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

const LOCAL_STORAGE_ACTIVE_KEY = 'financial_rag_active_session';

export const SessionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoadingSessions, setIsLoadingSessions] = useState<boolean>(true);

  const navigate = useNavigate();
  const location = useLocation();

  // Sync activeSessionId with URL path /s/<id>
  useEffect(() => {
    const match = location.pathname.match(/^\/s\/([^/]+)/);
    const urlSessionId = match ? match[1] : null;
    const localSessionId = localStorage.getItem(LOCAL_STORAGE_ACTIVE_KEY);

    if (urlSessionId) {
      setActiveSessionId(urlSessionId);
      localStorage.setItem(LOCAL_STORAGE_ACTIVE_KEY, urlSessionId);
    } else if (localSessionId) {
      // If we are at root, redirect to the last known session
      if (location.pathname === '/') {
        navigate(`/s/${localSessionId}`, { replace: true });
      } else {
        setActiveSessionId(null);
      }
    } else {
      setActiveSessionId(null);
    }
  }, [location.pathname, navigate]);

  const selectSession = useCallback((sessionId: string) => {
    navigate(`/s/${sessionId}`);
  }, [navigate]);

  const refreshSessions = useCallback(async () => {
    try {
      setIsLoadingSessions(true);
      const data = await api.listSessions();
      setSessions(data);
    } catch (err) {
      console.error('Failed to load session list:', err);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  const createSession = useCallback(async (title?: string): Promise<string> => {
    const res = await api.createSession(title);
    const newId = res.session_id;
    selectSession(newId);
    await refreshSessions();
    return newId;
  }, [selectSession, refreshSessions]);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  return (
    <SessionContext.Provider
      value={{
        sessions,
        activeSessionId,
        isLoadingSessions,
        createSession,
        selectSession,
        refreshSessions,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
};

export const useSessionContext = () => {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSessionContext must be used within a SessionProvider');
  }
  return context;
};
