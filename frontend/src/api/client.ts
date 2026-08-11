import type {
  HealthResponse,
  CreateSessionResponse,
  SessionSummary,
  TurnItem,
  QueryResponse,
  ChunkDetailResponse,
} from './types';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function safeFetch(url: string, options?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, options);
  } catch (err) {
    if (err instanceof TypeError && err.message.includes('Failed to fetch')) {
      throw new Error('Backend is offline. Please ensure the server is running on port 8000.');
    }
    throw err;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `HTTP error ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string' 
          ? errorJson.detail 
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // fallback to status text
    }
    throw new Error(errorDetail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  /**
   * Health check to detect backend status
   */
  async checkHealth(): Promise<HealthResponse> {
    const res = await safeFetch(`${API_BASE_URL}/health`);
    return handleResponse<HealthResponse>(res);
  },

  /**
   * Create a new session
   */
  async createSession(title?: string): Promise<CreateSessionResponse> {
    const res = await safeFetch(`${API_BASE_URL}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: title ? JSON.stringify({ title }) : undefined,
    });
    return handleResponse<CreateSessionResponse>(res);
  },

  /**
   * List all sessions ordered by most recently active first
   */
  async listSessions(): Promise<SessionSummary[]> {
    const res = await safeFetch(`${API_BASE_URL}/sessions`);
    return handleResponse<SessionSummary[]>(res);
  },

  /**
   * Get full turn history for a session
   */
  async getSessionTurns(sessionId: string): Promise<TurnItem[]> {
    const res = await safeFetch(
      `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/turns`
    );
    return handleResponse<TurnItem[]>(res);
  },

  /**
   * Rename a session
   */
  async updateSessionTitle(sessionId: string, title: string): Promise<void> {
    await safeFetch(
      `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/title`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      }
    );
  },

  /**
   * Submit a query to a session
   */
  async submitQuery(sessionId: string, question: string): Promise<QueryResponse> {
    const res = await safeFetch(
      `${API_BASE_URL}/sessions/${encodeURIComponent(sessionId)}/query`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      }
    );
    return handleResponse<QueryResponse>(res);
  },

  /**
   * Fetch full chunk text and metadata by chunk_id
   */
  async getChunk(chunkId: string): Promise<ChunkDetailResponse> {
    const res = await safeFetch(
      `${API_BASE_URL}/chunks/${encodeURIComponent(chunkId)}`
    );
    return handleResponse<ChunkDetailResponse>(res);
  },
};
