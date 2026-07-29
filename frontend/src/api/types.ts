/**
 * API Data Models and Contracts
 */

export interface HealthResponse {
  status: string;
}

export interface CreateSessionResponse {
  session_id: string;
  created_at: string;
}

export interface SessionSummary {
  session_id: string;
  created_at: string;
  title: string | null;
  last_active: string;
}

export interface ChunkSource {
  company?: string;
  ticker?: string;
  year?: string;
  section?: string;
  section_name?: string;
  chunk_id?: string;
  chunk_type?: string;
  table_name?: string;
  text?: string;
  [key: string]: unknown;
}

export interface QueryRequest {
  question: string;
}

export interface QueryResponse {
  raw_question: string;
  resolved_question: string;
  question_was_resolved: boolean;
  final_answer: string;
  cache_hit: boolean;
  chunk_sources: ChunkSource[];
  error_message: string | null;
  /** Optional field reserved for future backend contract (e.g. "hallucination_exhausted") */
  answer_source?: string;
}

/**
 * DB Row mapping returned by GET /sessions/{session_id}/turns
 */
export interface TurnItem {
  turn_id: number;
  session_id: string;
  turn_number: number;
  raw_question: string;
  resolved_question: string;
  route: string | null;
  companies_json: string | null;
  final_answer: string | null;
  error_message: string | null;
  created_at: string;
}

export interface ChunkDetailResponse {
  chunk_id: string;
  text: string;
  metadata: {
    company?: string;
    ticker?: string;
    year?: string;
    section_name?: string;
    table_name?: string;
    chunk_type?: string;
    block_idx?: number;
    parent_chunk_id?: string;
    [key: string]: any;
  };
}
