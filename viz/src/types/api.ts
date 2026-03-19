// Auto-sync: update when src/web_api.rs structs change

/**
 * Matches InferRequest in src/web_api.rs.
 *
 * Sent by the client to POST /api/v1/infer, POST /api/v1/stream,
 * or over the WebSocket at /api/v1/ws.
 *
 * Note: the Rust struct uses HashMap<String, String> for metadata, so
 * Record<string, string> is the correct TypeScript equivalent.
 */
export interface InferRequest {
  /** The input prompt text to be processed by the pipeline. */
  prompt: string;
  /** Optional client-supplied session identifier; one is generated if absent. */
  session_id?: string;
  /** Arbitrary key-value metadata forwarded through the pipeline. */
  metadata?: Record<string, string>;
  /** If true, the client would like a streaming (WebSocket) response. */
  stream: boolean;
}

/**
 * Processing state of an inference request.
 * Matches RequestStatus in src/web_api.rs (#[serde(rename_all = "lowercase")]).
 */
export type RequestStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'timeout';

/**
 * Matches InferResponse in src/web_api.rs.
 *
 * Returned by POST /api/v1/infer, GET /api/v1/status/{id},
 * GET /api/v1/result/{id}, and sent over the WebSocket.
 */
export interface InferResponse {
  /** Unique identifier assigned to this inference request. */
  request_id: string;
  /** Current processing status of the request. */
  status: RequestStatus;
  /** Final inference result, present only when status is 'completed'. */
  result?: string;
  /** Error description, present only when status is 'failed'. */
  error?: string;
}

/**
 * Matches ServerConfig in src/web_api.rs.
 */
export interface ServerConfig {
  host: string;
  port: number;
  max_request_size: number;
  timeout_seconds: number;
}

/**
 * Shape of a single item in the GET /api/v1/requests paginated list.
 */
export interface RequestListItem {
  request_id: string;
  status: RequestStatus;
}

/**
 * Response body for GET /api/v1/requests.
 */
export interface RequestListResponse {
  total: number;
  offset: number;
  limit: number;
  items: RequestListItem[];
}

/**
 * Response body for GET /api/v1/explain/{request_id}.
 */
export interface ExplainResponse {
  request_id: string;
  regime: 'TIMELIKE' | 'SPACELIKE' | 'LIGHTLIKE';
  beta: number;
  gamma: number;
  ds2: number;
  interpretation: string;
}

/**
 * Response body for GET /health.
 */
export interface HealthResponse {
  status: 'healthy';
  version: string;
}
