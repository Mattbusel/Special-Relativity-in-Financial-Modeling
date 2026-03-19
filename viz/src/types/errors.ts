export type AppErrorCode =
  | 'WEBSOCKET_CONNECTION_ERROR'
  | 'WEBSOCKET_PARSE_ERROR'
  | 'RENDER_ERROR'
  | 'DATA_LOAD_ERROR';

export interface AppError {
  code: AppErrorCode;
  message: string;
  detail?: string;
  timestamp: number;
}

export function makeError(code: AppErrorCode, message: string, detail?: string): AppError {
  return { code, message, detail, timestamp: Date.now() };
}
