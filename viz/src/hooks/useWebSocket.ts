import { useCallback, useEffect, useRef, useState } from 'react';

export interface UseWebSocketReturn<T = unknown> {
  data: T | null;
  isConnected: boolean;
  error: string | null;
  send: (payload: unknown) => void;
}

const BASE_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 30_000;
const BACKOFF_FACTOR = 2;

/**
 * Connects to a WebSocket endpoint with exponential-backoff reconnection.
 * Messages are expected to be JSON; non-JSON text frames are ignored.
 *
 * WS protocol for ws://localhost:8080/api/v1/ws:
 *   Client sends: { prompt: string, session_id?: string, metadata?: object, stream: true }
 *   Server sends: { request_id, status, result?, error? }
 */
export function useWebSocket<T = unknown>(
  url: string,
  enabled = true,
): UseWebSocketReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(BASE_BACKOFF_MS);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      // Remove handlers before closing to suppress spurious onclose reconnect
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current || !enabled) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        backoffRef.current = BASE_BACKOFF_MS;
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (evt: MessageEvent<string>) => {
        if (!mountedRef.current) return;
        try {
          const parsed = JSON.parse(evt.data) as T;
          setData(parsed);
        } catch {
          // Non-JSON frame — ignore silently
        }
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setError('WebSocket connection error');
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        wsRef.current = null;
        setIsConnected(false);

        // Exponential backoff reconnect
        const delay = Math.min(backoffRef.current, MAX_BACKOFF_MS);
        backoffRef.current = Math.min(backoffRef.current * BACKOFF_FACTOR, MAX_BACKOFF_MS);
        reconnectTimerRef.current = setTimeout(connect, delay);
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open WebSocket');
      const delay = Math.min(backoffRef.current, MAX_BACKOFF_MS);
      backoffRef.current = Math.min(backoffRef.current * BACKOFF_FACTOR, MAX_BACKOFF_MS);
      reconnectTimerRef.current = setTimeout(connect, delay);
    }
  }, [url, enabled]);

  useEffect(() => {
    mountedRef.current = true;
    if (enabled) {
      connect();
    }
    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, [connect, cleanup, enabled]);

  const send = useCallback((payload: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  }, []);

  return { data, isConnected, error, send };
}
