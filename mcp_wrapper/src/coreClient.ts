/**
 * Core Client — typed, resilient HTTP client to the NEXUS core service.
 *
 * This is the ONLY place that talks to the underlying API. Every tool
 * handler goes through here, never fetch()/axios directly — centralizing
 * auth, timeouts, retries and error typing in one place.
 */

import { request } from "undici";
import {
  CORE_BASE_URL,
  CORE_TIMEOUT_MS,
  CORE_MAX_RETRIES,
  CORE_RETRY_BASE_DELAY_MS,
} from "./constants.js";
import { CoreServiceError, type CoreErrorBody } from "./types.js";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryable(status: number | null): boolean {
  if (status === null) return true; // network-level failure
  return status === 429 || status >= 500;
}

export async function callCore<TResponse>(
  path: string,
  method: "GET" | "POST",
  body?: unknown,
): Promise<TResponse> {
  const url = `${CORE_BASE_URL}${path}`;
  const apiKey = process.env.NEXUS_CORE_API_KEY;

  let lastError: unknown;

  for (let attempt = 0; attempt <= CORE_MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), CORE_TIMEOUT_MS);

    try {
      // --- NEXUS PATCH mcp_tool_grounding_similarity_search ---
      // The core service authenticates with APIKeyHeader("X-API-Key")
      // (see core/similarity_search_api_api.py, _require_api_key), not a
      // Bearer token -- sending "authorization: Bearer" got a 401 even
      // once the route path itself was fixed.
      const res = await request(url, {
        method,
        headers: {
          "content-type": "application/json",
          ...(apiKey ? { "x-api-key": apiKey } : {}),
        },
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeout);
      const status = res.statusCode;
      const json = (await res.body.json()) as unknown;

      if (status >= 200 && status < 300) {
        return json as TResponse;
      }

      const errBody = json as CoreErrorBody;
      const message =
        errBody?.error?.message ?? `Core service returned HTTP ${status}`;
      const code = errBody?.error?.code ?? "UNKNOWN_ERROR";
      const requestId = errBody?.error?.request_id;

      if (isRetryable(status) && attempt < CORE_MAX_RETRIES) {
        await sleep(CORE_RETRY_BASE_DELAY_MS * 2 ** attempt);
        continue;
      }

      throw new CoreServiceError(message, status, code, requestId);
    } catch (err) {
      clearTimeout(timeout);
      lastError = err;

      if (err instanceof CoreServiceError) throw err;

      const isAbort = err instanceof Error && err.name === "AbortError";
      if (attempt < CORE_MAX_RETRIES) {
        await sleep(CORE_RETRY_BASE_DELAY_MS * 2 ** attempt);
        continue;
      }
      throw new CoreServiceError(
        isAbort
          ? `Core service timed out after ${CORE_TIMEOUT_MS}ms`
          : `Core service unreachable: ${(err as Error).message}`,
        null,
        isAbort ? "TIMEOUT" : "NETWORK_ERROR",
      );
    }
  }

  throw lastError;
}
