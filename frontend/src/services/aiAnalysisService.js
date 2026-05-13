import { requestJson } from "./api.js";

export function queryAiAnalysis(payload) {
  return requestJson("/api/v1/ai-analysis/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function fetchAiAnalysisHistory({ limit = 20, offset = 0 } = {}) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return requestJson(`/api/v1/ai-analysis/history?${params.toString()}`);
}
