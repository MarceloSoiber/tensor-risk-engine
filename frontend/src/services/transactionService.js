import { requestJson } from "./api.js";

export function fetchTransactions() {
  return requestJson("/api/v1/transactions");
}

export function startFraudTestImport(payload = {}) {
  return requestJson("/api/v1/transactions/import/fraud-test", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function fetchTransactionImportJob(jobId) {
  return requestJson(`/api/v1/transactions/import-jobs/${encodeURIComponent(jobId)}`);
}

export function analyzeTransaction(payload) {
  return requestJson("/api/v1/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}
