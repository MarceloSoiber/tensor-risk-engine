import { requestJson } from "./api.js";

export function fetchTransactions() {
  return requestJson("/api/transactions");
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
