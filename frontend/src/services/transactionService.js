import { requestJson } from "./api.js";

export function fetchTransactions() {
  return requestJson("/api/transactions");
}
