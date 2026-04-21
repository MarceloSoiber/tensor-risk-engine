import { requestJson } from "./api.js";

export function fetchFraudCases() {
  return requestJson("/api/fraud/cases");
}
