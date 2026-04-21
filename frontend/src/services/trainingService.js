import { requestJson } from "./api.js";

export function startTrainingJob(payload) {
  return requestJson("/api/v1/training/jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export function listTrainingJobs() {
  return requestJson("/api/v1/training/jobs");
}
