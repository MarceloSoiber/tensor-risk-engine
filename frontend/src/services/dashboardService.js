import { requestJson } from "./api.js";

export function fetchDashboardOverview({ days = 30, limit = 50 } = {}) {
  const params = new URLSearchParams({
    days: String(days),
    limit: String(limit),
  });
  return requestJson(`/api/v1/dashboard/overview?${params.toString()}`);
}
