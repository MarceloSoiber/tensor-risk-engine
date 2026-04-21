import { createAuthPage } from "../../modules/auth/page.js";
import { createDashboardPage } from "../../modules/dashboard/page.js";
import { createFraudPage } from "../../modules/fraud/page.js";
import { createModelMonitoringPage } from "../../modules/model-monitoring/page.js";
import { createTransactionsPage } from "../../modules/transactions/page.js";

const ROUTE_DEFINITIONS = [
  { hash: "#/dashboard", label: "Dashboard", render: createDashboardPage },
  { hash: "#/auth", label: "Auth", render: createAuthPage },
  { hash: "#/transactions", label: "Transactions", render: createTransactionsPage },
  { hash: "#/fraud", label: "Fraud", render: createFraudPage },
  { hash: "#/model-monitoring", label: "Model Monitoring", render: createModelMonitoringPage },
];

export function createRoutes(context) {
  return ROUTE_DEFINITIONS.map((route) => ({
    ...route,
    render: () => route.render(context),
  }));
}

export function resolveRoute(routes, hash) {
  return routes.find((route) => route.hash === hash) ?? routes[0];
}
