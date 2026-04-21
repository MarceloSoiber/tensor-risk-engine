import { createApiCheckForm } from "../../components/forms/api-check-form.js";
import { createHealthCard } from "../../components/charts/health-card.js";
import { createButton } from "../../components/ui/button.js";
import { createElement } from "../../utils/dom.js";

export function createDashboardPage(context) {
  const page = createElement("section", { className: "dashboard-screen" });

  const layout = createElement("div", { className: "dashboard-layout" });

  const sidebar = createElement("aside", { className: "dashboard-sidebar" });
  sidebar.append(
    createElement("span", { className: "panel__eyebrow", text: "Sidebar" }),
    createElement("h3", { className: "panel__title", text: "Collapsible navigation" }),
    createElement("p", {
      className: "panel__description",
      text: "Keep shortcuts, filters, and status items here so the primary canvas stays clear.",
    }),
    createElement("nav", { className: "dashboard-sidebar__nav", attrs: { "aria-label": "Dashboard shortcuts" } }),
  );

  const sidebarNav = sidebar.querySelector(".dashboard-sidebar__nav");
  sidebarNav?.append(
    createButton({ label: "Dashboard", href: "#/dashboard", variant: "secondary" }),
    createButton({ label: "Transactions", href: "#/transactions", variant: "secondary" }),
    createButton({ label: "Fraud", href: "#/fraud", variant: "secondary" }),
    createButton({ label: "Model monitoring", href: "#/model-monitoring", variant: "secondary" }),
  );

  const workspace = createElement("section", { className: "dashboard-workspace" });

  const kpis = createElement("section", { className: "dashboard-kpis" });
  kpis.append(
    createMetricCard("Daily fraud rate", "0.42%", "Stable versus the previous day"),
    createMetricCard("Flagged transactions", "128", "+12 in the last hour"),
    createMetricCard("Model drift", "Low", "No retraining required"),
    createMetricCard("Alert backlog", "7", "2 require immediate review"),
  );

  const chartPanel = createElement("article", { className: "panel dashboard-chart" });
  chartPanel.append(
    createElement("span", { className: "panel__eyebrow", text: "KPI + primary chart" }),
    createElement("h3", { className: "panel__title", text: "Fraud trend over time" }),
    createElement("div", { className: "chart-placeholder" }),
  );

  const tablePanel = createElement("article", { className: "panel dashboard-table" });
  tablePanel.append(
    createElement("span", { className: "panel__eyebrow", text: "Transactions table" }),
    createElement("h3", { className: "panel__title", text: "Detailed transaction list" }),
    createTransactionTable(),
  );

  const alertsPanel = createElement("article", { className: "panel dashboard-alerts" });
  alertsPanel.append(
    createElement("span", { className: "panel__eyebrow", text: "Live alerts" }),
    createElement("h3", { className: "panel__title", text: "Real-time monitoring" }),
    createAlertList(),
  );

  const analyzerPanel = createElement("article", { className: "panel dashboard-analyzer" });
  const healthCard = createHealthCard({
    title: "Backend health",
    description: "Run the health check to validate the API connection.",
  });

  analyzerPanel.append(
    createElement("span", { className: "panel__eyebrow", text: "Fraud analyzer" }),
    createElement("h3", { className: "panel__title", text: "Form and result" }),
    createApiCheckForm({ onStateChange: healthCard.update }),
    healthCard.element,
  );

  workspace.append(kpis, chartPanel, tablePanel, alertsPanel, analyzerPanel);
  layout.append(sidebar, workspace);
  page.append(layout);
  return page;
}

function createMetricCard(title, value, detail) {
  const card = createElement("article", { className: "metric-card" });
  card.append(
    createElement("span", { className: "metric-card__title", text: title }),
    createElement("strong", { className: "metric-card__value", text: value }),
    createElement("span", { className: "metric-card__detail", text: detail }),
  );
  return card;
}

function createTransactionTable() {
  const table = createElement("div", { className: "data-table" });
  const rows = [
    ["TX-1042", "Card present", "$1,240.00", "Low"],
    ["TX-1043", "Card not present", "$340.00", "Medium"],
    ["TX-1044", "Recurring", "$84.90", "Low"],
    ["TX-1045", "Velocity spike", "$2,980.00", "High"],
  ];

  table.append(
    createElement("div", {
      className: "data-table__row data-table__row--head",
      children: [
        createElement("span", { text: "Transaction" }),
        createElement("span", { text: "Type" }),
        createElement("span", { text: "Amount" }),
        createElement("span", { text: "Risk" }),
      ],
    }),
  );

  for (const row of rows) {
    table.append(
      createElement("div", {
        className: "data-table__row",
        children: [
          createElement("span", { text: row[0] }),
          createElement("span", { text: row[1] }),
          createElement("span", { text: row[2] }),
          createElement("span", { text: row[3] }),
        ],
      }),
    );
  }

  return table;
}

function createAlertList() {
  const list = createElement("div", { className: "alert-list" });
  const alerts = [
    "Unusual purchase pattern detected in the last 15 minutes.",
    "Two high-risk transactions require manual review.",
    "Training pipeline completed with no drift escalation.",
  ];

  for (const alert of alerts) {
    list.append(createElement("article", { className: "alert-item", text: alert }));
  }

  return list;
}
