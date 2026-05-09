import { fetchDashboardOverview } from "../../services/dashboardService.js";
import { createElement } from "../../utils/dom.js";

export function createDashboardPage() {
  const page = createElement("section", { className: "dashboard-screen dashboard-screen--overview" });
  page.append(createDashboardState("Loading dashboard analytics..."));

  fetchDashboardOverview({ days: 30, limit: 50 })
    .then((overview) => {
      page.replaceChildren(createDashboardContent(overview));
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : "Unexpected dashboard error.";
      page.replaceChildren(createDashboardState(`Unable to load dashboard analytics: ${message}`, "error"));
    });

  return page;
}

function createDashboardContent(overview) {
  const fragment = document.createDocumentFragment();
  const kpis = overview.kpis ?? {};
  const transactions = overview.recent_transactions ?? [];

  if ((kpis.total_transactions ?? 0) === 0) {
    fragment.append(createDashboardState("No transaction data is available yet.", "empty"));
    return fragment;
  }

  const kpiSection = createElement("section", { className: "dashboard-kpis" });
  kpiSection.append(
    createMetricCard("Transactions", formatNumber(kpis.total_transactions), `${formatNumber(kpis.analyzed_transactions)} analyzed`),
    createMetricCard("Known fraud rate", formatPercent(kpis.known_fraud_rate), "From labeled historical data"),
    createMetricCard("Review or reject rate", formatPercent(kpis.reject_review_rate), "From model decisions"),
    createMetricCard("Average risk score", formatNullablePercent(kpis.average_risk_score), "Across scored transactions"),
  );

  const trendPanel = createPanel("Portfolio trend", "Fraud trend over time");
  trendPanel.append(createTrendChart(overview.fraud_trend ?? []));

  const decisionPanel = createPanel("Model decisions", "Decision breakdown");
  decisionPanel.append(createDecisionBars(overview.decision_breakdown ?? []));

  const categoryPanel = createPanel("Category risk", "Highest concentration");
  categoryPanel.append(createCategoryList(overview.category_risk ?? []));

  const hourlyPanel = createPanel("Hourly volume", "Activity by transaction hour");
  hourlyPanel.append(createHourlyChart(overview.hourly_activity ?? []));

  const tablePanel = createPanel("Transactions table", "Latest transactions");
  tablePanel.classList.add("dashboard-table");
  tablePanel.append(createTransactionTable(transactions));

  const alertsPanel = createPanel("Live alerts", "Real-time monitoring");
  alertsPanel.classList.add("dashboard-alerts");
  alertsPanel.append(createAlertList(overview.alerts ?? []));

  fragment.append(kpiSection, trendPanel, decisionPanel, categoryPanel, hourlyPanel, tablePanel, alertsPanel);
  return fragment;
}

function createDashboardState(message, variant = "loading") {
  return createElement("article", {
    className: `panel dashboard-state dashboard-state--${variant}`,
    children: [
      createElement("span", { className: "panel__eyebrow", text: "Dashboard" }),
      createElement("h3", { className: "panel__title", text: message }),
    ],
  });
}

function createPanel(eyebrow, title) {
  const panel = createElement("article", { className: "panel dashboard-panel" });
  panel.append(
    createElement("span", { className: "panel__eyebrow", text: eyebrow }),
    createElement("h3", { className: "panel__title", text: title }),
  );
  return panel;
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

function createTrendChart(rows) {
  const chart = createElement("div", { className: "analytics-chart analytics-chart--trend" });
  const maxTotal = Math.max(1, ...rows.map((row) => row.total ?? 0));
  for (const row of rows.slice(-14)) {
    chart.append(
      createElement("div", {
        className: "analytics-chart__bar",
        attrs: {
          title: `${row.date}: ${formatNumber(row.total)} transactions`,
          style: `--bar-size: ${Math.max(6, ((row.total ?? 0) / maxTotal) * 100)}%; --risk-size: ${Math.max(0, ((row.known_fraud_count ?? 0) / maxTotal) * 100)}%;`,
        },
        children: [createElement("span", { text: formatShortDate(row.date) })],
      }),
    );
  }
  return chart;
}

function createDecisionBars(rows) {
  const list = createElement("div", { className: "analytics-list" });
  const total = Math.max(1, rows.reduce((sum, row) => sum + (row.count ?? 0), 0));
  for (const row of rows) {
    list.append(createProgressRow(row.decision, row.count ?? 0, (row.count ?? 0) / total));
  }
  return list;
}

function createCategoryList(rows) {
  const list = createElement("div", { className: "analytics-list" });
  if (!rows.length) {
    list.append(createElement("article", { className: "alert-item", text: "No category data available." }));
    return list;
  }
  for (const row of rows) {
    list.append(createProgressRow(row.category, formatPercent(row.fraud_rate), row.fraud_rate ?? 0));
  }
  return list;
}

function createProgressRow(label, value, ratio) {
  return createElement("article", {
    className: "analytics-row",
    attrs: { style: `--row-size: ${Math.max(2, Math.min(100, ratio * 100))}%;` },
    children: [
      createElement("div", {
        className: "analytics-row__meta",
        children: [
          createElement("strong", { text: normalizeLabel(label) }),
          createElement("span", { text: String(value) }),
        ],
      }),
      createElement("div", { className: "analytics-row__track" }),
    ],
  });
}

function createHourlyChart(rows) {
  const chart = createElement("div", { className: "hourly-chart" });
  const maxTotal = Math.max(1, ...rows.map((row) => row.total ?? 0));
  for (const row of rows) {
    chart.append(
      createElement("span", {
        className: "hourly-chart__bar",
        attrs: {
          title: `${String(row.hour).padStart(2, "0")}:00 - ${formatNumber(row.total)} transactions`,
          style: `--hour-size: ${Math.max(4, ((row.total ?? 0) / maxTotal) * 100)}%;`,
        },
      }),
    );
  }
  return chart;
}

function createTransactionTable(rows) {
  const table = createElement("div", { className: "data-table data-table--dashboard" });
  table.append(
    createElement("div", {
      className: "data-table__row data-table__row--head",
      children: [
        createElement("span", { text: "Merchant" }),
        createElement("span", { text: "Category" }),
        createElement("span", { text: "Amount" }),
        createElement("span", { text: "Decision" }),
        createElement("span", { text: "Risk" }),
      ],
    }),
  );

  if (!rows.length) {
    table.append(createElement("div", { className: "data-table__empty", text: "No recent transactions found." }));
    return table;
  }

  for (const row of rows) {
    table.append(
      createElement("div", {
        className: "data-table__row",
        children: [
          createElement("span", { text: row.merchant }),
          createElement("span", { text: normalizeLabel(row.category) }),
          createElement("span", { text: formatCurrency(row.amount) }),
          createElement("span", { text: normalizeLabel(row.decision ?? labelKnownFraud(row.is_fraud)) }),
          createElement("span", { text: formatNullablePercent(row.risk_score) }),
        ],
      }),
    );
  }

  return table;
}

function createAlertList(alerts) {
  const list = createElement("div", { className: "alert-list" });
  for (const alert of alerts) {
    list.append(createElement("article", { className: "alert-item", text: alert }));
  }
  return list;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value ?? 0);
}

function formatPercent(value) {
  return new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 }).format(value ?? 0);
}

function formatNullablePercent(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  return formatPercent(value);
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value ?? 0);
}

function formatShortDate(value) {
  if (!value) {
    return "";
  }
  const [, month, day] = String(value).split("-");
  return `${month}/${day}`;
}

function normalizeLabel(value) {
  if (!value) {
    return "Unknown";
  }
  return String(value).replaceAll("_", " ");
}

function labelKnownFraud(value) {
  if (value === true) {
    return "Known fraud";
  }
  if (value === false) {
    return "Known genuine";
  }
  return "Unscored";
}
