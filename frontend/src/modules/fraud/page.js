import { createButton } from "../../components/ui/button.js";
import { fetchAiAnalysisHistory, queryAiAnalysis } from "../../services/aiAnalysisService.js";
import { createElement } from "../../utils/dom.js";

const CATEGORY_OPTIONS = [
  ["", "All categories"],
  ["entertainment", "Entertainment"],
  ["food_dining", "Food dining"],
  ["gas_transport", "Gas transport"],
  ["grocery_net", "Grocery online"],
  ["grocery_pos", "Grocery POS"],
  ["health_fitness", "Health fitness"],
  ["home", "Home"],
  ["kids_pets", "Kids and pets"],
  ["misc_net", "Misc online"],
  ["misc_pos", "Misc POS"],
  ["personal_care", "Personal care"],
  ["shopping_net", "Shopping online"],
  ["shopping_pos", "Shopping POS"],
  ["travel", "Travel"],
];

export function createFraudPage() {
  const page = createElement("section", { className: "ai-analysis-screen" });
  const resultPanel = createResultPanel();
  const historyPanel = createHistoryPanel({
    onSelect: resultPanel.renderResult,
  });
  const queryPanel = createQueryPanel({
    onResult: (result) => {
      resultPanel.renderResult(result);
      historyPanel.load();
    },
  });

  page.append(queryPanel, resultPanel.element, historyPanel.element);
  historyPanel.load();
  return page;
}

function createQueryPanel({ onResult }) {
  const panel = createElement("article", { className: "panel ai-query" });
  panel.append(
    createElement("span", { className: "panel__eyebrow", text: "AI Analysis" }),
    createElement("h2", { className: "panel__title", text: "Ask the local model about transaction risk" }),
    createElement("p", {
      className: "panel__description",
      text: "Filter the transaction context, ask an analyst question, and review a saved AI-assisted summary.",
    }),
  );

  const source = createSelectField("Source", "source", [
    ["all", "All transactions"],
    ["analyzed", "Analyzed transactions"],
    ["historical", "Historical labeled data"],
  ]);
  const decision = createSelectField("Decision", "decision", [
    ["", "Any decision"],
    ["approve", "Approve"],
    ["review", "Review"],
    ["reject", "Reject"],
  ]);
  const category = createSelectField("Category", "category", CATEGORY_OPTIONS);
  const limit = createInputField("Transaction limit", "limit", "number", "25", { min: "1", max: "200", step: "1" });
  const question = createTextAreaField(
    "Question",
    "question",
    "Which patterns in this filtered transaction set deserve analyst attention?",
  );
  const status = createElement("div", { className: "ai-query__status", text: "Ready to analyze local transaction context." });

  const submitButton = createButton({
    label: "Run AI analysis",
    onClick: async () => {
      await submitAiAnalysis({ source, decision, category, limit, question, status, submitButton, onResult });
    },
  });

  const form = createElement("form", { className: "ai-query__form" });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitAiAnalysis({ source, decision, category, limit, question, status, submitButton, onResult });
  });

  form.append(
    createElement("div", {
      className: "ai-query__filters",
      children: [source.wrapper, decision.wrapper, category.wrapper, limit.wrapper],
    }),
    question.wrapper,
    createElement("div", { className: "ai-query__actions", children: [submitButton] }),
    status,
  );

  panel.append(form);
  return panel;
}

function createResultPanel() {
  const element = createElement("article", { className: "panel ai-result-panel" });
  const body = createElement("div", { className: "ai-result ai-result--empty" });
  body.append(
    createElement("span", { className: "ai-result__label", text: "No analysis selected" }),
    createElement("p", { text: "Run a question or select a saved analysis to inspect the model output." }),
  );

  element.append(
    createElement("span", { className: "panel__eyebrow", text: "Result" }),
    createElement("h2", { className: "panel__title", text: "AI-assisted findings" }),
    body,
  );

  return {
    element,
    renderResult(result) {
      body.className = "ai-result";
      body.replaceChildren(createAnalysisResult(result));
    },
    renderError(message) {
      body.className = "ai-result ai-result--error";
      body.replaceChildren(
        createElement("span", { className: "ai-result__label", text: "Analysis failed" }),
        createElement("p", { text: message }),
      );
    },
  };
}

function createHistoryPanel({ onSelect }) {
  const element = createElement("article", { className: "panel ai-history-panel" });
  const list = createElement("div", { className: "ai-history" });
  const reloadButton = createButton({
    label: "Reload history",
    variant: "secondary",
    onClick: () => load(),
  });

  element.append(
    createElement("div", {
      className: "ai-history-panel__header",
      children: [
        createElement("div", {
          children: [
            createElement("span", { className: "panel__eyebrow", text: "Saved analyses" }),
            createElement("h2", { className: "panel__title", text: "History" }),
          ],
        }),
        reloadButton,
      ],
    }),
    list,
  );

  async function load() {
    reloadButton.disabled = true;
    list.replaceChildren(createElement("article", { className: "alert-item", text: "Loading saved analyses..." }));
    try {
      const payload = await fetchAiAnalysisHistory({ limit: 20, offset: 0 });
      const analyses = payload.analyses ?? [];
      if (!analyses.length) {
        list.replaceChildren(createElement("article", { className: "alert-item", text: "No saved analyses yet." }));
        return;
      }
      list.replaceChildren(...analyses.map((item) => createHistoryItem(item, onSelect)));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unexpected history loading error.";
      list.replaceChildren(createElement("article", { className: "alert-item alert-item--error", text: message }));
    } finally {
      reloadButton.disabled = false;
    }
  }

  return { element, load };
}

async function submitAiAnalysis({ source, decision, category, limit, question, status, submitButton, onResult }) {
  const parsedLimit = Number.parseInt(limit.control.value, 10);
  const cleanedQuestion = question.control.value.trim();

  if (!cleanedQuestion || cleanedQuestion.length < 3) {
    status.textContent = "Question must be at least 3 characters.";
    status.className = "ai-query__status ai-query__status--error";
    question.control.focus();
    return;
  }

  submitButton.disabled = true;
  status.textContent = "Running local AI analysis...";
  status.className = "ai-query__status ai-query__status--loading";

  try {
    const result = await queryAiAnalysis({
      question: cleanedQuestion,
      filters: {
        source: source.control.value,
        decision: decision.control.value || null,
        category: category.control.value || null,
        limit: Number.isInteger(parsedLimit) ? parsedLimit : 25,
      },
    });
    status.textContent = "Analysis saved.";
    status.className = "ai-query__status ai-query__status--success";
    onResult(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected AI analysis error.";
    status.textContent = message;
    status.className = "ai-query__status ai-query__status--error";
  } finally {
    submitButton.disabled = false;
  }
}

function createAnalysisResult(result) {
  const summary = result.data_summary ?? {};
  return createElement("div", {
    className: "ai-result__content",
    children: [
      createElement("span", {
        className: "ai-result__label",
        text: `Analysis #${result.analysis_id ?? "--"} | ${result.model ?? "Local model"}`,
      }),
      createElement("h3", { className: "ai-result__answer", text: result.answer ?? "No answer returned." }),
      createListBlock("Key insights", result.insights ?? []),
      createListBlock("Recommended actions", result.recommended_actions ?? []),
      createSummaryGrid(summary),
      createTransactionContext(result.transactions ?? []),
    ],
  });
}

function createListBlock(title, items) {
  const listItems = items.length ? items : ["No items returned."];
  return createElement("section", {
    className: "ai-result-block",
    children: [
      createElement("h4", { text: title }),
      createElement("ul", {
        children: listItems.map((item) => createElement("li", { text: item })),
      }),
    ],
  });
}

function createSummaryGrid(summary) {
  const items = [
    ["Transactions", formatNumber(summary.transaction_count)],
    ["Filtered total", formatNumber(summary.filtered_total)],
    ["Average risk", formatNullablePercent(summary.average_risk_score)],
    ["Total amount", formatCurrency(summary.total_amount)],
  ];
  return createElement("section", {
    className: "ai-summary-grid",
    children: items.map(([label, value]) =>
      createElement("article", {
        className: "ai-summary-card",
        children: [
          createElement("span", { text: label }),
          createElement("strong", { text: value }),
        ],
      }),
    ),
  });
}

function createTransactionContext(transactions) {
  const table = createElement("div", { className: "data-table data-table--ai" });
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

  if (!transactions.length) {
    table.append(createElement("div", { className: "data-table__empty", text: "No transaction context was saved for this history item." }));
    return table;
  }

  for (const transaction of transactions.slice(0, 10)) {
    table.append(
      createElement("div", {
        className: "data-table__row",
        children: [
          createElement("span", { text: transaction.merchant ?? "Unknown merchant" }),
          createElement("span", { text: normalizeLabel(transaction.category) }),
          createElement("span", { text: formatCurrency(transaction.amount) }),
          createElement("span", { text: normalizeLabel(transaction.decision ?? labelKnownFraud(transaction.is_fraud)) }),
          createElement("span", { text: formatNullablePercent(transaction.risk_score) }),
        ],
      }),
    );
  }

  return table;
}

function createHistoryItem(item, onSelect) {
  const button = createElement("button", {
    className: "ai-history__item",
    attrs: { type: "button" },
  });
  button.append(
    createElement("span", { className: "ai-history__question", text: item.question }),
    createElement("span", {
      className: "ai-history__meta",
      text: `${formatNumber(item.data_summary?.transaction_count)} transactions | ${formatShortDateTime(item.created_at)}`,
    }),
  );
  button.addEventListener("click", () => onSelect(item));
  return button;
}

function createSelectField(label, name, options) {
  const control = createElement("select", {
    className: "transaction-input__control transaction-input__control--select",
    attrs: { name },
    children: options.map(([value, text]) => createElement("option", { text, attrs: { value } })),
  });
  const wrapper = createElement("label", {
    className: "transaction-input transaction-input--select",
    children: [
      createElement("span", { className: "training-field__label", text: label }),
      control,
    ],
  });
  return { wrapper, control };
}

function createInputField(label, name, type, value, attrs = {}) {
  const control = createElement("input", {
    className: "transaction-input__control",
    attrs: { name, type, value, ...attrs },
  });
  const wrapper = createElement("label", {
    className: "transaction-input",
    children: [
      createElement("span", { className: "training-field__label", text: label }),
      control,
    ],
  });
  return { wrapper, control };
}

function createTextAreaField(label, name, value) {
  const control = createElement("textarea", {
    className: "transaction-input__control ai-query__textarea",
    text: value,
    attrs: { name, rows: "6" },
  });
  const wrapper = createElement("label", {
    className: "transaction-input",
    children: [
      createElement("span", { className: "training-field__label", text: label }),
      control,
    ],
  });
  return { wrapper, control };
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value ?? 0);
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value ?? 0);
}

function formatNullablePercent(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  return new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 }).format(value);
}

function formatShortDateTime(value) {
  if (!value) {
    return "Unknown time";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
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
