import { createButton } from "../../components/ui/button.js";
import { analyzeTransaction } from "../../services/transactionService.js";
import { createElement, formatJson } from "../../utils/dom.js";

export function createTransactionsPage() {
  const page = createElement("section", { className: "page page--transactions" });
  const analysisPanel = createAnalysisPanel();
  const formPanel = createTransactionFormPanel({
    onAnalyzed: analysisPanel.renderResult,
  });

  page.append(
    formPanel,
    analysisPanel.element,
  );

  return page;
}

function createTransactionFormPanel({ onAnalyzed }) {
  const panel = createElement("article", { className: "panel transaction-entry" });
  panel.append(
    createElement("span", { className: "panel__eyebrow", text: "Transaction intake" }),
    createElement("h2", { className: "panel__title", text: "Add a transaction for risk analysis" }),
    createElement("p", {
      className: "panel__description",
      text: "Submit the transaction signals and receive the current fraud decision from the prediction API.",
    }),
  );

  const amount = createTransactionInput({
    label: "Amount",
    name: "amount",
    type: "number",
    min: "0",
    max: "1000000",
    step: "0.01",
    value: "250.00",
  });
  const velocity = createTransactionInput({
    label: "Transactions in the last hour",
    name: "velocity_1h",
    type: "number",
    min: "0",
    max: "500",
    step: "1",
    value: "2",
  });
  const merchantRisk = createTransactionInput({
    label: "Merchant risk",
    name: "merchant_risk",
    type: "number",
    min: "0",
    max: "1",
    step: "0.01",
    value: "0.2",
  });
  const deviceTrust = createTransactionInput({
    label: "Device trust",
    name: "device_trust",
    type: "number",
    min: "0",
    max: "1",
    step: "0.01",
    value: "0.9",
  });

  const fields = [amount, velocity, merchantRisk, deviceTrust];
  const status = createElement("pre", {
    className: "api-result transaction-status",
    text: "Ready to analyze a transaction.",
  });
  const submitButton = createButton({
    label: "Analyze transaction",
    onClick: async () => {
      await submitTransaction({
        fields,
        status,
        submitButton,
        onAnalyzed,
      });
    },
  });

  const form = createElement("form", { className: "transaction-form" });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitTransaction({
      fields,
      status,
      submitButton,
      onAnalyzed,
    });
  });

  form.append(
    createElement("div", {
      className: "transaction-form__grid",
      children: fields.map((field) => field.wrapper),
    }),
    createElement("div", {
      className: "transaction-form__actions",
      children: [submitButton],
    }),
    status,
  );
  panel.append(form);
  return panel;
}

function createAnalysisPanel() {
  const panel = createElement("article", { className: "panel transaction-analysis" });
  const resultCard = createElement("div", { className: "transaction-result transaction-result--idle" });
  const historyList = createElement("div", { className: "transaction-history" });
  const analyses = [];

  resultCard.append(
    createElement("span", { className: "transaction-result__label", text: "No analysis yet" }),
    createElement("strong", { className: "transaction-result__score", text: "--" }),
    createElement("p", {
      className: "panel__description",
      text: "Add a transaction to see the decision, score, and model reason.",
    }),
  );

  panel.append(
    createElement("span", { className: "panel__eyebrow", text: "Risk response" }),
    createElement("h2", { className: "panel__title", text: "Fraud analysis" }),
    resultCard,
    createElement("h3", { className: "transaction-history__title", text: "Session history" }),
    historyList,
  );

  const renderResult = ({ request, response }) => {
    analyses.unshift({ request, response, createdAt: new Date() });
    resultCard.className = `transaction-result transaction-result--${response.decision}`;
    resultCard.replaceChildren(
      createElement("span", { className: "transaction-result__label", text: response.decision }),
      createElement("strong", {
        className: "transaction-result__score",
        text: `${Math.round(response.risk_score * 100)}%`,
      }),
      createElement("p", {
        className: "panel__description",
        text: `Reason: ${response.reasons.join(", ")} · Model: ${response.model_version}`,
      }),
    );
    renderHistory(historyList, analyses);
  };

  renderHistory(historyList, analyses);
  return { element: panel, renderResult };
}

function createTransactionInput({ label, name, type, min, max, step, value }) {
  const wrapper = createElement("label", { className: "transaction-input" });
  const input = createElement("input", {
    attrs: {
      class: "transaction-input__control",
      name,
      type,
      min,
      max,
      step,
      value,
      required: true,
      inputmode: "decimal",
    },
  });

  wrapper.append(
    createElement("span", { className: "training-field__label", text: label }),
    input,
  );
  return { name, wrapper, input };
}

async function submitTransaction({ fields, status, submitButton, onAnalyzed }) {
  const payload = readTransactionPayload(fields);
  const validationError = validateTransactionPayload(payload);

  if (validationError) {
    status.textContent = validationError;
    return;
  }

  submitButton.disabled = true;
  status.textContent = "Analyzing transaction...";

  try {
    const response = await analyzeTransaction(payload);
    onAnalyzed?.({ request: payload, response });
    status.textContent = formatJson({
      message: "Transaction analyzed successfully.",
      request: payload,
      response,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected transaction analysis error.";
    status.textContent = `Unable to analyze transaction: ${message}`;
  } finally {
    submitButton.disabled = false;
  }
}

function readTransactionPayload(fields) {
  return fields.reduce((payload, field) => {
    const value = field.name === "velocity_1h" ? Number.parseInt(field.input.value, 10) : Number(field.input.value);
    return {
      ...payload,
      [field.name]: value,
    };
  }, {});
}

function validateTransactionPayload(payload) {
  if (!Number.isFinite(payload.amount) || payload.amount < 0 || payload.amount > 1_000_000) {
    return "Amount must be between 0 and 1,000,000.";
  }

  if (!Number.isInteger(payload.velocity_1h) || payload.velocity_1h < 0 || payload.velocity_1h > 500) {
    return "Transactions in the last hour must be an integer between 0 and 500.";
  }

  if (!isUnitInterval(payload.merchant_risk)) {
    return "Merchant risk must be between 0 and 1.";
  }

  if (!isUnitInterval(payload.device_trust)) {
    return "Device trust must be between 0 and 1.";
  }

  return null;
}

function renderHistory(container, analyses) {
  if (!analyses.length) {
    container.replaceChildren(
      createElement("p", { className: "panel__description", text: "Analyzed transactions will appear here." }),
    );
    return;
  }

  const items = analyses.slice(0, 5).map(({ request, response, createdAt }) =>
    createElement("article", {
      className: "transaction-history__item",
      children: [
        createElement("div", {
          className: "transaction-history__row",
          children: [
            createElement("strong", { text: `${formatCurrency(request.amount)} · ${response.decision}` }),
            createElement("span", { text: `${Math.round(response.risk_score * 100)}% risk` }),
          ],
        }),
        createElement("span", {
          text: `${request.velocity_1h} tx/hour · merchant ${request.merchant_risk} · device ${request.device_trust}`,
        }),
        createElement("small", { text: createdAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }),
      ],
    }),
  );

  container.replaceChildren(...items);
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

function isUnitInterval(value) {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}
