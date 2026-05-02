import { createButton } from "../../components/ui/button.js";
import { analyzeTransaction } from "../../services/transactionService.js";
import { listTrainingJobs } from "../../services/trainingService.js";
import { createElement, formatJson } from "../../utils/dom.js";

const TRANSACTION_FIELDS = [
  {
    label: "Amount",
    name: "amount",
    type: "number",
    min: "0",
    max: "1000000",
    step: "0.01",
    value: "84.35",
    parser: "float",
    validator: (value) => isNumberInRange(value, 0, 1_000_000),
    error: "Amount must be between 0 and 1,000,000.",
  },
  {
    label: "Transaction time",
    name: "transaction_datetime",
    type: "datetime-local",
    value: getLocalDateTimeValue(),
    parser: "datetime",
    validator: (value) => typeof value === "string" && value.length > 0 && !Number.isNaN(new Date(value).getTime()),
    error: "Transaction time must be a valid date and time.",
  },
  {
    label: "Merchant",
    name: "merchant",
    type: "text",
    value: "Northside Market",
    parser: "text",
    validator: (value) => isTextInRange(value, 2, 96),
    error: "Merchant must be 2 to 96 characters.",
  },
  {
    label: "Category",
    name: "category",
    type: "select",
    value: "grocery_pos",
    options: [
      ["grocery_pos", "Grocery POS"],
      ["shopping_net", "Shopping online"],
      ["travel", "Travel"],
      ["gas_transport", "Gas transport"],
      ["food_dining", "Food dining"],
      ["entertainment", "Entertainment"],
      ["health_fitness", "Health fitness"],
    ],
    parser: "text",
    validator: (value) => isTextInRange(value, 2, 48),
    error: "Category is required.",
  },
  {
    label: "Gender",
    name: "gender",
    type: "select",
    value: "F",
    options: [
      ["F", "F"],
      ["M", "M"],
    ],
    parser: "text",
    validator: (value) => ["F", "M"].includes(value),
    error: "Gender must be F or M.",
  },
  {
    label: "State",
    name: "state",
    type: "text",
    value: "CA",
    attrs: { maxlength: "2" },
    parser: "uppercase",
    validator: (value) => /^[A-Z]{2}$/.test(value),
    error: "State must be a two-letter code.",
  },
  {
    label: "Job",
    name: "job",
    type: "text",
    value: "Data analyst",
    parser: "text",
    validator: (value) => isTextInRange(value, 2, 96),
    error: "Job must be 2 to 96 characters.",
  },
  {
    label: "City population",
    name: "city_population",
    type: "number",
    min: "0",
    max: "50000000",
    step: "1",
    value: "884363",
    parser: "integer",
    validator: (value) => Number.isInteger(value) && value >= 0 && value <= 50_000_000,
    error: "City population must be an integer between 0 and 50,000,000.",
  },
  {
    label: "Customer latitude",
    name: "customer_latitude",
    type: "number",
    min: "-90",
    max: "90",
    step: "0.000001",
    value: "37.774929",
    parser: "float",
    validator: (value) => isNumberInRange(value, -90, 90),
    error: "Customer latitude must be between -90 and 90.",
  },
  {
    label: "Customer longitude",
    name: "customer_longitude",
    type: "number",
    min: "-180",
    max: "180",
    step: "0.000001",
    value: "-122.419416",
    parser: "float",
    validator: (value) => isNumberInRange(value, -180, 180),
    error: "Customer longitude must be between -180 and 180.",
  },
  {
    label: "Merchant latitude",
    name: "merchant_latitude",
    type: "number",
    min: "-90",
    max: "90",
    step: "0.000001",
    value: "37.783333",
    parser: "float",
    validator: (value) => isNumberInRange(value, -90, 90),
    error: "Merchant latitude must be between -90 and 90.",
  },
  {
    label: "Merchant longitude",
    name: "merchant_longitude",
    type: "number",
    min: "-180",
    max: "180",
    step: "0.000001",
    value: "-122.416667",
    parser: "float",
    validator: (value) => isNumberInRange(value, -180, 180),
    error: "Merchant longitude must be between -180 and 180.",
  },
  {
    label: "Transactions last hour",
    name: "transactions_last_hour",
    type: "number",
    min: "0",
    max: "500",
    step: "1",
    value: "1",
    parser: "integer",
    validator: (value) => Number.isInteger(value) && value >= 0 && value <= 500,
    error: "Transactions last hour must be an integer between 0 and 500.",
  },
  {
    label: "Transactions last 24h",
    name: "transactions_last_24h",
    type: "number",
    min: "0",
    max: "5000",
    step: "1",
    value: "8",
    parser: "integer",
    validator: (value) => Number.isInteger(value) && value >= 0 && value <= 5000,
    error: "Transactions last 24h must be an integer between 0 and 5,000.",
  },
  {
    label: "Average amount 24h",
    name: "average_amount_24h",
    type: "number",
    min: "0",
    max: "1000000",
    step: "0.01",
    value: "63.20",
    parser: "float",
    validator: (value) => isNumberInRange(value, 0, 1_000_000),
    error: "Average amount 24h must be between 0 and 1,000,000.",
  },
];

const SAMPLE_SCENARIOS = [
  {
    label: "Normal grocery purchase",
    values: {
      amount: "84.35",
      merchant: "Northside Market",
      category: "grocery_pos",
      gender: "F",
      state: "CA",
      job: "Data analyst",
      city_population: "884363",
      customer_latitude: "37.774929",
      customer_longitude: "-122.419416",
      merchant_latitude: "37.783333",
      merchant_longitude: "-122.416667",
      transactions_last_hour: "1",
      transactions_last_24h: "8",
      average_amount_24h: "63.20",
    },
  },
  {
    label: "High-value online shopping",
    values: {
      amount: "4280.99",
      merchant: "Vertex Electronics Online",
      category: "shopping_net",
      gender: "M",
      state: "NY",
      job: "Civil engineer",
      city_population: "8804190",
      customer_latitude: "40.712776",
      customer_longitude: "-74.005974",
      merchant_latitude: "40.758896",
      merchant_longitude: "-73.985130",
      transactions_last_hour: "6",
      transactions_last_24h: "27",
      average_amount_24h: "214.40",
    },
  },
  {
    label: "Travel distance anomaly",
    values: {
      amount: "917.45",
      merchant: "Canopy Suites Airport",
      category: "travel",
      gender: "F",
      state: "FL",
      job: "Registered nurse",
      city_population: "312181",
      customer_latitude: "47.606209",
      customer_longitude: "-122.332071",
      merchant_latitude: "25.761681",
      merchant_longitude: "-80.191788",
      transactions_last_hour: "3",
      transactions_last_24h: "18",
      average_amount_24h: "78.10",
    },
  },
];

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
    createElement("span", { className: "panel__eyebrow", text: "Simulator" }),
    createElement("h2", { className: "panel__title", text: "Card transaction" }),
  );

  const fields = TRANSACTION_FIELDS.map(createTransactionInput);
  const trainingSelect = createTrainingSelect();
  const status = createElement("pre", {
    className: "api-result transaction-status",
    text: "Ready",
  });
  const submitButton = createButton({
    label: "Analyze transaction",
    onClick: async () => {
      await submitTransaction({
        fields,
        trainingSelect,
        status,
        submitButton,
        onAnalyzed,
      });
    },
  });

  const scenarioButtons = createElement("div", {
    className: "transaction-scenarios",
    children: SAMPLE_SCENARIOS.map((scenario) => {
      const button = createButton({
        label: scenario.label,
        variant: "secondary",
        onClick: () => applyScenario(fields, scenario),
      });
      button.classList.add("transaction-scenarios__button");
      return button;
    }),
  });

  const form = createElement("form", { className: "transaction-form" });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitTransaction({
      fields,
      trainingSelect,
      status,
      submitButton,
      onAnalyzed,
    });
  });

  form.append(
    scenarioButtons,
    createElement("div", {
      className: "transaction-form__grid",
      children: [trainingSelect.wrapper, ...fields.map((field) => field.wrapper)],
    }),
    createElement("div", {
      className: "transaction-form__actions",
      children: [submitButton],
    }),
    status,
  );
  panel.append(form);
  hydrateTrainingSelect(trainingSelect, status);
  return panel;
}

function createAnalysisPanel() {
  const panel = createElement("article", { className: "panel transaction-analysis" });
  const resultCard = createElement("div", { className: "transaction-result transaction-result--idle" });
  const historyList = createElement("div", { className: "transaction-history" });
  const analyses = [];

  resultCard.append(
    createElement("span", { className: "transaction-result__label", text: "Awaiting analysis" }),
    createElement("strong", { className: "transaction-result__score", text: "--" }),
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

function createTransactionInput(field) {
  const { label, name, type, min, max, step, value, options, attrs = {} } = field;
  const wrapper = createElement("label", {
    className: type === "select" ? "transaction-input transaction-input--select" : "transaction-input",
  });
  const input = type === "select"
    ? createElement("select", {
      attrs: {
        class: "transaction-input__control transaction-input__control--select",
        name,
        required: true,
      },
      children: options.map(([optionValue, optionLabel]) =>
        createElement("option", {
          text: optionLabel,
          attrs: {
            value: optionValue,
            selected: optionValue === value ? true : undefined,
          },
        }),
      ),
    })
    : createElement("input", {
      attrs: {
        class: "transaction-input__control",
        name,
        type,
        min,
        max,
        step,
        value,
        required: true,
        inputmode: type === "number" ? "decimal" : undefined,
        ...attrs,
      },
    });

  wrapper.append(
    createElement("span", { className: "training-field__label", text: label }),
    input,
  );
  return { ...field, wrapper, input };
}

async function submitTransaction({ fields, trainingSelect, status, submitButton, onAnalyzed }) {
  const payload = readTransactionPayload(fields);
  const validationError = validateTransactionPayload(payload);

  if (validationError) {
    status.textContent = validationError;
    return;
  }

  submitButton.disabled = true;
  status.textContent = "Analyzing transaction...";

  try {
    payload.training_job_id = trainingSelect?.getValue() || null;
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

function createTrainingSelect() {
  const wrapper = createElement("div", { className: "transaction-input dataset-picker" });
  const button = createElement("button", {
    className: "dataset-picker__button",
    text: "Loading trained models...",
    attrs: {
      type: "button",
      "aria-haspopup": "listbox",
      "aria-expanded": "false",
      disabled: true,
    },
  });
  const list = createElement("div", {
    className: "dataset-picker__list",
    attrs: { role: "listbox", hidden: true },
  });
  let entries = [];
  let selectedJobId = "";

  button.addEventListener("click", () => {
    const isOpen = button.getAttribute("aria-expanded") === "true";
    setOpen(!isOpen);
  });

  document.addEventListener("click", (event) => {
    if (!wrapper.contains(event.target)) {
      setOpen(false);
    }
  });

  wrapper.append(
    createElement("span", { className: "training-field__label", text: "Training model" }),
    button,
    list,
  );

  return {
    wrapper,
    setEntries: (nextEntries) => {
      entries = nextEntries;
      selectedJobId = "";
      button.disabled = false;
      setOpen(false);
      renderOptions();
      button.textContent = "Latest baseline (automatic)";
      onChange();
    },
    setDisabledWithMessage: (message) => {
      entries = [];
      selectedJobId = "";
      button.disabled = true;
      button.textContent = message;
      list.replaceChildren();
      setOpen(false);
    },
    getValue: () => {
      const value = selectedJobId.trim();
      return value.length > 0 ? value : null;
    },
  };

  function renderOptions() {
    const automaticOption = createElement("button", {
      className: `dataset-picker__option${selectedJobId === "" ? " is-selected" : ""}`,
      attrs: { type: "button", role: "option", "aria-selected": selectedJobId === "" ? "true" : "false" },
    });
    automaticOption.append(
      createElement("span", { text: "Latest baseline (automatic)" }),
      createElement("small", { text: "Newest succeeded baseline model" }),
    );
    automaticOption.addEventListener("click", () => {
      selectedJobId = "";
      button.textContent = "Latest baseline (automatic)";
      renderOptions();
      onChange();
      setOpen(false);
    });

    const modelOptions = entries.map((entry) => {
      const option = createElement("button", {
        className: `dataset-picker__option${selectedJobId === entry.jobId ? " is-selected" : ""}`,
        attrs: {
          type: "button",
          role: "option",
          "aria-selected": selectedJobId === entry.jobId ? "true" : "false",
        },
      });
      option.append(
        createElement("span", { text: entry.label }),
        createElement("small", { text: entry.meta }),
      );
      option.addEventListener("click", () => {
        selectedJobId = entry.jobId;
        button.textContent = entry.label;
        renderOptions();
        onChange();
        setOpen(false);
      });
      return option;
    });

    list.replaceChildren(automaticOption, ...modelOptions);
  }

  function setOpen(isOpen) {
    button.setAttribute("aria-expanded", String(isOpen));
    list.hidden = !isOpen;
    wrapper.classList.toggle("is-open", isOpen);
  }

  function onChange() {
    const selected = entries.find((entry) => entry.jobId === selectedJobId) ?? null;
    if (selected) {
      button.textContent = selected.label;
    }
  }
}

async function hydrateTrainingSelect(trainingSelect, status) {
  try {
    const payload = await listTrainingJobs();
    const jobs = (payload?.jobs ?? [])
      .filter((job) => job.model_type === "baseline" && job.status === "succeeded")
      .map((job) => ({
        jobId: job.job_id,
        label: `${job.run_name || job.job_id} · ${job.job_id.slice(0, 8)}`,
        meta: `Updated ${formatDateTime(job.updated_at)}`,
      }));

    if (!jobs.length) {
      trainingSelect.setDisabledWithMessage("No succeeded baseline training found");
      return;
    }

    trainingSelect.setEntries(jobs);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error while loading trainings.";
    trainingSelect.setDisabledWithMessage("Unable to load trained models");
    status.textContent = `Unable to load training models: ${message}`;
  }
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "recently";
  }
  return date.toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function readTransactionPayload(fields) {
  return fields.reduce((payload, field) => {
    const value = parseFieldValue(field);
    return {
      ...payload,
      [field.name]: value,
    };
  }, {});
}

function validateTransactionPayload(payload) {
  for (const field of TRANSACTION_FIELDS) {
    if (!field.validator(payload[field.name])) {
      return field.error;
    }
  }

  if (payload.transactions_last_hour > payload.transactions_last_24h) {
    return "Transactions last hour cannot exceed transactions last 24h.";
  }

  return null;
}

function renderHistory(container, analyses) {
  if (!analyses.length) {
    container.replaceChildren(
      createElement("p", { className: "panel__description", text: "No session history" }),
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
            createElement("strong", { text: `${request.merchant} · ${request.category}` }),
            createElement("span", { text: `${Math.round(response.risk_score * 100)}% risk` }),
          ],
        }),
        createElement("span", {
          text: `${formatCurrency(request.amount)} · ${response.decision} · ${formatHistoryDate(request.transaction_datetime)}`,
        }),
        createElement("small", { text: `Analyzed ${createdAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` }),
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

function applyScenario(fields, scenario) {
  const values = {
    ...scenario.values,
    transaction_datetime: getLocalDateTimeValue(),
  };

  for (const field of fields) {
    if (Object.hasOwn(values, field.name)) {
      field.input.value = values[field.name];
    }
  }
}

function parseFieldValue(field) {
  const rawValue = field.input.value.trim();

  if (field.parser === "integer") {
    return Number.parseInt(rawValue, 10);
  }

  if (field.parser === "float") {
    return Number.parseFloat(rawValue);
  }

  if (field.parser === "uppercase") {
    return rawValue.toUpperCase();
  }

  return rawValue;
}

function isNumberInRange(value, min, max) {
  return Number.isFinite(value) && value >= min && value <= max;
}

function isTextInRange(value, minLength, maxLength) {
  return typeof value === "string" && value.trim().length >= minLength && value.trim().length <= maxLength;
}

function getLocalDateTimeValue(date = new Date()) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return localDate.toISOString().slice(0, 16);
}

function formatHistoryDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
