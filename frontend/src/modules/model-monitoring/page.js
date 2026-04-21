import { createButton } from "../../components/ui/button.js";
import { createElement, createTextSection, formatJson } from "../../utils/dom.js";
import { useStartTrainingJob, useTrainingJobs } from "../../hooks/use-training-jobs.js";

export function createModelMonitoringPage() {
  const page = createElement("section", { className: "page page--model-monitoring" });

  page.append(
    createTextSection({
      eyebrow: "Model monitoring module",
      title: "Watch drift, launch training, and review recent jobs.",
      description:
        "Use this screen to start a baseline or sequence training job, track the latest executions, and keep the model lifecycle visible from one place.",
    }),
  );

  const layout = createElement("div", { className: "monitoring-layout" });
  const launcher = createTrainingLauncher();
  const jobsPanel = createJobsPanel();

  layout.append(launcher, jobsPanel);
  page.append(layout);
  return page;
}

function createTrainingLauncher() {
  const panel = createElement("article", { className: "panel monitoring-launcher" });
  panel.append(
    createElement("span", { className: "panel__eyebrow", text: "Training controls" }),
    createElement("h2", { className: "panel__title", text: "Start a new training run" }),
    createElement("p", {
      className: "panel__description",
      text: "Choose a model type, give the run a name, and the training pipeline will always use fraudTrain.csv.",
    }),
  );

  const form = createElement("div", { className: "training-form" });
  const modelTypeGroup = createElement("div", { className: "training-field-group" });
  modelTypeGroup.append(
    createElement("span", { className: "training-field__label", text: "Model type" }),
    createOptionButtonGroup(),
  );

  const runNameInput = createInputField({
    label: "Run name",
    name: "run_name",
    placeholder: "e.g. july-baseline-refresh",
  });

  const featureSpecInput = createInputField({
    label: "Feature spec path",
    name: "feature_spec_path",
    placeholder: "Optional: specs/features.json",
  });

  const status = createElement("pre", {
    className: "api-result",
    text: "Select a model type and start a training job.",
  });

  const actions = createElement("div", { className: "dashboard-actions" });
  const startBaselineButton = createButton({
    label: "Start baseline training",
      onClick: () =>
        submitTrainingJob({
          model_type: "baseline",
          runNameInput,
          featureSpecInput,
          status,
        }),
  });
  const startSequenceButton = createButton({
    label: "Start sequence training",
      onClick: () =>
        submitTrainingJob({
          model_type: "sequence",
          runNameInput,
          featureSpecInput,
          status,
        }),
  });

  actions.append(startBaselineButton, startSequenceButton);
  form.append(modelTypeGroup, runNameInput.wrapper, featureSpecInput.wrapper, actions, status);
  panel.append(form);
  return panel;
}

function createJobsPanel() {
  const panel = createElement("article", { className: "panel monitoring-jobs" });
  panel.append(
    createElement("span", { className: "panel__eyebrow", text: "Recent runs" }),
    createElement("h2", { className: "panel__title", text: "Training job history" }),
    createElement("p", {
      className: "panel__description",
      text: "Refresh the latest jobs to inspect status, model type, and the fixed training dataset.",
    }),
  );

  const list = createElement("div", { className: "monitoring-job-list" });
  const status = createElement("pre", {
    className: "api-result",
    text: "No jobs loaded yet.",
  });

  const refreshButton = createButton({
    label: "Refresh jobs",
    variant: "secondary",
    onClick: async () => {
      await useTrainingJobs({
        onPending: () => {
          status.textContent = "Loading jobs...";
        },
        onSuccess: (payload) => {
          renderJobs(list, payload.jobs ?? []);
          status.textContent = `Loaded ${payload.jobs?.length ?? 0} jobs.`;
        },
        onError: (error) => {
          status.textContent = `Unable to load jobs: ${error.message}`;
          list.replaceChildren(createElement("p", { className: "panel__description", text: "No jobs available." }));
        },
      });
    },
  });

  panel.append(refreshButton, list, status);
  return panel;
}

function createOptionButtonGroup() {
  const group = createElement("div", { className: "training-option-group" });

  const baseline = createElement("label", { className: "training-option is-active" });
  const sequence = createElement("label", { className: "training-option" });

  baseline.append(
    createElement("input", {
      attrs: { type: "radio", name: "model_type", value: "baseline", checked: true },
    }),
    createElement("span", { text: "Baseline" }),
    createElement("small", { text: "Fastest option for quick retraining." }),
  );

  sequence.append(
    createElement("input", {
      attrs: { type: "radio", name: "model_type", value: "sequence" },
    }),
    createElement("span", { text: "Sequence" }),
    createElement("small", { text: "Uses the sequence training pipeline." }),
  );

  const baselineInput = baseline.querySelector("input");
  const sequenceInput = sequence.querySelector("input");

  baselineInput?.addEventListener("change", () => {
    baseline.classList.toggle("is-active", baselineInput.checked);
    sequence.classList.toggle("is-active", sequenceInput?.checked ?? false);
  });

  sequenceInput?.addEventListener("change", () => {
    baseline.classList.toggle("is-active", baselineInput?.checked ?? false);
    sequence.classList.toggle("is-active", sequenceInput.checked);
  });

  group.append(baseline, sequence);
  return group;
}

function createInputField({ label, name, placeholder }) {
  const wrapper = createElement("label", { className: "training-input" });
  const input = createElement("input", {
    attrs: {
      class: "training-input__control",
      name,
      placeholder,
      autocomplete: "off",
    },
  });

  wrapper.append(
    createElement("span", { className: "training-field__label", text: label }),
    input,
  );

  return { wrapper, input };
}

async function submitTrainingJob({ model_type, runNameInput, featureSpecInput, status }) {
  const payload = {
    model_type,
    run_name: trimToNull(runNameInput.input.value),
    feature_spec_path: trimToNull(featureSpecInput.input.value),
  };

  status.textContent = "Submitting training job...";

  const response = await useStartTrainingJob({
    payload,
    onPending: () => {
      status.textContent = "Submitting training job...";
    },
    onSuccess: (job) => {
      status.textContent = formatJson({
        message: "Training job started successfully.",
        job_id: job.job_id,
        status: job.status,
        model_type: job.model_type,
      });
    },
    onError: (error) => {
      status.textContent = `Unable to start training job: ${error.message}`;
    },
  });

  return response;
}

function renderJobs(list, jobs) {
  if (!jobs.length) {
    list.replaceChildren(createElement("p", { className: "panel__description", text: "No training jobs found." }));
    return;
  }

  const rows = jobs.slice(0, 6).map((job) => {
    const item = createElement("article", { className: "job-item" });
    item.append(
      createElement("strong", { text: `${job.job_id} · ${job.model_type}` }),
      createElement("span", { text: `Status: ${job.status}` }),
      createElement("span", { text: `Dataset: ${job.dataset_path}` }),
      createElement("span", { text: `Updated: ${job.updated_at}` }),
    );
    return item;
  });

  list.replaceChildren(...rows);
}

function trimToNull(value) {
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
}
