import { createButton } from "../../components/ui/button.js";
import { createElement, formatJson } from "../../utils/dom.js";
import { useDeleteTrainingJob, useStartTrainingJob, useTrainingJobs } from "../../hooks/use-training-jobs.js";

export function createModelMonitoringPage() {
  const page = createElement("section", { className: "page page--model-monitoring" });

  const layout = createElement("div", { className: "monitoring-layout" });
  const jobsPanel = createJobsPanel();
  const launcher = createTrainingLauncher({ onJobStarted: jobsPanel.refresh });

  layout.append(launcher, jobsPanel.element);
  page.append(layout);
  jobsPanel.startPolling();
  return page;
}

function createTrainingLauncher({ onJobStarted } = {}) {
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
  const summary = createElement("div", { className: "training-summary" });
  summary.append(
    createElement("div", {
      className: "training-summary__item",
      children: [
        createElement("span", { text: "Dataset" }),
        createElement("strong", { text: "fraudTrain.csv" }),
      ],
    }),
    createElement("div", {
      className: "training-summary__item",
      children: [
        createElement("span", { text: "Mode" }),
        createElement("strong", { text: "Live tracked" }),
      ],
    }),
  );

  const actions = createElement("div", { className: "training-actions" });
  const startButton = createButton({
    label: "Start baseline training",
    onClick: () =>
      submitTrainingJob({
        model_type: modelOptions.getValue(),
        runNameInput,
        featureSpecInput,
        status,
        onJobStarted,
      }),
  });

  const modelOptions = createOptionButtonGroup({
    onChange: (value) => {
      startButton.textContent = value === "baseline" ? "Start baseline training" : "Start sequence training";
    },
  });

  const modelTypeGroup = createElement("div", { className: "training-field-group" });
  modelTypeGroup.append(
    createElement("span", { className: "training-field__label", text: "Model type" }),
    modelOptions.element,
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
    className: "api-result training-submit-status",
    text: "Select a model type and start a training job.",
  });

  actions.append(startButton);
  form.append(summary, modelTypeGroup, runNameInput.wrapper, featureSpecInput.wrapper, actions, status);
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

  const liveProgress = createElement("div", { className: "training-live-progress" });
  const list = createElement("div", { className: "monitoring-job-list" });
  const status = createElement("pre", {
    className: "api-result",
    text: "No jobs loaded yet.",
  });
  let pollingIntervalId = null;
  let currentJobs = [];

  const renderCurrentJobs = () => {
    renderLiveProgress(liveProgress, currentJobs);
    renderJobs(list, currentJobs, {
      onRemove: async (job) => {
        await removeTrainingJob(job, {
          getJobs: () => currentJobs,
          setJobs: (jobs) => {
            currentJobs = jobs;
            renderCurrentJobs();
          },
          status,
          refresh,
        });
      },
    });
  };

  const refresh = async ({ silent = false } = {}) => {
    const payload = await useTrainingJobs({
      onPending: () => {
        if (!silent) {
          status.textContent = "Loading jobs...";
        }
      },
      onSuccess: (payload) => {
        currentJobs = payload.jobs ?? [];
        renderCurrentJobs();
        const activeJob = currentJobs.find((job) => ["queued", "running"].includes(job.status));
        status.textContent = activeJob
          ? `Live update: ${activeJob.model_type} training is ${activeJob.status}.`
          : `Loaded ${currentJobs.length} jobs. No active training run.`;
      },
      onError: (error) => {
        status.textContent = `Unable to load jobs: ${error.message}`;
        liveProgress.replaceChildren();
        list.replaceChildren(createElement("p", { className: "panel__description", text: "No jobs available." }));
      },
    });

    const hasActiveJob = Boolean(payload?.jobs?.some((job) => ["queued", "running"].includes(job.status)));
    if (!hasActiveJob && pollingIntervalId !== null) {
      window.clearInterval(pollingIntervalId);
      pollingIntervalId = null;
    }
  };

  const refreshButton = createButton({
    label: "Refresh jobs",
    variant: "secondary",
    onClick: () => refresh(),
  });

  panel.append(refreshButton, liveProgress, list, status);
  return {
    element: panel,
    refresh: async (options) => {
      await refresh(options);
      startPolling();
    },
    startPolling: () => {
      refresh({ silent: true });
      startPolling();
    },
  };

  function startPolling() {
    if (pollingIntervalId !== null) {
      return;
    }

    pollingIntervalId = window.setInterval(() => {
        if (!panel.isConnected) {
          window.clearInterval(pollingIntervalId);
          pollingIntervalId = null;
          return;
        }

        refresh({ silent: true });
      }, 2500);
  }
}

function createOptionButtonGroup({ onChange } = {}) {
  const group = createElement("div", { className: "training-option-group" });

  const baseline = createElement("label", { className: "training-option is-active" });
  const sequence = createElement("label", { className: "training-option" });

  baseline.append(
    createElement("input", {
      attrs: { type: "radio", name: "model_type", value: "baseline", checked: true },
    }),
    createElement("span", { text: "Baseline refresh" }),
    createElement("small", { text: "Fast validation path for tabular risk scoring." }),
  );

  sequence.append(
    createElement("input", {
      attrs: { type: "radio", name: "model_type", value: "sequence" },
    }),
    createElement("span", { text: "Sequence model" }),
    createElement("small", { text: "GRU/LSTM training with transaction history windows." }),
  );

  const baselineInput = baseline.querySelector("input");
  const sequenceInput = sequence.querySelector("input");

  baselineInput?.addEventListener("change", () => {
    baseline.classList.toggle("is-active", baselineInput.checked);
    sequence.classList.toggle("is-active", sequenceInput?.checked ?? false);
    if (baselineInput.checked) {
      onChange?.("baseline");
    }
  });

  sequenceInput?.addEventListener("change", () => {
    baseline.classList.toggle("is-active", baselineInput?.checked ?? false);
    sequence.classList.toggle("is-active", sequenceInput.checked);
    if (sequenceInput.checked) {
      onChange?.("sequence");
    }
  });

  group.append(baseline, sequence);
  return {
    element: group,
    getValue: () => (sequenceInput?.checked ? "sequence" : "baseline"),
  };
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

async function submitTrainingJob({ model_type, runNameInput, featureSpecInput, status, onJobStarted }) {
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
        run_name: job.run_name,
        status: job.status,
        model_type: job.model_type,
      });
      onJobStarted?.({ silent: true });
    },
    onError: (error) => {
      status.textContent = `Unable to start training job: ${error.message}`;
    },
  });

  return response;
}

function renderLiveProgress(container, jobs) {
  const activeJob = jobs.find((job) => ["queued", "running"].includes(job.status)) ?? jobs[0];
  if (!activeJob) {
    container.replaceChildren(
      createElement("p", { className: "panel__description", text: "No training runs available yet." }),
    );
    return;
  }

  container.replaceChildren(createProgressCard(activeJob, { prominent: true }));
}

function renderJobs(list, jobs, { onRemove } = {}) {
  if (!jobs.length) {
    list.replaceChildren(createElement("p", { className: "panel__description", text: "No training jobs found." }));
    return;
  }

  const rows = jobs.slice(0, 6).map((job) => {
    const title = job.run_name || job.job_id;
    const item = createElement("article", {
      className: "job-item",
      attrs: { "data-job-id": job.job_id },
    });
    const header = createElement("div", { className: "job-item__header" });
    header.append(
      createElement("strong", { text: `${title} · ${job.model_type}` }),
      createRemoveJobButton(job, onRemove),
    );

    item.append(
      header,
      ...(job.run_name ? [createElement("span", { text: `Job ID: ${job.job_id}` })] : []),
      createElement("span", { text: `Status: ${job.status}` }),
      createProgressCard(job),
      createElement("span", { text: `Dataset: ${job.dataset_path}` }),
      createElement("span", { text: `Updated: ${job.updated_at}` }),
    );
    return item;
  });

  list.replaceChildren(...rows);
}

function createRemoveJobButton(job, onRemove) {
  const button = createElement("button", {
    className: "job-item__remove",
    text: "Remove",
    attrs: {
      type: "button",
      "aria-label": `Remove training job ${job.job_id}`,
      disabled: ["queued", "running"].includes(job.status) ? "true" : null,
    },
  });

  if (!["queued", "running"].includes(job.status)) {
    button.addEventListener("click", () => onRemove?.(job));
  }

  return button;
}

async function removeTrainingJob(job, { getJobs, setJobs, status, refresh }) {
  const confirmed = window.confirm(
    `Remove training job ${job.job_id} and delete its generated artifacts folder?`,
  );
  if (!confirmed) {
    return;
  }

  await useDeleteTrainingJob({
    jobId: job.job_id,
    onPending: () => {
      status.textContent = `Removing training job ${job.job_id}...`;
    },
    onSuccess: async () => {
      const nextJobs = getJobs().filter((item) => item.job_id !== job.job_id);
      setJobs(nextJobs);
      status.textContent = `Removed training job ${job.job_id}.`;
      await refresh({ silent: true });
    },
    onError: (error) => {
      status.textContent = `Unable to remove training job: ${error.message}`;
    },
  });
}

function createProgressCard(job, { prominent = false } = {}) {
  const progress = resolveProgress(job);
  const wrapper = createElement("div", {
    className: `training-progress${prominent ? " training-progress--prominent" : ""}`,
  });
  const meta = createElement("div", { className: "training-progress__meta" });
  const runLabel = job.run_name || job.model_type;
  meta.append(
    createElement("span", {
      text: prominent ? `${runLabel} · ${job.status}` : `${progress.label} · ${job.status}`,
    }),
    createElement("strong", { text: `${Math.round(progress.percent)}%` }),
  );

  const track = createElement("div", {
    className: `training-progress__track${progress.indeterminate ? " is-indeterminate" : ""}`,
    attrs: {
      role: "progressbar",
      "aria-valuemin": "0",
      "aria-valuemax": "100",
      "aria-valuenow": String(Math.round(progress.percent)),
    },
  });
  const bar = createElement("span", {
    className: "training-progress__bar",
    attrs: { style: `width: ${progress.percent}%` },
  });
  track.append(bar);

  const details = createElement("span", {
    className: "training-progress__details",
    text: progress.detail,
  });

  wrapper.append(meta, track, details);
  return wrapper;
}

function resolveProgress(job) {
  if (job.status === "succeeded") {
    return {
      percent: 100,
      label: job.progress_stage || "Completed",
      detail: formatProgressDetail(job),
      indeterminate: false,
    };
  }

  if (["failed", "canceled"].includes(job.status)) {
    return {
      percent: 100,
      label: job.progress_stage || job.status,
      detail: job.error || formatProgressDetail(job),
      indeterminate: false,
    };
  }

  const percent = Number(job.progress_percent);
  if (Number.isFinite(percent)) {
    return {
      percent: clamp(percent, 4, 96),
      label: job.progress_stage || "Training",
      detail: formatProgressDetail(job),
      indeterminate: false,
    };
  }

  return {
    percent: job.status === "queued" ? 8 : 35,
    label: job.status === "queued" ? "Queued" : "Running",
    detail: "Waiting for live training progress.",
    indeterminate: job.status === "running",
  };
}

function formatProgressDetail(job) {
  const epoch = Number(job.progress_epoch);
  const total = Number(job.progress_total);
  const auc = Number(job.best_val_pr_auc);
  const parts = [];
  if (Number.isFinite(epoch) && Number.isFinite(total) && total > 0) {
    parts.push(`Step ${epoch}/${total}`);
  }
  if (Number.isFinite(auc)) {
    parts.push(`Best validation PR-AUC ${auc.toFixed(4)}`);
  }
  return parts.join(" · ") || job.progress_stage || "Progress will update automatically.";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function trimToNull(value) {
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
}
