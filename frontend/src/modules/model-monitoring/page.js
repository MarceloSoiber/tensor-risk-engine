import { createElement, createTextSection } from "../../utils/dom.js";

export function createModelMonitoringPage() {
  const page = createElement("section", { className: "page" });
  page.append(
    createTextSection({
      eyebrow: "Model monitoring module",
      title: "Watch drift, quality, and operational stability.",
      description: "This module can host model performance charts, alerts, and retraining signals.",
    }),
  );
  return page;
}
