import { createElement, createTextSection } from "../../utils/dom.js";

export function createFraudPage() {
  const page = createElement("section", { className: "page" });
  page.append(
    createTextSection({
      eyebrow: "Fraud module",
      title: "Centralize flagged cases and investigation notes.",
      description: "Use this module for case triage, scoring results, and analyst workflows.",
    }),
  );
  return page;
}
