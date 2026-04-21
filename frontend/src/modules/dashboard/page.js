import { createApiCheckForm } from "../../components/forms/api-check-form.js";
import { createHealthCard } from "../../components/charts/health-card.js";
import { createButton } from "../../components/ui/button.js";
import { createElement, createTextSection } from "../../utils/dom.js";

export function createDashboardPage(context) {
  const page = createElement("section", { className: "page page--dashboard" });

  const hero = createTextSection({
    eyebrow: "Operations overview",
    title: "Monitor API health and risk signals from one place.",
    description:
      "This dashboard keeps the initial backend connectivity test while making room for future fraud, transaction, and model monitoring flows.",
  });

  const actions = createElement("div", { className: "page__actions" });
  actions.append(
    createButton({
      label: "Open transactions",
      href: "#/transactions",
      variant: "secondary",
    }),
    createButton({
      label: "Review fraud module",
      href: "#/fraud",
    }),
  );

  const summary = createElement("div", { className: "dashboard-grid" });
  const healthCard = createHealthCard({
    title: "Backend health",
    description: "Run the health check to validate the API connection.",
  });
  const form = createApiCheckForm({
    onStateChange: healthCard.update,
  });

  const contextCard = createElement("article", { className: "panel panel--soft" });
  contextCard.append(
    createElement("span", { className: "panel__eyebrow", text: "Workspace" }),
    createElement("h2", { className: "panel__title", text: context.appName }),
    createElement("p", {
      className: "panel__description",
      text: "The source tree is now aligned with an app, module, component, service, hook, utils, and types layout.",
    }),
  );

  summary.append(healthCard.element, form, contextCard);

  page.append(hero, actions, summary);
  return page;
}
