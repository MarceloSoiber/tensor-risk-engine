import { createElement } from "../../utils/dom.js";

export function createHealthCard({ title, description }) {
  const card = createElement("article", { className: "panel health-card" });

  const heading = createElement("h2", { className: "panel__title", text: title });
  const subtitle = createElement("p", { className: "panel__description", text: description });
  const status = createElement("span", {
    className: "health-card__status health-card__status--idle",
    text: "Waiting",
  });
  const value = createElement("strong", { className: "health-card__value", text: "No request yet" });
  const details = createElement("pre", {
    className: "health-card__details",
    text: "Click the health check button to inspect the backend response.",
  });

  function update(state) {
    const normalizedState = state?.state ?? "idle";
    status.className = `health-card__status health-card__status--${normalizedState}`;
    status.textContent = state?.label ?? "Waiting";
    value.textContent = state?.summary ?? "No request yet";
    details.textContent = state?.details ?? "Click the health check button to inspect the backend response.";
  }

  card.append(
    createElement("span", { className: "panel__eyebrow", text: "Health summary" }),
    heading,
    subtitle,
    status,
    value,
    details,
  );

  return {
    element: card,
    update,
  };
}
