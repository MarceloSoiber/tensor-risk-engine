import { createButton } from "../ui/button.js";
import { useApiHealth } from "../../hooks/use-api-health.js";
import { createElement } from "../../utils/dom.js";

export function createApiCheckForm({ onStateChange }) {
  const panel = createElement("article", { className: "panel panel--form" });
  const title = createElement("h2", { className: "panel__title", text: "API check" });
  const description = createElement("p", {
    className: "panel__description",
    text: "Use this action to validate that the backend is reachable from the container network.",
  });
  const result = createElement("pre", {
    className: "api-result",
    text: "Awaiting health check.",
  });

  const trigger = createButton({
    label: "Test API",
    onClick: async () => {
      await useApiHealth({
        onPending: () => {
          result.textContent = "Consulting the API...";
          onStateChange?.({
            state: "pending",
            label: "Checking",
            summary: "Request in progress",
            details: "Waiting for the backend health endpoint.",
          });
        },
        onSuccess: (payload) => {
          result.textContent = JSON.stringify(payload, null, 2);
          onStateChange?.({
            state: "success",
            label: "Healthy",
            summary: "Backend responded successfully",
            details: JSON.stringify(payload, null, 2),
          });
        },
        onError: (error) => {
          const message = `Unable to connect to the API: ${error.message}`;
          result.textContent = message;
          onStateChange?.({
            state: "error",
            label: "Unavailable",
            summary: "Backend request failed",
            details: message,
          });
        },
      });
    },
  });

  panel.append(title, description, trigger, result);
  return panel;
}
