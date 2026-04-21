import { createElement, createTextSection } from "../../utils/dom.js";

export function createTransactionsPage() {
  const page = createElement("section", { className: "page" });
  page.append(
    createTextSection({
      eyebrow: "Transactions module",
      title: "Track transaction volume and suspicious patterns.",
      description: "This area is reserved for transaction review, filters, and operational timelines.",
    }),
  );
  return page;
}
