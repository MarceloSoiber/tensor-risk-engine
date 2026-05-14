import { createElement, createTextSection } from "../../utils/dom.js";

export function createAuthPage() {
  const page = createElement("section", { className: "page" });
  page.append(
    createTextSection({
      eyebrow: "Auth module",
      title: "Authentication flows belong here.",
      description: "Use this module for sign-in, session handling, and access control screens.",
    }),
  );
  return page;
}
