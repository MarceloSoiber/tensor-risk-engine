export function createElement(tagName, { className, text, attrs, children } = {}) {
  const element = document.createElement(tagName);

  if (className) {
    element.className = className;
  }

  if (typeof text === "string") {
    element.textContent = text;
  }

  if (attrs) {
    for (const [name, value] of Object.entries(attrs)) {
      if (value !== undefined && value !== null) {
        element.setAttribute(name, String(value));
      }
    }
  }

  if (Array.isArray(children)) {
    element.append(...children.filter(Boolean));
  }

  return element;
}

export function createTextSection({ eyebrow, title, description }) {
  const section = createElement("article", { className: "hero" });

  section.append(
    createElement("span", { className: "hero__eyebrow", text: eyebrow }),
    createElement("h1", { className: "hero__title", text: title }),
    createElement("p", { className: "hero__description", text: description }),
  );

  return section;
}

export function formatJson(value) {
  return JSON.stringify(value, null, 2);
}
