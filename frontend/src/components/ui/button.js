export function createButton({ label, href, onClick, variant = "primary", type = "button" }) {
  if (href) {
    const link = document.createElement("a");
    link.className = `button button--${variant}`;
    link.href = href;
    link.textContent = label;
    return link;
  }

  const button = document.createElement("button");
  button.className = `button button--${variant}`;
  button.type = type;
  button.textContent = label;

  if (typeof onClick === "function") {
    button.addEventListener("click", onClick);
  }

  return button;
}
