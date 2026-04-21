import { createElement } from "../../utils/dom.js";

export function createAppShell({ appName, tagline, routes, activeHash, content }) {
  const shell = createElement("div", { className: "app-shell" });

  const header = createElement("header", { className: "app-header" });
  const brand = createElement("div", { className: "brand" });
  brand.append(
    createElement("span", { className: "brand__eyebrow", text: "Risk intelligence" }),
    createElement("h1", { className: "brand__title", text: appName }),
    createElement("p", { className: "brand__tagline", text: tagline }),
  );

  const navigation = createElement("nav", { className: "app-nav", attrs: { "aria-label": "Primary" } });
  const list = createElement("ul", { className: "app-nav__list" });

  for (const route of routes) {
    const item = createElement("li", { className: "app-nav__item" });
    const link = createElement("a", {
      className: `app-nav__link${route.hash === activeHash ? " is-active" : ""}`,
      text: route.label,
      attrs: { href: route.hash },
    });

    item.append(link);
    list.append(item);
  }

  navigation.append(list);
  header.append(brand, navigation);

  const main = createElement("main", { className: "app-main" });
  main.append(content);

  shell.append(header, main);
  return shell;
}
