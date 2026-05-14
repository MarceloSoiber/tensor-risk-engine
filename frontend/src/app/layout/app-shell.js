import { createElement } from "../../utils/dom.js";

export function createAppShell({ appName, tagline, routes, activeHash }) {
  const shell = createElement("div", { className: "app-shell" });

  const sidebar = createElement("aside", { className: "dashboard-sidebar" });
  const navigation = createElement("nav", {
    className: "dashboard-sidebar__nav",
    attrs: { "aria-label": "Primary workspace navigation" },
  });

  for (const route of routes) {
    if (route.sidebar === false) {
      continue;
    }

    const link = createElement("a", {
      className: `button button--secondary dashboard-sidebar__link${route.hash === activeHash ? " is-active" : ""}`,
      text: route.label,
      attrs: { href: route.hash },
    });

    navigation.append(link);
  }

  sidebar.append(
    createElement("div", {
      className: "dashboard-sidebar__header",
      children: [
        createElement("span", { className: "dashboard-sidebar__mark", text: "FD" }),
        createElement("span", { className: "panel__eyebrow", text: "Navigation" }),
      ],
    }),
    navigation,
  );

  const header = createElement("header", { className: "app-header" });
  const brand = createElement("div", { className: "brand" });
  const status = createElement("div", { className: "app-header__status" });

  brand.append(
    createElement("span", { className: "brand__eyebrow", text: "Risk intelligence console" }),
    createElement("h1", { className: "brand__title", text: appName }),
    createElement("p", { className: "brand__tagline", text: tagline }),
  );

  status.append(
    createElement("span", { className: "app-header__status-label", text: "Workspace" }),
    createElement("strong", { className: "app-header__status-value", text: "Live monitoring" }),
  );

  header.append(brand, status);

  const main = createElement("main", { className: "app-main" });
  const workspace = createElement("section", {
    className: "dashboard-workspace",
    attrs: { "aria-live": "polite" },
  });
  main.append(workspace);

  shell.append(header, sidebar, main);
  return { shell, workspace };
}

export function updateActiveSidebarLink(shell, activeHash) {
  const links = shell.querySelectorAll(".dashboard-sidebar__link");
  for (const link of links) {
    link.classList.toggle("is-active", link.getAttribute("href") === activeHash);
  }
}
