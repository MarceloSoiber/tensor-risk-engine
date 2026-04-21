import { createAppContext } from "./app/providers/index.js";
import { createAppShell } from "./app/layout/app-shell.js";
import { createRoutes, resolveRoute } from "./app/routes/index.js";

const root = document.getElementById("app");

if (!root) {
  throw new Error("Application root element was not found.");
}

const context = createAppContext();
const routes = createRoutes(context);

function render() {
  const route = resolveRoute(routes, window.location.hash);
  document.title = `${route.label} | ${context.appName}`;

  root.replaceChildren(
    createAppShell({
      appName: context.appName,
      tagline: context.tagline,
      routes,
      activeHash: route.hash,
      content: route.render(),
    }),
  );
}

if (!window.location.hash) {
  window.location.hash = "#/dashboard";
} else {
  render();
}

window.addEventListener("hashchange", render);
