const button = document.getElementById("checkApi");
const result = document.getElementById("result");

button?.addEventListener("click", async () => {
  result.textContent = "Consultando API...";

  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    result.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    result.textContent = `Erro ao conectar na API: ${error}`;
  }
});
