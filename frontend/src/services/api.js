const DEFAULT_HEADERS = {
  Accept: "application/json",
};

export async function requestJson(path, options = {}) {
  try {
    const response = await fetch(path, {
      ...options,
      headers: {
        ...DEFAULT_HEADERS,
        ...(options.headers ?? {}),
      },
    });

    const contentType = response.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.text();

    if (!response.ok) {
      const message =
        typeof payload === "string"
          ? payload
          : payload?.message || payload?.detail || `Request failed with status ${response.status}`;
      throw new Error(message);
    }

    return payload;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }

    throw new Error("Unexpected API error.");
  }
}

export function fetchHealth() {
  return requestJson("/api/health");
}
