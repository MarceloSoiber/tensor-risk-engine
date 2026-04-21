import { fetchHealth } from "../services/api.js";

export async function useApiHealth({ onPending, onSuccess, onError } = {}) {
  try {
    onPending?.();
    const payload = await fetchHealth();
    onSuccess?.(payload);
    return payload;
  } catch (error) {
    const normalizedError = error instanceof Error ? error : new Error("Unexpected API error.");
    onError?.(normalizedError);
    return null;
  }
}
