import { listTrainingJobs, startTrainingJob } from "../services/trainingService.js";

export async function useTrainingJobs({ onPending, onSuccess, onError } = {}) {
  try {
    onPending?.();
    const payload = await listTrainingJobs();
    onSuccess?.(payload);
    return payload;
  } catch (error) {
    const normalizedError = error instanceof Error ? error : new Error("Unexpected training API error.");
    onError?.(normalizedError);
    return null;
  }
}

export async function useStartTrainingJob({ payload, onPending, onSuccess, onError } = {}) {
  try {
    onPending?.();
    const response = await startTrainingJob(payload);
    onSuccess?.(response);
    return response;
  } catch (error) {
    const normalizedError = error instanceof Error ? error : new Error("Unexpected training API error.");
    onError?.(normalizedError);
    return null;
  }
}
