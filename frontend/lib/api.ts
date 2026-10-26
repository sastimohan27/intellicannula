import axios from "axios";
import type { Patient, PatientHistory, SiteCheckin, VeinScan } from "./types";

/**
 * API base URL. A configured `localhost` URL is rewritten to the host the page
 * was loaded from, so opening the dashboard on a phone at
 * http://<laptop-IP>:3000 calls the API on the laptop, not on the phone.
 */
function resolveApiUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  if (typeof window === "undefined") return configured;
  const url = new URL(configured);
  const page = window.location.hostname;
  if (["localhost", "127.0.0.1"].includes(url.hostname) && !["localhost", "127.0.0.1"].includes(page)) {
    url.hostname = page;
  }
  return url.origin;
}

export const API_URL = resolveApiUrl();

const client = axios.create({ baseURL: API_URL, timeout: 120_000 });

/** Surface the backend's `detail` message rather than a generic axios string. */
function describe(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (error.code === "ECONNABORTED") return "Request timed out.";
    if (!error.response) return `Cannot reach the API at ${API_URL}.`;
    return `${error.response.status} ${error.response.statusText}`;
  }
  return error instanceof Error ? error.message : "Unknown error";
}

export class ApiError extends Error {}

async function call<T>(fn: () => Promise<{ data: T }>): Promise<T> {
  try {
    return (await fn()).data;
  } catch (error) {
    throw new ApiError(describe(error));
  }
}

export const getPatient = (id: number) =>
  call<Patient>(() => client.get(`/patients/${id}`));

export const listPatients = () =>
  call<Patient[]>(() => client.get("/patients"));

export const createPatient = (name: string, mrn: string) =>
  call<Patient>(() => client.post("/patients", { name, mrn }));

export const getHistory = (id: number) =>
  call<PatientHistory>(() => client.get(`/patients/${id}/history`));

function upload(path: string, file: File) {
  const form = new FormData();
  form.append("file", file);
  return client.post(path, form);
}

export const postVeinScan = (id: number, file: File) =>
  call<VeinScan>(() => upload(`/patients/${id}/vein-scan`, file));

export const postSiteCheck = (id: number, file: File) =>
  call<SiteCheckin>(() => upload(`/patients/${id}/site-check`, file));

export const alertsSocketUrl = () =>
  `${API_URL.replace(/^http/, "ws")}/ws/alerts`;
