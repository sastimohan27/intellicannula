/**
 * Pure timeline helpers, shared by the dashboard and patient page and unit
 * tested in tests/timeline.test.ts (run with `npm test`).
 */

export const CHECK_INTERVAL_HOURS = 24;

/** Hours between an ISO timestamp from the API (UTC, "Z") and `now` (ms). */
export function hoursSince(iso: string, now: number): number {
  return (now - new Date(iso).getTime()) / 36e5;
}

/** True when the latest site photo is at least CHECK_INTERVAL_HOURS old. */
export function isCheckDue(timestamps: string[], now: number): boolean {
  if (timestamps.length === 0) return false;
  return hoursSince(timestamps[timestamps.length - 1], now) >= CHECK_INTERVAL_HOURS;
}

/**
 * Point a configured `localhost` API URL at the host the page was served
 * from, so a phone opening http://<laptop-IP>:3000 reaches the laptop's API.
 */
export function rewriteApiUrl(configured: string, pageHost: string | null): string {
  const url = new URL(configured);
  const local = ["localhost", "127.0.0.1"];
  if (pageHost && local.includes(url.hostname) && !local.includes(pageHost)) {
    url.hostname = pageHost;
  }
  return url.origin;
}
