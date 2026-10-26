"use client";

import { useEffect, useState } from "react";
import { alertsSocketUrl } from "@/lib/api";
import type { AlertMessage } from "@/lib/types";

/**
 * Subscribes to /ws/alerts and shows a dismissible banner.
 *
 * Filters to the patient currently on screen. A clinician looking at one
 * patient should not be interrupted by another patient's alert in this view —
 * that belongs on a ward dashboard, not here, and mixing them invites acting
 * on the wrong record.
 */
export default function AlertBanner({ patientId }: { patientId: number }) {
  const [alert, setAlert] = useState<AlertMessage | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout>;
    let closed = false;

    const connect = () => {
      socket = new WebSocket(alertsSocketUrl());
      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        try {
          const message: AlertMessage = JSON.parse(event.data);
          if (message.patient_id === patientId) setAlert(message);
        } catch {
          /* ignore malformed frames rather than tearing down the socket */
        }
      };
      socket.onclose = () => {
        setConnected(false);
        // Reconnect: a dropped socket must not silently stop alerts for the
        // rest of the shift.
        if (!closed) retry = setTimeout(connect, 3000);
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      socket?.close();
    };
  }, [patientId]);

  if (!alert) {
    return (
      <p className="text-xs text-slate-500">
        Alerts {connected ? "connected" : "reconnecting…"}
      </p>
    );
  }

  return (
    <div className="rounded-lg border-l-4 border-red-600 bg-red-50 p-4 dark:bg-red-950/40">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-semibold text-red-800 dark:text-red-200">
            Site alert — {alert.patient_name}
          </p>
          <p className="mt-1 text-sm text-red-700 dark:text-red-300">
            {alert.reason}
          </p>
          <p className="mt-1 text-xs text-red-600 dark:text-red-400">
            Redness score {alert.risk_score.toFixed(2)} · trend {alert.trend} · advisory
            only — confirm before acting
          </p>
        </div>
        <button
          onClick={() => setAlert(null)}
          className="shrink-0 rounded px-2 py-1 text-sm text-red-700 hover:bg-red-100 dark:text-red-300 dark:hover:bg-red-900"
          aria-label="Dismiss alert"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}
