"use client";

import { useEffect, useState } from "react";
import { getNirStatus } from "@/lib/api";
import type { NirStatus as Status } from "@/lib/types";

/** States what the vein-scan workflow accepts, from the API rather than assumed. */
export default function NirStatus() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getNirStatus().then(setStatus).catch((e) => setError(e instanceof Error ? e.message : "unavailable"));
  }, []);

  return (
    <div className="rounded-lg border border-violet-500/40 bg-violet-500/10 p-3 text-sm">
      <p className="font-semibold text-violet-300">NIR modality required</p>
      <p className="mt-1 text-slate-300">
        Vein scans accept near-infrared images only. Phone cameras capture RGB and are rejected —
        they are not a substitute for NIR imaging.
      </p>
      {status && (
        <p className="mt-1 text-xs text-slate-400">
          Live NIR capture device:{" "}
          {status.live_capture_available
            ? <span className="text-emerald-400">{status.registered_devices.join(", ")}</span>
            : <span className="text-amber-400">none registered</span>}
          {" · "}verified CUBITAL NIR frames available for upload: {status.dataset_frames.toLocaleString()}
        </p>
      )}
      {error && <p className="mt-1 text-xs text-red-400">NIR status unavailable: {error}</p>}
    </div>
  );
}
