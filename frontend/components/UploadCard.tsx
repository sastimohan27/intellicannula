"use client";

import { useRef, useState } from "react";

/**
 * File upload, optionally opening the rear camera on mobile (`camera`).
 * The vein scan passes camera={false}: a phone camera is RGB, not NIR.
 */
export default function UploadCard({
  title, description, buttonLabel, onUpload, disabled, camera = false,
}: {
  title: string;
  description: string;
  buttonLabel: string;
  onUpload: (file: File) => Promise<void>;
  disabled?: boolean;
  camera?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handle(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      await onUpload(file);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
      <h3 className="font-semibold">{title}</h3>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{description}</p>

      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        capture={camera ? "environment" : undefined}
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0])}
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={busy || disabled}
        className="mt-3 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {busy ? "Processing…" : buttonLabel}
      </button>

      {error && (
        <p className="mt-2 text-sm text-red-600 dark:text-red-400">{error}</p>
      )}
    </div>
  );
}
