"use client";

import { use, useCallback, useEffect, useState } from "react";
import AlertBanner from "@/components/AlertBanner";
import RiskChart from "@/components/RiskChart";
import UploadCard from "@/components/UploadCard";
import { getHistory, getPatient, postSiteCheck, postVeinScan } from "@/lib/api";
import type { Patient, PatientHistory, SiteCheckin, VeinScan } from "@/lib/types";

function riskBadge(score: number) {
  if (score >= 0.6) return { label: "High", cls: "bg-red-600" };
  if (score >= 0.3) return { label: "Moderate", cls: "bg-amber-500" };
  return { label: "Low", cls: "bg-emerald-600" };
}

export default function PatientPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const patientId = Number(use(params).id);

  const [patient, setPatient] = useState<Patient | null>(null);
  const [scan, setScan] = useState<VeinScan | null>(null);
  const [checkin, setCheckin] = useState<SiteCheckin | null>(null);
  const [history, setHistory] = useState<PatientHistory | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [p, h] = await Promise.all([
        getPatient(patientId),
        getHistory(patientId),
      ]);
      setPatient(p);
      setHistory(h);
      setLoadError(null);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Failed to load patient");
    }
  }, [patientId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (loadError) {
    return (
      <main className="mx-auto max-w-3xl p-6">
        <p className="rounded border border-red-300 bg-red-50 p-4 text-red-700 dark:bg-red-950/40 dark:text-red-300">
          {loadError}
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-4xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-bold">{patient?.name ?? "Loading…"}</h1>
        <p className="text-sm text-slate-500">
          MRN {patient?.mrn ?? "—"} · patient #{patientId}
        </p>
      </header>

      <AlertBanner patientId={patientId} />

      {/* 1 — Pre-cannulation */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">1 · Pre-cannulation vein scan</h2>
        <UploadCard
          title="Capture forearm image"
          description="NIR image of the antecubital fossa. Returns a suggested insertion point and needle angle."
          buttonLabel="Upload vein scan"
          onUpload={async (file) => setScan(await postVeinScan(patientId, file))}
        />

        {scan && (
          <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
            {scan.insertion_x === null ? (
              // A refusal is a valid clinical answer — show it plainly rather
              // than rendering a marker at a coordinate the model did not give.
              <div>
                <p className="font-medium text-amber-700 dark:text-amber-400">
                  No insertion site recommended
                </p>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                  {scan.reason}
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-4 text-sm">
                  <span>
                    Site <strong>({scan.insertion_x}, {scan.insertion_y})</strong>
                  </span>
                  <span>
                    Angle <strong>{scan.angle_deg?.toFixed(1)}°</strong>
                  </span>
                  <span>
                    Vein width <strong>{scan.vein_width_px?.toFixed(1)} px</strong>
                  </span>
                  <span>
                    Confidence <strong>{scan.confidence.toFixed(2)}</strong>
                  </span>
                </div>
                {scan.overlay_png_base64 && (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={`data:image/png;base64,${scan.overlay_png_base64}`}
                    alt="Predicted vein mask with the selected insertion point marked"
                    className="max-w-full rounded"
                  />
                )}
                <p className="text-xs text-slate-500">
                  Angle is a heuristic proxy for depth, not a measurement.
                  Model {scan.model_version}. Advisory only — confirm before acting.
                </p>
              </div>
            )}
          </div>
        )}
      </section>

      {/* 2 — Post-cannulation */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">2 · Post-cannulation site check</h2>
        <UploadCard
          title="Capture IV site"
          description="RGB photo of the cannula site with surrounding skin visible."
          buttonLabel="Upload site check"
          onUpload={async (file) => {
            setCheckin(await postSiteCheck(patientId, file));
            await refresh();
          }}
        />

        {checkin && (
          <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
            <div className="flex flex-wrap items-center gap-3">
              <span
                className={`rounded px-2 py-1 text-sm font-semibold text-white ${riskBadge(checkin.risk_score).cls}`}
              >
                {riskBadge(checkin.risk_score).label} · {checkin.risk_score.toFixed(2)}
              </span>
              <span className="text-sm text-slate-600 dark:text-slate-400">
                trend {checkin.trend} · Δ {checkin.delta_from_previous >= 0 ? "+" : ""}
                {checkin.delta_from_previous.toFixed(3)}
              </span>
            </div>

            {checkin.alert_triggered && (
              <p className="mt-2 text-sm font-medium text-red-700 dark:text-red-400">
                Alert: {checkin.alert_reason}
              </p>
            )}
            {checkin.registration_degraded && (
              <p className="mt-2 text-xs text-amber-700 dark:text-amber-400">
                Alignment fell back to translation only — compare this reading
                with the baseline cautiously.
              </p>
            )}
          </div>
        )}
      </section>

      {/* 3 — Trend */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">3 · Risk over time</h2>
        <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-700">
          <RiskChart points={history?.points ?? []} />
          {history && history.n_checkins > 0 && (
            <p className="mt-2 text-xs text-slate-500">
              {history.n_checkins} check-in{history.n_checkins === 1 ? "" : "s"} ·
              latest trend {history.latest_trend}
            </p>
          )}
        </div>
      </section>

      <footer className="border-t border-slate-200 pt-4 text-xs text-slate-500 dark:border-slate-700">
        Research prototype. Not a medical device. Every output is advisory and
        requires clinician confirmation.
      </footer>
    </main>
  );
}
