"use client";

import { use, useCallback, useEffect, useState } from "react";
import AlertBanner from "@/components/AlertBanner";
import RiskChart from "@/components/RiskChart";
import UploadCard from "@/components/UploadCard";
import { getHistory, getPatient, postSiteCheck, postVeinScan } from "@/lib/api";
import type { HistoryPoint, Patient, PatientHistory, SiteCheckin, VeinScan } from "@/lib/types";

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
  const [loadedAt, setLoadedAt] = useState<number | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [p, h] = await Promise.all([
        getPatient(patientId),
        getHistory(patientId),
      ]);
      setPatient(p);
      setHistory(h);
      setLoadedAt(Date.now());
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
                  {scan.vein_grade && (
                    <span className="text-blue-700 dark:text-blue-400">
                      Grade <strong>{scan.vein_grade}</strong>
                    </span>
                  )}
                  <span className="text-slate-500" title={scan.reason ?? undefined}>
                    Confidence <strong>{scan.confidence.toFixed(2)}</strong>
                    {scan.reason && <span className="ml-1">· {scan.reason}</span>}
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
        <CheckDue points={history?.points ?? []} now={loadedAt} />
        <UploadCard
          title="Capture IV site"
          description="Daily photo of the cannula site. Hold the phone ~20 cm away, flash off, hub in the centre, with normal skin visible around the dressing. The first photo becomes this patient's baseline."
          buttonLabel="Upload site check"
          onUpload={async (file) => {
            setCheckin(await postSiteCheck(patientId, file));
            await refresh();
          }}
        />

        {checkin && <SiteCheckResult checkin={checkin} />}
      </section>

      {/* 3 — Trend */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">3 · Redness over time</h2>
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

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Reminder when the daily site photo is overdue. */
function CheckDue({ points, now }: { points: HistoryPoint[]; now: number | null }) {
  if (points.length === 0 || now === null) return null;
  const last = new Date(points[points.length - 1].timestamp).getTime();
  const hours = (now - last) / 36e5;
  if (hours < 24) return null;
  return (
    <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-300">
      Site check due — last photo {Math.floor(hours / 24)} day{hours >= 48 ? "s" : ""} ago.
    </p>
  );
}

function SiteCheckResult({ checkin }: { checkin: SiteCheckin }) {
  const level = checkin.redness_level;
  const levelCls =
    level === null ? "bg-slate-600"
      : level >= 3 ? "bg-red-600"
      : level === 2 ? "bg-orange-600"
      : level === 1 ? "bg-amber-500"
      : "bg-emerald-600";
  const change = checkin.change_since_baseline;

  return (
    <div className="space-y-4 rounded-lg border border-slate-200 p-4 dark:border-slate-700">
      <div className="flex flex-wrap items-center gap-3">
        <span className={`rounded px-2 py-1 text-sm font-semibold text-white ${levelCls}`}>
          {checkin.redness_label ?? "Measured"}
        </span>
        <span className="text-sm text-slate-600 dark:text-slate-400">
          score <strong>{checkin.risk_score.toFixed(2)}</strong> · {checkin.risk_velocity ?? checkin.trend}
        </span>
        {checkin.cannula_confidence !== null && (
          <span className="text-sm text-slate-500">
            Cannula detected · {checkin.cannula_confidence.toFixed(2)}
          </span>
        )}
      </div>

      {checkin.reddened_fraction !== null && (
        <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          <Stat label="Reddened skin around site" value={pct(checkin.reddened_fraction)} />
          <Stat
            label="Change since baseline"
            value={change === null ? "—" : `${change >= 0 ? "+" : ""}${Math.round(change * 100)} pts`}
          />
          <Stat
            label="Redness reaches"
            value={checkin.reddened_fraction >= 0.1 && checkin.spread_hub_widths !== null
              ? `${checkin.spread_hub_widths.toFixed(1)} hub widths` : "—"}
          />
          <Stat
            label="Skin visible in ring"
            value={checkin.skin_visible_fraction === null ? "—" : pct(checkin.skin_visible_fraction)}
          />
        </dl>
      )}

      {checkin.overlay_png_base64 && (
        <figure>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${checkin.overlay_png_base64}`}
            alt="Site photo: green box is the detected hub, cyan outlines the measured ring, red marks skin counted as reddened"
            className="max-h-[28rem] max-w-full rounded"
          />
          <figcaption className="mt-1 text-xs text-slate-500">
            Green: detected hub · cyan: measured ring · red: skin counted as reddened.
            Dressing, tape and plastic are excluded.
          </figcaption>
        </figure>
      )}

      {checkin.clinical_explanation && checkin.clinical_explanation.length > 0 && (
        <div className="space-y-1 rounded border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800/50">
          <p className="text-sm font-medium text-slate-700 dark:text-slate-300">What was measured</p>
          <ul className="list-disc pl-5 text-sm text-slate-600 dark:text-slate-400">
            {checkin.clinical_explanation.map((reason, i) => <li key={i}>{reason}</li>)}
          </ul>
        </div>
      )}

      {checkin.alert_triggered && (
        <p className="text-sm font-medium text-red-700 dark:text-red-400">
          Alert: {checkin.alert_reason}
        </p>
      )}
      <p className="text-xs text-slate-500">
        Measures visible redness only — not pain, swelling or a palpable cord, so it is not a
        VIP score. Method {checkin.model_version}; not clinically validated.
      </p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-slate-200 p-2 dark:border-slate-700">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-0.5 font-semibold">{value}</dd>
    </div>
  );
}
