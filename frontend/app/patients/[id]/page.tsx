"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import AlertBanner from "@/components/AlertBanner";
import RiskChart from "@/components/RiskChart";
import NirStatus from "@/components/NirStatus";
import UploadCard from "@/components/UploadCard";
import VeinResult from "@/components/VeinResult";
import { hoursSince, isCheckDue } from "@/lib/timeline";
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
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
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

  // Initial load: state is only set in the promise callbacks, never
  // synchronously inside the effect.
  useEffect(() => {
    let active = true;
    Promise.all([getPatient(patientId), getHistory(patientId)])
      .then(([p, h]) => {
        if (!active) return;
        setPatient(p);
        setHistory(h);
        setLoadedAt(Date.now());
      })
      .catch((e) => active && setLoadError(e instanceof Error ? e.message : "Failed to load patient"));
    return () => { active = false; };
  }, [patientId]);

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
        <Link href="/" className="text-xs text-teal-400 hover:underline">← Dashboard</Link>
        <h1 className="mt-1 text-2xl font-bold">{patient?.name ?? "Loading…"}</h1>
        <p className="text-sm text-slate-500">
          MRN {patient?.mrn ?? "—"} · patient #{patientId}
        </p>
      </header>

      <AlertBanner patientId={patientId} />

      {/* 1 — Pre-cannulation */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">1 · Pre-cannulation vein scan (NIR)</h2>
        <NirStatus />
        <UploadCard
          title="Upload NIR forearm image"
          description="Near-infrared image of the forearm from a registered NIR device, or a verified CUBITAL NIR frame. RGB photos are rejected before the model runs."
          buttonLabel="Upload NIR scan"
          onUpload={async (file) => {
            const result = await postVeinScan(patientId, file);
            setOriginalUrl((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(file); });
            setScan(result);
          }}
        />
        {scan && <VeinResult scan={scan} originalUrl={originalUrl} />}
      </section>

      {/* 2 — Post-cannulation */}
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">2 · Post-cannulation site check</h2>
        <CheckDue points={history?.points ?? []} now={loadedAt} />
        <UploadCard
          title="Capture IV site"
          description="Daily photo of the cannula site. Hold the phone ~20 cm away, flash off, hub in the centre, with normal skin visible around the dressing. The first photo becomes this patient's baseline."
          buttonLabel="Capture / upload site photo"
          camera
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
          <AssessmentList points={history?.points ?? []} />
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
  if (now === null || !isCheckDue(points.map((p) => p.timestamp), now)) return null;
  const hours = hoursSince(points[points.length - 1].timestamp, now);
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
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded bg-slate-700 px-2 py-1 font-semibold text-white">Image accepted</span>
        {checkin.assessment_label && (
          <span className={`rounded px-2 py-1 font-semibold text-white ${STATUS_CLS[checkin.assessment_status ?? ""] ?? "bg-slate-600"}`}>
            {checkin.assessment_label}
          </span>
        )}
        {checkin.sample_origin === "simulated_demo" ? (
          <span className="rounded border border-fuchsia-500 px-2 py-1 font-semibold text-fuchsia-300">
            SIMULATED demo sample — not patient data
          </span>
        ) : (
          <span className="rounded border border-slate-600 px-2 py-1">Uploaded photo</span>
        )}
        <span className="text-slate-500">assessed {new Date(checkin.created_at).toLocaleString()}</span>
      </div>

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

      <div className={`grid gap-3 ${checkin.baseline_overlay_png_base64 ? "sm:grid-cols-2" : ""}`}>
        {checkin.baseline_overlay_png_base64 && (
          <SiteImage
            b64={checkin.baseline_overlay_png_base64}
            title={`Baseline${checkin.baseline_created_at ? ` · ${new Date(checkin.baseline_created_at).toLocaleString()}` : ""}`}
          />
        )}
        {checkin.overlay_png_base64 && (
          <SiteImage b64={checkin.overlay_png_base64} title={`This check-in · ${new Date(checkin.created_at).toLocaleString()}`} />
        )}
      </div>
      <p className="text-xs text-slate-500">
        Green: detected hub · cyan: measured ring · red: skin counted as reddened.
        Dressing, tape and plastic are excluded.
        {checkin.comparison_status === "compared" && checkin.scene_match_inliers !== null &&
          ` Same site as baseline confirmed (${checkin.scene_match_inliers} matching landmarks).`}
        {checkin.comparison_status === "baseline_unusable" &&
          " The baseline photo has no verified cannula, so no comparison was made."}
      </p>

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

const STATUS_CLS: Record<string, string> = {
  baseline_recorded: "bg-sky-700",
  no_significant_change: "bg-emerald-700",
  visible_change: "bg-amber-600",
  review_recommended: "bg-red-700",
  unable_to_compare: "bg-slate-600",
};

function SiteImage({ b64, title }: { b64: string; title: string }) {
  return (
    <figure>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`data:image/png;base64,${b64}`} alt={title} className="max-h-[24rem] w-full rounded object-contain" />
      <figcaption className="mt-1 text-xs text-slate-500">{title}</figcaption>
    </figure>
  );
}

/** Previous assessments, newest first, in the viewer's local time. */
function AssessmentList({ points }: { points: HistoryPoint[] }) {
  if (points.length === 0) return null;
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="text-slate-500">
          <tr><th className="py-1 pr-3">#</th><th className="pr-3">Time</th><th className="pr-3">Score</th><th className="pr-3">Change vs previous</th><th>Alert</th></tr>
        </thead>
        <tbody>
          {points.map((p, i) => ({ p, i })).reverse().map(({ p, i }) => (
            <tr key={i} className="border-t border-slate-800">
              <td className="py-1 pr-3">{i + 1}{i === 0 && <span className="ml-1 text-sky-400">baseline</span>}</td>
              <td className="pr-3">{new Date(p.timestamp).toLocaleString()}</td>
              <td className="pr-3">{p.risk_score.toFixed(2)}</td>
              <td className="pr-3">{i === 0 ? "—" : `${p.delta_from_previous >= 0 ? "+" : ""}${p.delta_from_previous.toFixed(2)}`}</td>
              <td>{p.alert_triggered ? <span className="text-red-400">yes</span> : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
