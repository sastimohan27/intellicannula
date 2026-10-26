"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { createPatient, getHistory, getModels, listPatients, API_URL } from "@/lib/api";
import { isCheckDue } from "@/lib/timeline";
import type { ModelInfo, Patient, PatientHistory } from "@/lib/types";

export default function Home() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // New patient form
  const [name, setName] = useState("");
  const [mrn, setMrn] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [models, setModels] = useState<ModelInfo | null>(null);
  const router = useRouter();

  const fetchPatients = async () => {
    try {
      setLoading(true);
      const data = await listPatients();
      setPatients(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect to API");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    listPatients()
      .then((data) => { setPatients(data); setError(null); })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to connect to API"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    getModels().then(setModels).catch(() => setModels(null));
  }, []);

  // Demo patients get a DEMO- MRN so they are visibly separate from real records.
  const handleCreateDemo = async () => {
    try {
      setSubmitting(true);
      const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
      const created = await createPatient("Demo Patient (synthetic)", `DEMO-${stamp}`);
      router.push(`/patients/${created.id}`);
    } catch (err) {
      alert(err instanceof Error ? err.message : "Could not create demo patient");
      setSubmitting(false);
    }
  };

  const handleCreatePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !mrn.trim()) return;
    try {
      setSubmitting(true);
      await createPatient(name.trim(), mrn.trim());
      setName("");
      setMrn("");
      await fetchPatients();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Could not create patient");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="h-8 w-8 rounded-lg bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400 font-bold text-lg">
            IC
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-100 tracking-tight">IntelliCannula</h1>
            <p className="text-xs text-slate-400">Clinical Decision Support for Peripheral IV</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-xs text-slate-300 bg-slate-800/80 px-3 py-1.5 rounded-full border border-slate-700">
            <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>API: {API_URL}</span>
          </div>
          <a
            href={`${API_URL}/docs`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-slate-400 hover:text-teal-400 transition"
          >
            Swagger API Docs ↗
          </a>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8 space-y-8">
        {/* Clinical Advisory Banner */}
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 flex items-start space-x-3 text-amber-200">
          <div className="text-amber-400 text-lg">⚠️</div>
          <div className="text-sm">
            <strong className="font-semibold text-amber-300">Clinical Protocol:</strong> IntelliCannula is advisory decision support. The system <em>recommends, never acts</em>—all suggested insertion sites, angles and site-change measurements require clinician validation. Redness change alone cannot confirm or exclude infection. Use synthetic or de-identified data only: there is no authentication.
          </div>
        </div>

        {/* Hero & Quick Access */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 to-slate-900/40 p-6 flex flex-col justify-between">
            <div>
              <span className="inline-block text-xs font-semibold uppercase tracking-wider text-teal-400 bg-teal-500/10 px-2.5 py-1 rounded-md mb-3 border border-teal-500/20">
                Bedside Workflow
              </span>
              <h2 className="text-2xl font-bold text-white mb-2">Two-Phase Cannulation Management</h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">
                Before insertion: NIR forearm vein segmentation (Attention U-Net) — NIR images only. After insertion: daily RGB site photos, YOLOv11 cannula verification, and redness change measured against the patient&apos;s own baseline photo.
              </p>
            </div>

            <div className="flex flex-wrap gap-3 pt-2">
              <button
                onClick={() => void handleCreateDemo()}
                disabled={submitting}
                className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-teal-500 text-slate-950 font-semibold text-sm hover:bg-teal-400 transition shadow-lg shadow-teal-500/20 disabled:opacity-50"
              >
                <span>Create demo patient</span>
                <span>→</span>
              </button>
              <a
                href="#enrolled-patients"
                className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 text-slate-200 text-sm font-medium hover:bg-slate-700 transition border border-slate-700"
              >
                <span>View Patient Directory</span>
              </a>
            </div>
          </div>

          {/* Live model information (from the API) and measured results (from docs) */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 flex flex-col justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Models &amp; measured results</h3>
            <div className="space-y-3 text-sm">
              <Row label="Vein U-Net (NIR only)" value={models ? `${models.vein.version} · test Dice 0.567` : "…"} />
              <Row label="Cannula detector" value={models ? `${models.cannula_detector.version} · gate ≥ ${models.cannula_detector.conf_threshold}` : "…"} note="trained on synthetic images; real-world accuracy not measured" />
              <Row label="Site monitoring" value={models ? models.infection_risk.version : "…"} note="redness measurement — not clinically validated" />
            </div>
          </div>
        </section>

        {/* Patient Directory Section */}
        <section id="enrolled-patients" className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Patient Monitoring Directory</h2>
              <p className="text-sm text-slate-400">Select a patient to review vein scans, site photos against their baseline, redness trends and alerts.</p>
            </div>
            <button
              onClick={() => void fetchPatients()}
              className="text-xs text-slate-400 hover:text-teal-400 border border-slate-800 px-3 py-1.5 rounded-lg transition"
            >
              ↻ Refresh
            </button>
          </div>

          {error && (
            <div className="rounded-xl border border-red-500/40 bg-red-950/30 p-4 text-red-300 text-sm">
              {error} — Make sure the backend API is running on {API_URL}.
            </div>
          )}

          {loading ? (
            <div className="p-8 text-center text-slate-500 text-sm">Loading patient records...</div>
          ) : patients.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-800 p-8 text-center text-slate-400">
              <p className="mb-2">No patients enrolled yet.</p>
              <p className="text-xs text-slate-500">Create a patient below, or a demo patient above.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {patients.map((p) => (
                <Link
                  key={p.id}
                  href={`/patients/${p.id}`}
                  className="group rounded-xl border border-slate-800 bg-slate-900/60 p-5 hover:border-teal-500/50 hover:bg-slate-900 transition flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-mono text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                        {p.mrn}
                        {p.mrn.startsWith("DEMO-") && <span className="ml-2 text-fuchsia-300">demo</span>}
                      </span>
                      <span className="text-xs text-slate-500">#{p.id}</span>
                    </div>
                    <h3 className="text-base font-semibold text-white group-hover:text-teal-300 transition">
                      {p.name}
                    </h3>
                  </div>

                  <PatientSummary patientId={p.id} />
                  <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                    <span>Enrolled: {new Date(p.created_at).toLocaleDateString()}</span>
                    <span className="text-teal-400 group-hover:translate-x-0.5 transition">Open Monitor →</span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>

        {/* Enroll New Patient Section */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6">
          <h3 className="text-lg font-bold text-white mb-1">Enroll New Patient</h3>
          <p className="text-sm text-slate-400 mb-4">Add a new subject to track cannulation insertion sites and sequential checks.</p>

          <form onSubmit={handleCreatePatient} className="flex flex-col sm:flex-row gap-3">
            <input
              type="text"
              placeholder="Patient Full Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-teal-500"
              required
            />
            <input
              type="text"
              placeholder="MRN (e.g. PAT-102)"
              value={mrn}
              onChange={(e) => setMrn(e.target.value)}
              className="w-full sm:w-48 rounded-xl border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-teal-500"
              required
            />
            <button
              type="submit"
              disabled={submitting}
              className="px-6 py-2.5 rounded-xl bg-teal-600 text-white font-medium text-sm hover:bg-teal-500 transition disabled:opacity-50"
            >
              {submitting ? "Enrolling..." : "+ Enroll Patient"}
            </button>
          </form>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-600">
        IntelliCannula Research Prototype · Clinical Decision Support System · J-Component Evaluation
      </footer>
    </div>
  );
}

function Row({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="py-1.5 border-b border-slate-800 last:border-0">
      <div className="flex justify-between items-center gap-3">
        <span className="text-slate-400">{label}</span>
        <span className="text-teal-300 font-mono text-xs text-right">{value}</span>
      </div>
      {note && <p className="text-[11px] text-amber-300/80 mt-0.5">{note}</p>}
    </div>
  );
}

/** Latest check-in, alert count and whether the daily photo is overdue. */
function PatientSummary({ patientId }: { patientId: number }) {
  const [history, setHistory] = useState<PatientHistory | null>(null);
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    getHistory(patientId).then((h) => { setHistory(h); setNow(Date.now()); }).catch(() => setHistory(null));
  }, [patientId]);
  if (!history || now === null) return null;
  if (history.n_checkins === 0) {
    return <p className="mt-3 text-xs text-slate-500">No site photos yet.</p>;
  }
  const last = history.points[history.points.length - 1];
  const due = isCheckDue(history.points.map((p) => p.timestamp), now);
  const alerts = history.points.filter((p) => p.alert_triggered).length;
  return (
    <div className="mt-3 space-y-1 text-xs">
      <p className="text-slate-300">
        {history.n_checkins} check-in{history.n_checkins === 1 ? "" : "s"} · latest score {last.risk_score.toFixed(2)} · trend {history.latest_trend}
      </p>
      <p className="text-slate-500">Last photo {new Date(last.timestamp).toLocaleString()}</p>
      <div className="flex gap-2">
        {alerts > 0 && <span className="rounded bg-red-900/60 px-1.5 py-0.5 text-red-300">{alerts} alert{alerts === 1 ? "" : "s"}</span>}
        {due && <span className="rounded bg-amber-900/60 px-1.5 py-0.5 text-amber-300">site check due</span>}
      </div>
    </div>
  );
}
