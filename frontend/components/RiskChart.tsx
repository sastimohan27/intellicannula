"use client";

import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import type { HistoryPoint } from "@/lib/types";

/**
 * Risk over time, with alert points highlighted.
 *
 * The y-axis is pinned to [0, 1] rather than auto-scaled. An auto-scaled axis
 * makes a flat, low-risk series look like a dramatic climb, which is exactly
 * the wrong impression for a monitoring chart.
 */
export default function RiskChart({ points }: { points: HistoryPoint[] }) {
  if (points.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-slate-500">
        No check-ins yet. Upload a site photo to start the trend.
      </p>
    );
  }

  const data = points.map((p, i) => ({
    index: i + 1,
    risk: Number(p.risk_score.toFixed(4)),
    time: new Date(p.timestamp).toLocaleString(),
    alert: p.alert_triggered ? Number(p.risk_score.toFixed(4)) : null,
    degraded: p.registration_degraded,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.15} />
        <XAxis dataKey="index" tick={{ fontSize: 12 }}
               label={{ value: "check-in", position: "insideBottom", offset: -4, fontSize: 12 }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
        <Tooltip
          formatter={(value) => (typeof value === "number" ? value.toFixed(3) : String(value ?? ""))}
          labelFormatter={(i) => data[Number(i) - 1]?.time ?? ""}
        />
        <ReferenceLine y={0.6} stroke="#dc2626" strokeDasharray="4 4"
                       label={{ value: "alert 0.6", fontSize: 11, fill: "#dc2626" }} />
        <Line type="monotone" dataKey="risk" stroke="#2563eb" strokeWidth={2} dot={{ r: 3 }} />
        <Scatter dataKey="alert" fill="#dc2626" shape="circle" />
      </LineChart>
    </ResponsiveContainer>
  );
}
