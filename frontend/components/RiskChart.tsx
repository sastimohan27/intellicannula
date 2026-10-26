"use client";

import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
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
    alert: p.alert_triggered,
    degraded: p.registration_degraded,
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.15} />
        <XAxis dataKey="index" tick={{ fontSize: 12 }}
               label={{ value: "check-in", position: "insideBottom", offset: -4, fontSize: 12 }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 12 }}
               label={{ value: "redness score (0-1)", angle: -90, position: "insideLeft", fontSize: 11 }} />
        <Tooltip
          formatter={(value) => (typeof value === "number" ? value.toFixed(3) : String(value ?? ""))}
          labelFormatter={(i) => data[Number(i) - 1]?.time ?? ""}
          contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6, fontSize: 12 }}
          labelStyle={{ color: "#94a3b8" }}
        />
        <ReferenceLine y={0.6} stroke="#dc2626" strokeDasharray="4 4"
                       label={{ value: "alert 0.6", fontSize: 11, fill: "#dc2626" }} />
        {/* Alert points are coloured via the line's own dots. A separate <Scatter>
            series would add its own rows (x index, unnamed value) to the tooltip. */}
        <Line type="linear" dataKey="risk" name="redness score" stroke="#2563eb" strokeWidth={2}
              isAnimationActive={false}
              dot={({ cx, cy, payload, index }) => (
                <circle key={index} cx={cx} cy={cy} r={payload.alert ? 4 : 3}
                        fill={payload.alert ? "#dc2626" : "#2563eb"} stroke="none" />
              )} />
      </LineChart>
    </ResponsiveContainer>
  );
}
