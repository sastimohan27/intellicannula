"use client";

import { useEffect, useState } from "react";
import type { VeinScan } from "@/lib/types";

/** Turn the white-on-black mask PNG into a red, alpha-only layer for the overlay. */
function useRedMask(maskB64: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!maskB64) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height);
      for (let i = 0; i < data.data.length; i += 4) {
        const on = data.data[i] > 127;
        data.data[i] = 239; data.data[i + 1] = 68; data.data[i + 2] = 68;
        data.data[i + 3] = on ? 255 : 0;
      }
      ctx.putImageData(data, 0, 0);
      if (!cancelled) setUrl(canvas.toDataURL());
    };
    img.src = `data:image/png;base64,${maskB64}`;
    return () => { cancelled = true; };
  }, [maskB64]);
  return maskB64 ? url : null;
}

const SOURCE_LABEL: Record<string, string> = {
  cubital_dataset: "CUBITAL NIR dataset frame",
  registered_device: "Registered NIR device",
};

export default function VeinResult({ scan, originalUrl }: { scan: VeinScan; originalUrl: string | null }) {
  const redMask = useRedMask(scan.mask_png_base64);
  const [opacity, setOpacity] = useState(0.7);
  // The model sees the image stretched to 512x512, so the mask maps back by the
  // same stretch: both layers fill the same box. Marker coordinates are in that
  // 512x512 frame too.
  const marker = scan.insertion_x !== null && scan.insertion_y !== null
    ? { left: `${(scan.insertion_x / 512) * 100}%`, top: `${(scan.insertion_y / 512) * 100}%` }
    : null;

  return (
    <div className="space-y-4 rounded-lg border border-slate-200 p-4 dark:border-slate-700">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded bg-emerald-700 px-2 py-1 font-semibold text-white">NIR verified</span>
        <span className="rounded border border-slate-600 px-2 py-1">
          {SOURCE_LABEL[scan.acquisition_source ?? ""] ?? scan.acquisition_source}
          {scan.dataset_split && ` · ${scan.dataset_split} split`}
        </span>
        {scan.dataset_split === "train" && (
          <span className="rounded border border-amber-600 px-2 py-1 text-amber-400">
            training image — not a held-out result
          </span>
        )}
        {scan.inference_ms !== null && (
          <span className="text-slate-500">inference {Math.round(scan.inference_ms)} ms</span>
        )}
        <span className="text-slate-500">model {scan.model_version}</span>
      </div>

      {scan.insertion_x === null ? (
        <div>
          <p className="font-medium text-amber-700 dark:text-amber-400">No reliable vein detected — no insertion site suggested</p>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">{scan.reason}</p>
        </div>
      ) : (
        <div className="flex flex-wrap gap-4 text-sm">
          <span>Suggested site <strong>({scan.insertion_x}, {scan.insertion_y})</strong> px</span>
          <span>Vein width <strong>{scan.vein_width_px?.toFixed(1)} px</strong></span>
          {scan.vein_grade && <span>Grade <strong>{scan.vein_grade}</strong></span>}
          <span>Angle <strong>{scan.angle_deg?.toFixed(1)}°</strong> <span className="text-slate-500">(heuristic)</span></span>
          <span className="text-slate-500">
            Selector confidence <strong>{scan.confidence.toFixed(2)}</strong>{scan.reason && ` · ${scan.reason}`}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Panel title="NIR input">
          {originalUrl && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={originalUrl} alt="Uploaded NIR image" className="absolute inset-0 h-full w-full" />
          )}
        </Panel>
        <Panel title="Predicted vein mask">
          {scan.mask_png_base64 && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={`data:image/png;base64,${scan.mask_png_base64}`} alt="Binary vein mask" className="absolute inset-0 h-full w-full" />
          )}
        </Panel>
        <Panel title="Overlay">
          {originalUrl && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={originalUrl} alt="" className="absolute inset-0 h-full w-full" />
          )}
          {redMask && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={redMask} alt="Vein mask over the NIR image" style={{ opacity }}
                 className="absolute inset-0 h-full w-full" />
          )}
          {marker && (
            <span className="absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-lime-400"
                  style={marker} aria-label="Suggested insertion site" />
          )}
        </Panel>
      </div>
      <label className="flex items-center gap-3 text-xs text-slate-500">
        Mask opacity
        <input type="range" min={0} max={1} step={0.05} value={opacity}
               onChange={(e) => setOpacity(Number(e.target.value))} className="w-40" />
        {Math.round(opacity * 100)}%
      </label>

      <p className="text-xs text-slate-500">
        The overlay is the model&apos;s binary mask, unmodified. Test-set Dice 0.567 (288 images):
        it misses some veins and marks some non-veins. The angle is a heuristic, not a depth
        measurement. Advisory only — confirm before acting.
      </p>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <figure>
      <div className="relative aspect-square w-full overflow-hidden rounded bg-black">{children}</div>
      <figcaption className="mt-1 text-xs text-slate-500">{title}</figcaption>
    </figure>
  );
}
