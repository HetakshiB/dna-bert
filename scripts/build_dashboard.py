import base64
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ART_DIR = OUTPUTS_DIR
ART_DIR.mkdir(parents=True, exist_ok=True)

metrics = json.loads((OUTPUTS_DIR / "metrics.json").read_text(encoding="utf-8"))
classification = metrics["classification"]
our_auroc = float(classification["auroc"])
our_auprc = float(classification["auprc"])
our_f1 = float(classification["f1_at_0.5"])
brier_score = float(classification["brier_score"])
test_samples = int(metrics.get("test_samples", 0))
provenance = metrics.get("provenance", {})
evaluation_scope = provenance.get("evaluation_scope", "legacy report")
evaluation_time = provenance.get("evaluation_timestamp_utc", "timestamp unavailable")

def b64(img_name: str) -> str:
    path = OUTPUTS_DIR / img_name
    return base64.b64encode(path.read_bytes()).decode("ascii")

img_benchmark = b64("benchmark_comparison.png")
img_reliability = b64("reliability_diagram.png")
img_seq = b64("attribution_seq.png")
img_epigen = b64("attribution_epigen_3d.png")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CRISPR Multimodal Engine - Visual Dashboard</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{ background: #0f172a; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; transition: transform 0.2s; }}
    .card:hover {{ transform: translateY(-2px); border-color: #38bdf8; }}
    .badge {{ background: #0284c7; color: #fff; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700; }}
  </style>
</head>
<body class="p-6">
  <div class="max-w-6xl mx-auto space-y-8">
    <!-- Header -->
    <div class="border-b border-slate-700 pb-6">
      <div class="flex items-center justify-between">
        <div>
          <span class="badge uppercase tracking-wider mb-2 inline-block">Multimodal AI Engine</span>
          <h1 class="text-3xl font-extrabold text-white tracking-tight">CRISPR Off-Target Cleavage & Safety Predictor</h1>
          <p class="text-slate-400 mt-1">Unified 4-Modality Architecture: DNA Sequence + 6 ENCODE Epigenetic Marks + 3D Genome (Hi-C) + Single-Cell RNA-seq</p>
        </div>
        <div class="text-right">
          <div class="text-2xl font-black text-emerald-400">{our_auroc:.4f} AUROC</div>
          <div class="text-xs text-slate-400 font-semibold">{test_samples} samples · {evaluation_scope} · {evaluation_time}</div>
        </div>
      </div>
    </div>

    <!-- 4 Images Visual Gallery Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <!-- Image 1 -->
      <div class="card p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h3 class="text-lg font-bold text-sky-400">1. Baseline Benchmark Comparison</h3>
          <span class="text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded">Historical chart; reference values</span>
        </div>
        <div class="rounded-lg overflow-hidden bg-slate-900 border border-slate-800 p-2 flex items-center justify-center">
          <img src="data:image/png;base64,{img_benchmark}" alt="Baseline Benchmark Comparison" class="w-full h-auto rounded object-contain max-h-[300px]">
        </div>
        <p class="text-xs text-slate-300 leading-relaxed">
          Current model AUROC is <strong>{our_auroc:.4f}</strong> from metrics.json. Published baseline values shown here are references only; baseline implementations are not run by this dashboard.
        </p>
      </div>

      <!-- Image 2 -->
      <div class="card p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h3 class="text-lg font-bold text-sky-400">2. Reliability & Calibration Curve</h3>
          <span class="text-xs font-semibold bg-blue-950 text-blue-300 border border-blue-800 px-2 py-0.5 rounded">Brier: {brier_score:.4f}</span>
        </div>
        <div class="rounded-lg overflow-hidden bg-slate-900 border border-slate-800 p-2 flex items-center justify-center">
          <img src="data:image/png;base64,{img_reliability}" alt="Reliability Diagram" class="w-full h-auto rounded object-contain max-h-[300px]">
        </div>
        <p class="text-xs text-slate-300 leading-relaxed">
          Reliability diagram from the latest quantitative evaluation: 10 uniform probability bins compared with observed positive fractions.
        </p>
      </div>

      <!-- Image 3 -->
      <div class="card p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h3 class="text-lg font-bold text-sky-400">3. Sequence Integrated Gradients</h3>
          <span class="text-xs font-semibold bg-purple-950 text-purple-300 border border-purple-800 px-2 py-0.5 rounded">Captum Attribution</span>
        </div>
        <div class="rounded-lg overflow-hidden bg-slate-900 border border-slate-800 p-2 flex items-center justify-center">
          <img src="data:image/png;base64,{img_seq}" alt="Sequence Attribution" class="w-full h-auto rounded object-contain max-h-[300px]">
        </div>
        <p class="text-xs text-slate-300 leading-relaxed">
          Stored attribution image from a prior evaluation. Regenerate it for the current checkpoint; latent attribution alone does not establish biological causality.
        </p>
      </div>

      <!-- Image 4 -->
      <div class="card p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h3 class="text-lg font-bold text-sky-400">4. Epigenetic & 3D Genome SHAP</h3>
          <span class="text-xs font-semibold bg-amber-950 text-amber-300 border border-amber-800 px-2 py-0.5 rounded">KernelExplainer</span>
        </div>
        <div class="rounded-lg overflow-hidden bg-slate-900 border border-slate-800 p-2 flex items-center justify-center">
          <img src="data:image/png;base64,{img_epigen}" alt="SHAP Feature Importance" class="w-full h-auto rounded object-contain max-h-[300px]">
        </div>
        <p class="text-xs text-slate-300 leading-relaxed">
          Stored attribution image from a prior evaluation. Regenerate it for the current checkpoint and review feature-availability metadata before interpreting it.
        </p>
      </div>
    </div>

    <!-- Complete Benchmark Table -->
    <div class="card p-6 space-y-4">
      <h3 class="text-xl font-bold text-white">Literature-reported reference metrics</h3>
      <p class="text-amber-300 text-sm">Baseline implementations were not independently run. Published datasets and protocols can differ; this table is descriptive and does not establish head-to-head superiority or target attainment.</p>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-sm text-slate-300 border-collapse">
          <thead>
            <tr class="border-b border-slate-700 text-xs font-bold text-slate-400 uppercase tracking-wider">
              <th class="py-3 px-4">Model</th>
              <th class="py-3 px-4">Modalities</th>
              <th class="py-3 px-4">Architecture</th>
              <th class="py-3 px-4">Reported AUROC</th>
              <th class="py-3 px-4">Reported AUPRC</th>
              <th class="py-3 px-4">Reported F1</th>
              <th class="py-3 px-4">Provenance</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800">
            <tr>
              <td class="py-3 px-4 font-semibold text-slate-200">CRISPR-DIPOFF</td>
              <td class="py-3 px-4 text-xs">Sequence</td>
              <td class="py-3 px-4 text-xs">BiLSTM + Attention</td>
              <td class="py-3 px-4 font-mono">0.7820</td>
              <td class="py-3 px-4 font-mono">0.6240</td>
              <td class="py-3 px-4 font-mono">0.6850</td>
              <td class="py-3 px-4">Literature-reported reference</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-slate-200">CRISPR-Net</td>
              <td class="py-3 px-4 text-xs">Sequence (mismatch+indel)</td>
              <td class="py-3 px-4 text-xs">Recurrent CNN</td>
              <td class="py-3 px-4 font-mono">0.8040</td>
              <td class="py-3 px-4 font-mono">0.6480</td>
              <td class="py-3 px-4 font-mono">0.7120</td>
              <td class="py-3 px-4">Literature-reported reference</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-slate-200">DNABERT-6 (Vanilla)</td>
              <td class="py-3 px-4 text-xs">Sequence (6-mers)</td>
              <td class="py-3 px-4 text-xs">12-layer Transformer</td>
              <td class="py-3 px-4 font-mono">0.8160</td>
              <td class="py-3 px-4 font-mono">0.6650</td>
              <td class="py-3 px-4 font-mono">0.7280</td>
              <td class="py-3 px-4">Literature-reported reference</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-slate-200">EpiCRISPR</td>
              <td class="py-3 px-4 text-xs">Sequence + Epigenetics</td>
              <td class="py-3 px-4 text-xs">BiLSTM + Epigenetic Dense</td>
              <td class="py-3 px-4 font-mono">0.8290</td>
              <td class="py-3 px-4 font-mono">0.6830</td>
              <td class="py-3 px-4 font-mono">0.7410</td>
              <td class="py-3 px-4">Literature-reported reference</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-slate-200">DNABERT-Epi</td>
              <td class="py-3 px-4 text-xs">Sequence + Epigenetics</td>
              <td class="py-3 px-4 text-xs">DNABERT + MLP Fusion</td>
              <td class="py-3 px-4 font-mono">0.8450</td>
              <td class="py-3 px-4 font-mono">0.7090</td>
              <td class="py-3 px-4 font-mono">0.7620</td>
              <td class="py-3 px-4">Literature-reported reference</td>
            </tr>
            <tr class="bg-sky-950/40 border-l-4 border-sky-400">
              <td class="py-3 px-4 font-bold text-white">Multimodal CRISPR (Ours)</td>
              <td class="py-3 px-4 text-xs font-semibold text-sky-300">Seq + Epigen + 3D + scRNA</td>
              <td class="py-3 px-4 text-xs font-semibold text-sky-300">4-Branch Fusion + Dual Heads</td>
              <td class="py-3 px-4 font-mono font-bold text-emerald-400">{our_auroc:.4f}</td>
              <td class="py-3 px-4 font-mono font-bold text-emerald-400">{our_auprc:.4f}</td>
              <td class="py-3 px-4 font-mono font-bold text-emerald-400">{our_f1:.4f}</td>
              <td class="py-3 px-4 text-sky-400 font-extrabold">Current metrics.json</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Ready-to-Run Commands -->
    <div class="card p-6 space-y-4">
      <h3 class="text-xl font-bold text-white">Terminal Commands to Run Locally</h3>
      <div class="space-y-3">
        <div>
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wide">In Git Bash:</span>
          <pre class="mt-1 p-3 bg-slate-950 text-sky-300 rounded text-xs font-mono overflow-x-auto border border-slate-800">source .venv/Scripts/activate &amp;&amp; python scripts/test_dataloaders.py &amp;&amp; python train.py --smoke-test &amp;&amp; python evaluate.py --sample 100 &amp;&amp; python scripts/compare_benchmarks.py</pre>
        </div>
        <div>
          <span class="text-xs font-bold text-slate-400 uppercase tracking-wide">In Windows PowerShell:</span>
          <pre class="mt-1 p-3 bg-slate-950 text-emerald-300 rounded text-xs font-mono overflow-x-auto border border-slate-800">.\\.venv\\Scripts\\Activate.ps1; python scripts/test_dataloaders.py; python train.py --smoke-test; python evaluate.py --sample 100; python scripts/compare_benchmarks.py</pre>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""

dest = ART_DIR / "crispr_dashboard.html"
dest.write_text(html, encoding="utf-8")
print(f"Generated {dest} ({len(html)} bytes)")
