import { useEffect, useMemo, useState, type FormEvent } from "react"
import { Activity, AlertCircle, ArrowRight, Cpu, Dna, Layers3, LoaderCircle, Radio, ShieldAlert } from "lucide-react"
import type { HealthResponse, ModelInfoResponse, PredictionResponse } from "./types"

const API = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "")
const MISSING_MODALITIES = [
  { key: "epigenetic", label: "Epigenetic features", detail: "Six epigenetic inputs are omitted." },
  { key: "3d_genome", label: "3D genome features", detail: "Loop, TAD, and compartment inputs are omitted." },
  { key: "singlecell_expression", label: "Single-cell expression", detail: "Expression input is omitted." },
] as const

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [model, setModel] = useState<ModelInfoResponse | null>(null)
  const [sequence, setSequence] = useState("")
  const [consent, setConsent] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [errors, setErrors] = useState<string[]>([])
  const [apiError, setApiError] = useState("")
  const [loading, setLoading] = useState(false)
  const seq = sequence.trim().replace(/\s/g, "").toUpperCase()

  useEffect(() => {
    let live = true
    async function load() {
      try {
        const response = await fetch(API + "/health")
        const status = await response.json() as HealthResponse
        if (!live) return
        setHealth(status)
        if (status.status === "ready") {
          const modelResponse = await fetch(API + "/model-info")
          if (modelResponse.ok && live) setModel(await modelResponse.json() as ModelInfoResponse)
        }
      } catch {
        if (live) setHealth({ status: "not_ready", device: null, error: "Could not reach FastAPI. Start the backend, then refresh." })
      }
    }
    void load()
    return () => { live = false }
  }, [])

  const seqErrors = useMemo(() => {
    const issues: string[] = []
    if (!seq) issues.push("Enter a DNA sequence.")
    if (seq && seq.length < 6) issues.push("DNABERT-6 requires at least 6 nucleotides.")
    if (seq.length > 10000) issues.push("Sequence exceeds the 10,000 nucleotide limit.")
    if (/[^ACGTN]/.test(seq)) issues.push("Use only A, C, G, T, and N.")
    return issues
  }, [seq])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const issues = [...seqErrors]
    if (!consent) issues.push("Consent is required before zero-filled missing-feature inference.")
    setErrors(issues)
    setApiError("")
    setResult(null)
    if (issues.length) return

    const body = {
      sequence: seq,
      mode: "missing_feature_demo",
      allow_missing_features: true,
    }
    setLoading(true)
    try {
      const response = await fetch(API + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      const data = await response.json()
      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item: { msg?: string }) => item.msg).join(" ")
          : data.detail
        throw new Error(typeof detail === "string" ? detail : `Prediction request failed (${response.status}).`)
      }
      setResult(data as PredictionResponse)
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Prediction request failed.")
    } finally {
      setLoading(false)
    }
  }

  const down = health?.status !== "ready"
  return <div className="min-h-screen bg-canvas text-ink">
    <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1440px] items-center justify-between px-4 py-3 sm:px-7 lg:px-10">
        <div className="flex items-center gap-3">
          <div className="brand-mark"><Dna size={20} /></div>
          <div><b className="text-sm">CRISPR <span className="text-teal">Atlas</span></b><div className="eyebrow">Inference workspace</div></div>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden text-right sm:block"><div className="eyebrow">Backbone</div><div className="text-xs font-semibold">{model?.pretrained_backbone || "DNABERT-6"}</div></div>
          <div className={"status-pill " + (down ? "status-wait" : "status-good")}>
            <span className={"status-dot " + (down ? "bg-amber-500" : "bg-emerald-500")} />{down ? "API unavailable" : "API ready"}
          </div>
        </div>
      </div>
    </header>

    <main className="mx-auto max-w-[1440px] px-4 pb-12 pt-7 sm:px-7 lg:px-10 lg:pt-10">
      <section className="hero-panel relative mb-8 overflow-hidden rounded-3xl px-6 py-7 text-white sm:px-9 sm:py-9">
        <div className="hero-orb orb-one" /><div className="hero-orb orb-two" />
        <div className="relative z-10 grid gap-7 md:grid-cols-[1fr_auto] md:items-end">
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[.17em] text-amber-100"><ShieldAlert size={13} />Missing Feature Demo</div>
            <h1 className="text-2xl font-semibold leading-tight tracking-tight sm:text-4xl">Explore a sequence-only<br className="hidden sm:block" /> CRISPR model demonstration.</h1>
            <p className="mt-3 max-w-xl text-sm leading-6 text-slate-300">Submit a held-out or demonstration DNA sequence. The multimodal model runs with its non-sequence inputs omitted and zero-filled.</p>
          </div>
          <div className="flex gap-4"><Stat label="Input window" value="256 tokens" /><Stat label="Prediction heads" value="Regression + risk" /></div>
        </div>
      </section>

      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div><div className="eyebrow mb-1">Prediction workspace</div><h2 className="text-lg font-bold">Sequence demonstration</h2></div>
        <div className="flex items-center gap-2 text-xs text-slate-500"><span className={"status-dot " + (down ? "bg-amber-500" : "bg-emerald-500")} />{down ? health?.error || "Checking backend status" : `Connected · ${health?.device || "device ready"}`}</div>
      </div>

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1.55fr)_minmax(340px,.85fr)]">
        <form onSubmit={submit} noValidate className="space-y-5">
          <section className="panel-card p-5 sm:p-6">
            <Heading icon={<Dna size={17} />} number="01" title="Target sequence" sub="DNA sequence encoded with the cached DNABERT-6 tokenizer." tag="A · C · G · T · N" />
            <textarea aria-label="DNA target sequence" rows={4} spellCheck={false} value={sequence} onChange={event => { setSequence(event.target.value); setResult(null) }} placeholder="Paste DNA sequence" className="form-input min-h-28 resize-y font-mono text-sm leading-6 tracking-wide" />
            <div className="mt-2 flex flex-wrap justify-between gap-2 text-[11px] text-slate-400"><span>{seq.length.toLocaleString()} nucleotides · whitespace ignored</span><span>Max 10,000 nt · 256-token window</span></div>
            {sequence && seqErrors.length > 0 && <p className="mt-2 flex items-center gap-2 text-xs text-rose-600"><AlertCircle size={14} />{seqErrors[0]}</p>}
          </section>

          <section className="panel-card p-5 sm:p-6">
            <Heading icon={<Layers3 size={17} />} number="02" title="Missing Feature Demo mode" sub="This demonstration sends the sequence only; no other mode is available in this interface." />
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-bold text-amber-900"><ShieldAlert size={15} />Mode fixed: Missing Feature Demo</div>
            <div role="alert" className="rounded-xl border-2 border-amber-300 bg-amber-50 p-4 text-sm leading-6 text-amber-950 shadow-sm">
              <b className="flex items-center gap-2"><AlertCircle size={17} />Scientific limitation</b>
              <p className="mt-2">Epigenetic, 3D genome, and single-cell modalities are zero-filled placeholders in this request, not biological measurements. These are <b>not full multimodal predictions</b> and are <b>not experimentally validated biological results</b>. Treat outputs as an exploratory software demonstration only.</p>
            </div>
            <div className="mt-4 space-y-2" aria-label="Unavailable modalities">
              {MISSING_MODALITIES.map(item => <div key={item.key} className="flex items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-700"><AlertCircle size={12} /></span>
                <div><b className="text-xs text-slate-800">{item.label} · unavailable</b><p className="mt-0.5 text-[11px] leading-4 text-slate-500">{item.detail} The API uses a zero placeholder.</p></div>
              </div>)}
            </div>
          </section>

          <label className="consent-box">
            <input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} className="mt-0.5 h-4 w-4 rounded border-amber-400 text-amber-600 focus:ring-amber-500" />
            <span><b>I understand and consent:</b> the three listed modalities are omitted and zero-filled, these placeholders are not measurements, and the outputs are not experimentally validated biological results.</span>
          </label>

          {(errors.length > 0 || apiError) && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
            <b className="flex items-center gap-2"><AlertCircle size={15} />{apiError ? "Request failed" : "Review the form"}</b>
            {apiError ? <p className="mt-1 text-xs">{apiError}</p> : <ul className="ml-6 mt-1 list-disc text-xs leading-5">{errors.map(error => <li key={error}>{error}</li>)}</ul>}
          </div>}
          {down && <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900"><b>Backend not ready.</b> Start FastAPI at port 8000. {health?.error}</div>}
          <button type="submit" disabled={loading || down} className="primary-button w-full sm:w-auto">{loading ? <><LoaderCircle size={17} className="animate-spin" />Running inference...</> : <>Run missing-feature demo <ArrowRight size={17} /></>}</button>
          <p className="text-[11px] text-slate-400">Inference only · no training or biological feature lookup is performed here.</p>
        </form>

        <aside className="space-y-5 xl:sticky xl:top-24">
          <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-card">
            <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4"><div><div className="eyebrow">Output</div><h2 className="mt-1 text-sm font-bold">Prediction summary</h2></div><span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold uppercase text-slate-500">{result ? "Complete" : "Awaiting input"}</span></div>
            {loading ? <div className="flex min-h-80 flex-col items-center justify-center p-8 text-center"><LoaderCircle size={31} className="animate-spin text-teal" /><p className="mt-4 text-sm font-semibold">Processing sequence</p><p className="mt-1 max-w-xs text-xs leading-5 text-slate-500">DNABERT-6 is encoding the input and the model is calculating outputs.</p></div> : result ? <div className="p-5">
              {result.warning && <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-950"><b className="mb-1 flex items-center gap-2"><AlertCircle size={14} />Incomplete-feature result</b>{result.warning}</div>}
              <div className="metric-highlight"><div className="flex items-center gap-2 text-xs font-medium text-teal-900"><Activity size={15} />Predicted cleavage frequency</div><div className="mt-2 font-mono text-3xl font-semibold">{result.predicted_cleavage_frequency.toPrecision(5)}</div><div className="mt-1 text-[10px] text-slate-500">Regression output · not a measurement</div></div>
              <div className="mt-3 grid grid-cols-2 gap-3"><div className="metric-tile"><div className="text-[10px] font-semibold uppercase text-slate-400">High-efficiency probability</div><div className="mt-2 text-2xl font-bold">{(result.high_efficiency_probability * 100).toFixed(1)}%</div><div className="probability-track"><span style={{ width: Math.min(100, Math.max(0, result.high_efficiency_probability * 100)) + "%" }} /></div></div><div className="metric-tile"><div className="text-[10px] font-semibold uppercase text-slate-400">Classification</div><div className={"mt-2 inline-flex rounded-full px-2.5 py-1 text-xs font-bold " + (result.high_efficiency_prediction ? "bg-rose-50 text-rose-700" : "bg-emerald-50 text-emerald-700")}>{result.high_efficiency_prediction ? "High efficiency" : "Lower efficiency"}</div><div className="mt-2 text-[10px] text-slate-400">Threshold: 0.50</div></div></div>
              <div className="mt-4 rounded-xl border border-slate-100 p-4"><div className="mb-3 flex justify-between"><h3 className="text-xs font-bold">Feature availability</h3><span className="text-[10px] text-amber-700">Sequence only</span></div><div className="space-y-2">{MISSING_MODALITIES.map(modality => {
                const status = result.feature_availability[modality.key]
                return <div key={modality.key} className="flex items-start gap-2"><span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-amber-100 text-amber-700"><AlertCircle size={10} /></span><div><b className="text-[11px]">{modality.label} · unavailable</b><p className="text-[10px] leading-4 text-slate-500">{status?.provenance || "Zero-filled placeholder; not measured."}</p></div></div>
              })}</div></div>
              <div className="mt-4 flex justify-between border-t border-slate-100 pt-3 text-[10px] text-slate-400">{result.sequence_nucleotides} nt · {result.sequence_tokens_used} attended tokens<span>{result.sequence_length_tokens}-token input</span></div>
            </div> : <div className="flex min-h-80 flex-col items-center justify-center p-8 text-center"><div className="empty-illustration"><Dna size={27} /></div><p className="mt-5 text-sm font-semibold">Your result will appear here</p><p className="mt-1 max-w-xs text-xs leading-5 text-slate-500">Enter a valid DNA sequence and consent to run the missing-feature demonstration.</p><div className="mt-5 flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-[10px] text-slate-500"><Radio size={13} className="text-teal" />Sequence only · 256 tokens</div></div>}
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-card"><div className="mb-4 flex items-center gap-2"><Cpu size={16} className="text-teal" /><h3 className="text-xs font-bold">Model & checkpoint</h3></div>{model ? <div className="space-y-3 text-[11px]"><Info label="Architecture" value={model.architecture} /><Info label="Backbone" value={model.pretrained_backbone} /><Info label="Device" value={model.device.toUpperCase()} /><Info label="Parameters" value={(model.parameter_count / 1e6).toFixed(1) + "M"} /><Info label="Checkpoint" value={model.checkpoint_path} /><Info label="SHA-256" value={model.checkpoint_sha256.slice(0, 14) + "…"} /></div> : <p className="text-xs leading-5 text-slate-500">Model metadata appears when the backend has loaded its checkpoint.</p>}</section>
          <div className="flex items-start gap-2 px-1 text-[10px] leading-4 text-slate-400"><ShieldAlert size={13} className="mt-0.5 shrink-0" />Missing modalities are zero-filled. Predictions are not experimentally validated.</div>
        </aside>
      </div>
    </main>

    <footer className="border-t border-slate-200 bg-white"><div className="mx-auto flex max-w-[1440px] flex-wrap justify-between gap-2 px-4 py-4 text-[10px] text-slate-400 sm:px-7 lg:px-10"><span>CRISPR Atlas · Missing Feature Demo</span><span className="flex items-center gap-1.5"><Radio size={12} />Local API · no feature lookup</span></div></footer>
  </div>
}

function Heading({ icon, number, title, sub, tag }: { icon: React.ReactNode; number: string; title: string; sub: string; tag?: string }) {
  return <div className="mb-5 flex items-start justify-between gap-3"><div className="flex items-start gap-3"><div className="section-icon">{icon}</div><div><div className="flex flex-wrap items-center gap-2"><span className="eyebrow">{number}</span><h2 className="text-sm font-bold">{title}</h2></div><p className="mt-1 text-xs leading-5 text-slate-500">{sub}</p></div></div>{tag && <span className="hidden rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-500 sm:block">{tag}</span>}</div>
}
function Stat({ label, value }: { label: string; value: string }) { return <div className="hero-stat"><span>{label}</span><b>{value}</b></div> }
function Info({ label, value }: { label: string; value: string }) { return <div className="flex justify-between gap-3"><span className="shrink-0 text-slate-400">{label}</span><span title={value} className="truncate text-right font-medium">{value}</span></div> }
