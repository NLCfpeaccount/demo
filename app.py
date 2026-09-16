# Agentic ATS — Validation & Feasibility Demonstrator
# Probes RAM/CPU/disk/throughput of the full agentic loop inside THIS container.
# Not the product: it is the go/no-go gate before QLoRA fine-tuning begins.
import gc, io, json, os, re, shutil, time, random, zlib
import psutil
import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, ValidationError

_PROC   = psutil.Process(os.getpid())
rss_mb  = lambda: _PROC.memory_info().rss / 1e6
cpu_pct = lambda: psutil.cpu_percent(interval=None)

def mem_limit_mb():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 2**60:
                return int(v) / 1e6
        except Exception:
            pass
    return 1024.0

def disk_free_mb():
    return shutil.disk_usage("/").free / 1e6

def snap(stage, note=""):
    st.session_state.metrics.append({
        "step": len(st.session_state.metrics), "stage": stage,
        "rss_mb": round(rss_mb(), 1), "cpu_pct": cpu_pct(),
        "disk_free_mb": round(disk_free_mb(), 0), "note": note})

# ---------------- schema, grammar, synthetic data ----------------
SCHEMA_PROMPT = ("You are a technical-recruiter extraction engine. Return ONLY valid JSON with keys: "
                 "name (str), top_skills (list[str]), years_experience (number), education (str), "
                 "key_projects (list[str]), summary (str).")

class Profile(BaseModel):
    name: str
    top_skills: list[str]
    years_experience: float
    education: str
    key_projects: list[str]
    summary: str

def get_grammar():
    if "grammar" not in st.session_state:
        try:
            from llama_cpp import LlamaGrammar
            st.session_state.grammar = LlamaGrammar.from_json_schema(
                json.dumps(Profile.model_json_schema()))
        except Exception:
            st.session_state.grammar = None
    return st.session_state.grammar

NAMES  = ["Asha R.", "Kiran V.", "Meera S.", "Dev P.", "Ira N.", "Ravi T.", "Zoya K.", "Arun M."]
SKILLS = ["Python", "SQL", "PyTorch", "Pandas", "SHAP", "LightGBM", "NumPy", "Streamlit", "Docker", "Git"]

def synth_resume(i):                      # PARSE stage (synthetic; real PDFs in v1)
    rnd = random.Random(i)
    body = "\n".join([
        f"Name: {rnd.choice(NAMES)}", f"Experience: {round(rnd.uniform(0.5, 6.0), 1)} years",
        f"Skills: {', '.join(rnd.sample(SKILLS, 5))}",
        "Education: B.Tech " + rnd.choice(["AI&DS", "CSE", "ECE"]),
        "Projects: " + "; ".join(f"proj{j}: built {rnd.choice(SKILLS)} pipeline" for j in range(3)),
        "Summary: " + rnd.choice(["ML generalist", "CV engineer", "Data scientist"]) + " with production exposure."])
    return body * rnd.randint(1, 3)

# ---------------- agent stages ----------------
def extract_one(llm, text, grammar):      # EXTRACT + VALIDATE + retry
    msgs = [{"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user",   "content": text[:6000]}]
    u, secs = {}, 0.0
    for attempt in (1, 2):
        kw = dict(temperature=0.0, max_tokens=384)
        if grammar is not None:
            kw["grammar"] = grammar
        t0 = time.time()
        r = llm.create_chat_completion(messages=msgs, **kw)
        secs = time.time() - t0
        raw = r["choices"][0]["message"]["content"]
        u = r.get("usage", {})
        try:
            return Profile.model_validate_json(raw), attempt, u, secs
        except ValidationError as e:
            msgs += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": f"Invalid JSON: {e.errors()[:1]}. Return ONLY corrected valid JSON."}]
    return None, 2, u, secs

JD = "Seeking ML engineer: Python, PyTorch, deployment, explainability."

def judge(llm, prof, mode):               # JUDGE stage
    if mode == "mock (zero-LLM)":
        return 50 + zlib.crc32(prof.name.encode()) % 50, "mock deterministic score"
    if mode.startswith("groq") and "GROQ_API_KEY" in st.secrets:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}"},
                          json={"model": "llama-3.1-8b-instant", "temperature": 0.0, "max_tokens": 80,
                                "messages": [
                                    {"role": "system", "content": "Score 0-100 against the JD. Reply: SCORE: <int> REASON: <one sentence>"},
                                    {"role": "user",   "content": JD + "\n" + prof.model_dump_json()}]},
                          timeout=60)
        txt = r.json()["choices"][0]["message"]["content"]
    else:
        r = llm.create_chat_completion(temperature=0.0, max_tokens=80, messages=[
            {"role": "system", "content": "Score 0-100 against the JD. Reply: SCORE: <int> REASON: <one sentence>"},
            {"role": "user",   "content": JD + "\n" + prof.model_dump_json()}])
        txt = r["choices"][0]["message"]["content"]
    m = re.search(r"(\d{1,3})", txt)
    return (int(m.group(1)) if m else 0), txt.strip()[:120]

def run_loop(K, judge_mode, use_grammar):
    llm = st.session_state.llm
    grammar = get_grammar() if use_grammar else None
    snap("pre_loop", f"grammar={'on' if grammar else 'off'} judge={judge_mode}")
    bar = st.progress(0.0)
    for i in range(K):
        text = synth_resume(i)
        snap(f"parse_{i}", f"{len(text)} chars")
        prof, attempts, u, secs = extract_one(llm, text, grammar)
        ct = u.get("completion_tokens", 0)
        score, _ = judge(llm, prof, judge_mode) if prof else (0, "extraction failed")
        st.session_state.results.append({
            "resume": i, "valid": prof is not None, "attempts": attempts,
            "prompt_tok": u.get("prompt_tokens", 0), "compl_tok": ct,
            "secs": round(secs, 2), "tok_per_s": round(ct / secs, 1) if secs else 0,
            "score": score, "rss_mb": round(rss_mb(), 1)})
        bar.progress((i + 1) / K)
    snap("post_loop")

# ---------------- UI ----------------
st.set_page_config(page_title="ATS Validation Demonstrator", layout="wide")
st.title("🧪 Agentic ATS — Validation & Feasibility Demonstrator")
st.caption("Go/no-go gate: proves the agentic loop survives this container's memory ceiling BEFORE any fine-tuning effort is spent.")

st.session_state.setdefault("metrics", [])
st.session_state.setdefault("results", [])
if not st.session_state.metrics:
    snap("baseline_imports")
LIMIT = mem_limit_mb()

with st.sidebar:
    st.header("Model source")
    src = st.radio("Source", ["base (Qwen official)", "custom (your HF repo)"])
    if src.startswith("base"):
        repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        fname = st.selectbox("GGUF quant", ["qwen2.5-0.5b-instruct-q4_k_m.gguf",
                                            "qwen2.5-0.5b-instruct-q2_k.gguf",
                                            "qwen2.5-0.5b-instruct-q8_0.gguf"])
    else:
        repo  = st.text_input("Repo ID", "naresh-cl/ats-student-gguf")
        fname = st.text_input("GGUF file", "student-q4_k_m.gguf")
    st.header("Probe config")
    n_ctx      = st.select_slider("n_ctx", [512, 1024, 2048, 4096], value=2048)
    n_threads  = st.select_slider("n_threads", [1, 2, 4], value=2)
    K          = st.select_slider("Resumes per run", [1, 5, 10, 20], value=5)
    q8_kv      = st.checkbox("q8_0 KV cache (halves KV RAM)")
    use_gram   = st.checkbox("Grammar-constrained JSON (kills retries)")
    judge_mode = st.radio("Judge", ["mock (zero-LLM)", "local SLM", "groq (needs secret)"])

st.metric("Container memory limit (cgroup)", f"{LIMIT:.0f} MB")

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("1 · Load model"):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        snap("pre_load")
        t0 = time.time()
        token = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None
        path = hf_hub_download(repo, fname, token=token)
        snap("post_download", f"file={os.path.getsize(path)/1e6:.0f}MB disk_free={disk_free_mb():.0f}MB")
        kw = dict(cache_type_k="q8_0", cache_type_q="q8_0") if q8_kv else {}
        st.session_state.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads,
                                     n_batch=256, verbose=False, **kw)
        st.session_state.cfg = dict(repo=repo, fname=fname, n_ctx=n_ctx,
                                    n_threads=n_threads, q8_kv=q8_kv)
        snap("post_load", f"{fname} ctx={n_ctx} thr={n_threads} q8kv={q8_kv} load={time.time()-t0:.1f}s")
with c2:
    if st.button("2 · Run agentic loop"):
        if "llm" not in st.session_state:
            st.warning("Load the model first."); st.stop()
        run_loop(K, judge_mode, use_gram)
with c3:
    if st.button("3 · Teardown"):
        st.session_state.pop("llm", None)
        st.session_state.pop("grammar", None)
        gc.collect()
        snap("post_teardown")
with c4:
    if st.button("4 · Measure fitz import"):
        import importlib
        snap("pre_fitz_import")
        importlib.import_module("fitz")
        snap("post_fitz_import")

st.subheader("Memory trace (RSS MB)")
dfm = pd.DataFrame(st.session_state.metrics)
st.line_chart(dfm.set_index("step")["rss_mb"])
st.dataframe(dfm, hide_index=True)

st.subheader("Per-resume results")
dfr = pd.DataFrame(st.session_state.results)
if len(dfr):
    st.dataframe(dfr, hide_index=True)
    buf = io.BytesIO()
    dfr.to_excel(buf, index=False)
    st.download_button("Download ranked.xlsx", buf.getvalue(),
                       file_name="ranked.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.session_state.metrics:
    peak      = max(m["rss_mb"] for m in st.session_state.metrics)
    post_load = next((m["rss_mb"] for m in st.session_state.metrics if m["stage"] == "post_load"), None)
    post_loop = next((m["rss_mb"] for m in st.session_state.metrics if m["stage"] == "post_loop"), None)
    leak = (post_loop - post_load) if (post_loop and post_load) else None
    tps  = dfr["tok_per_s"].mean() if len(dfr) else 0.0
    st.subheader("Verdict")
    st.write(f"Limit **{LIMIT:.0f} MB** · peak **{peak:.0f} MB** · headroom **{LIMIT-peak:.0f} MB** · "
             f"leak **{round(leak,1) if leak is not None else 'n/a'} MB** · mean gen **{tps:.1f} tok/s**")
    if len(dfr):
        st.write(f"Pre-fine-tune quality baseline (informational): validity **{100*dfr['valid'].mean():.0f}%**, "
                 f"mean attempts **{dfr['attempts'].mean():.2f}** — keep these numbers; they are your post-QLoRA comparison.")
    ok = peak < LIMIT - 75 and (leak is None or leak < 60) and tps >= 2
    (st.success if ok else st.error)(
        "FEASIBLE — serving stack validated on this container. Fine-tuning is safe to start."
        if ok else "NOT YET — drop n_ctx / quant / KV precision and re-probe. Do not fine-tune until this passes ON STREAMLIT CLOUD.")
    st.download_button("Export baseline JSON (metrics + results + config)",
                       json.dumps({"config": st.session_state.get("cfg", {}),
                                   "metrics": st.session_state.metrics,
                                   "results": st.session_state.results}, indent=2),
                       file_name="ats_baseline.json", mime="application/json")
