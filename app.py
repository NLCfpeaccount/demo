# Agentic ATS · Live Demo v2 — chat-style agentic resume parser (Cloud-hardened)
import gc, json, os, re, time, threading
import psutil, streamlit as st
from pydantic import BaseModel, ValidationError

# ---------------- FIXED CONFIG (locked lab verdict — no UI knobs) ----------------
REPO, FILE = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
# Fine-tune swap-in later = change ONLY these two lines:
# REPO, FILE = "naresh-cl/ats-student-gguf", "student-q4_k_m.gguf"
N_CTX, N_THREADS, N_BATCH = 2048, 2, 128
MAX_CHARS, SYS_TOKENS, OUT_TOKENS = 6000, 340, 384
DEFAULT_JD = "Seeking ML engineer: Python, PyTorch, deployment, explainability."

P = psutil.Process(os.getpid())
rss_mb = lambda: P.memory_info().rss / 1e6
def get_secret(k):
    try: return st.secrets[k]
    except Exception: return None
def journal(**kw):
    kw["rss_mb"] = round(rss_mb(), 1)
    print(json.dumps(kw), flush=True)              # survives Cloud kills

# ---------------- A · FORM, PROMPT, PARSER ----------------
SCHEMA_PROMPT = (
    "You are a precise resume-parsing engine. You copy facts from a resume into a fixed JSON form.\n"
    "Rules:\n1. Reply with ONE JSON object only. First character '{', last '}'. No prose, no fences.\n"
    "2. Keys exactly: name, top_skills, years_experience, education, key_projects, summary.\n"
    "3. Types: name=str; top_skills=list[str] max 6; years_experience=number (0 if unknown); "
    "education=str; key_projects=list[str] max 3; summary=str one sentence.\n"
    "4. Missing field? Use \"\" / [] / 0. Never omit a key, never write null.\n"
    "5. The resume may repeat sections; extract each fact once.\n"
    "Example resume: \"Name: X. 2 years. Skills: Python, SQL. B.Tech CSE. Project: built ETL pipeline. Summary: data engineer.\"\n"
    "Example JSON: {\"name\":\"X\",\"top_skills\":[\"Python\",\"SQL\"],\"years_experience\":2,"
    "\"education\":\"B.Tech CSE\",\"key_projects\":[\"built ETL pipeline\"],\"summary\":\"data engineer.\"}")
REMINDER = ("\n---END OF RESUME---\nNow output the JSON object for the resume above. "
            "Start with { and end with }. Nothing else.")

class Profile(BaseModel):
    name: str; top_skills: list[str]; years_experience: float
    education: str; key_projects: list[str]; summary: str

def clean_text(raw: str) -> str:
    raw = re.sub(r"page \d+( of \d+)?", "", raw, flags=re.I)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    return re.sub(r"\n{3,}", "\n\n", raw).strip()

def pdf_to_text(data: bytes) -> str:
    import pymupdf                                  # lazy; text-only sessions never pay it
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "".join(p.get_text() for p in doc)

def parse_json_defensive(raw: str):
    s = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", raw.strip()))
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i: s = s[i:j + 1]
    return Profile.model_validate_json(s)

# ---------------- D · MODEL SERVER (self-healing load) ----------------
@st.cache_resource(show_spinner="Waking the clerk — first visitor pays ~30 s…")
def get_llm():
    import inspect, shutil
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    journal(event="pre_download", disk_free_mb=round(shutil.disk_usage("/").free / 1e6))
    try:
        path = hf_hub_download(REPO, FILE, token=get_secret("HF_TOKEN"))
    except Exception as e:
        journal(event="download_retry", err=str(e)[:200])
        os.environ["HF_HUB_DISABLE_XET"] = "1"      # xet flaky on some Cloud egress paths
        path = hf_hub_download(REPO, FILE, token=get_secret("HF_TOKEN"))
    journal(event="post_download", disk_free_mb=round(shutil.disk_usage("/").free / 1e6),
            file_mb=round(os.path.getsize(path) / 1e6))
    params = inspect.signature(Llama.__init__).parameters
    variants = []
    if "cache_type_k" in params: variants.append({"cache_type_k": "q8_0"})
    if "type_k" in params:       variants.append({"type_k": 8})
    variants.append({})                              # f16 KV last-resort
    last = None
    for kw in variants:
        try:
            llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS,
                        n_batch=N_BATCH, verbose=False, **kw)
            journal(event="load_ok", kv=str(kw or "f16"))
            return llm
        except Exception as e:
            last = e
            journal(event="load_retry", kv=str(kw or "f16"), err=str(e)[:200])
    raise last

if "_lock" not in st.session_state: st.session_state._lock = threading.Lock()
LOCK = st.session_state._lock

def fit_to_context(llm, text: str):
    n = llm.n_ctx(); out = OUT_TOKENS
    toks = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    budget = n - SYS_TOKENS - out - 16
    if budget < 64:
        out = max(96, n - SYS_TOKENS - 96); budget = max(32, n - SYS_TOKENS - out - 16)
    if len(toks) > budget:
        toks = toks[:budget]; text = llm.detokenize(toks).decode("utf-8", "ignore")
    return text, out

def extract_resume(llm, text: str):
    text, out = fit_to_context(llm, text)
    msgs = [{"role": "system", "content": SCHEMA_PROMPT}, {"role": "user", "content": text + REMINDER}]
    for att in (1, 2):
        t0 = time.time()
        r = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=out)
        secs = time.time() - t0
        raw = r["choices"][0]["message"]["content"]
        try: return parse_json_defensive(raw), att, secs
        except ValidationError as e:
            msgs += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": f"Errors: {e.errors()[:2]}. Fix ONLY these. "
                                                  "Keep other values. Reply JSON object only."}]
    return None, 2, secs

def judge_candidate(llm, prof: Profile, jd: str):
    r = llm.create_chat_completion(temperature=0.0, max_tokens=80, messages=[
        {"role": "system", "content": "Score 0-100 vs JD. Reply: SCORE: <int> REASON: <sentence>"},
        {"role": "user", "content": jd + "\n" + prof.model_dump_json()}])
    txt = r["choices"][0]["message"]["content"]
    m = re.search(r"(\d{1,3})", txt)
    return (int(m.group(1)) if m else 0), txt.split("REASON:")[-1].strip()[:140]

# ---------------- E · CHAT UI ----------------
st.set_page_config(page_title="Agentic ATS — AI Resume Parser", layout="centered")
st.title("🎓 Agentic ATS — AI Resume Parser")
st.caption("Upload a PDF (sidebar) or paste resume text below. The agent parses → validates → "
           "self-corrects → scores against the job description.")

with st.sidebar:
    st.header("📄 Input")
    up = st.file_uploader("Resume (PDF / TXT / MD)", type=["pdf", "txt", "md"])
    jd = st.text_area("Job description", DEFAULT_JD, height=140)
    if st.button("🗑 Clear conversation"): st.session_state.chat = []
    st.divider()
    st.caption("Fixed build: Qwen2.5-0.5B · q4_k_m · ctx2048 · thr2 · q8 K-cache. "
               "Fine-tuned student slots in later via two constants.")

st.session_state.setdefault("chat", [])
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🛰 agent trace"):
                for line in msg["trace"]: st.text(line)
        if msg.get("form"): st.json(msg["form"])

prompt = st.chat_input("Paste resume text here, or upload a PDF and say 'parse'…")

def run_agent(raw_text: str, source: str):
    trace = [f"source: {source}", f"raw chars: {len(raw_text)}"]
    text = clean_text(raw_text)[:MAX_CHARS]
    trace.append(f"cleaned+capped: {len(text)} chars")
    if not LOCK.acquire(blocking=False):
        return {"role": "assistant", "ok": False,
                "content": "🧑‍💼 The clerk is serving another visitor right now — resend in a few seconds."}
    try:
        llm = get_llm()
        trace.append("model: awake")
        prof, att, secs = extract_resume(llm, text)
        trace.append(f"extract: attempts={att}, {secs:.1f}s")
        if prof is None:
            trace.append("validate: FAILED twice")
            return {"role": "assistant", "ok": False, "trace": trace,
                    "content": "😬 The form came back invalid twice. Try a cleaner scan or paste the text instead."}
        trace.append("validate: schema OK ✅")
        score, reason = judge_candidate(llm, prof, jd)
        trace.append(f"judge: {score}/100")
        journal(event="parse", chars=len(text), att=att, score=score)
        head = (f"### {prof.name or 'Candidate'} — **{score}/100**\n*{reason}*\n\n"
                f"**Experience:** {prof.years_experience} yrs · **Education:** {prof.education}\n"
                f"**Top skills:** {', '.join(prof.top_skills) or '—'}")
        return {"role": "assistant", "ok": True, "trace": trace,
                "form": prof.model_dump(), "content": head}
    except Exception as e:
        journal(event="error", kind=type(e).__name__, msg=str(e)[:300])
        return {"role": "assistant", "ok": False, "trace": trace,
                "content": f"⚠️ Agent hit a snag: `{type(e).__name__}: {str(e)[:160]}` — nothing stored; resend to retry."}
    finally:
        LOCK.release(); gc.collect()

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    if up is not None and not st.session_state.pop("parsed_current_file", False):
        data = up.read()
        raw = pdf_to_text(data) if up.name.lower().endswith(".pdf") else data.decode("utf-8", "ignore")
        del data                                    # PDF bytes die here; only text lives on
        reply = run_agent(raw, up.name)
        if reply.get("ok"): st.session_state.parsed_current_file = True   # retry keeps the file on failure
    elif len(prompt) > 200:
        reply = run_agent(prompt, "pasted text")
    else:
        reply = {"role": "assistant",
                 "content": "I parse resumes! **Upload a PDF** in the sidebar and send any message, "
                            "or **paste the resume text** (200+ chars) right here."}
    st.session_state.chat.append(reply)
    st.session_state.chat = st.session_state.chat[-12:]
    st.rerun()
