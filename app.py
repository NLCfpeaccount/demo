# =====================================================================
#  Agentic ATS · Live Demo — chat-style agentic resume parser (FINAL)
#  Every fix from the deployment war baked in. Fixed config, no knobs.
#  1 config · 2 helpers · 3 form+prompt+parser · 4 model server
#  5 pipeline (extract → validate → judge) · 6 chat UI
# =====================================================================
import gc, json, os, re, time, threading
import psutil, streamlit as st
from pydantic import BaseModel, ValidationError, field_validator

# ---------------- 1 · FIXED CONFIG (locked Feasibility Lab verdict) ----------------
REPO, FILE = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
# Fine-tune swap-in later = change ONLY these two lines:
# REPO, FILE = "naresh-cl/ats-student-gguf", "student-q4_k_m.gguf"
N_CTX, N_THREADS, N_BATCH = 4096, 2, 128   # 4096: real 3-page PDFs fit without truncation
MAX_CHARS = 6000                           # ≈2,100 tokens — hard cap on any input
SYS_TOKENS, OUT_TOKENS = 340, 512          # prompt overhead / generation budget
REPEAT_PENALTY = 1.25                      # suppresses degenerate repetition loops
DEFAULT_JD = "Seeking ML engineer: Python, PyTorch, deployment, explainability."

P = psutil.Process(os.getpid())
rss_mb = lambda: P.memory_info().rss / 1e6
def get_secret(k):
    try: return st.secrets[k]
    except Exception: return None
def journal(**kw):
    kw["rss_mb"] = round(rss_mb(), 1)
    print(json.dumps(kw), flush=True)      # survives Cloud OOM kills

# ---------------- 2 · TEXT HELPERS (the scanner) ----------------
def clean_text(raw: str) -> str:
    """Strip footers/blank runs AND collapse repeated lines — repeated blocks
    (tool stacks, project bullets) trigger small-model repetition loops."""
    raw = re.sub(r"page \d+( of \d+)?", "", raw, flags=re.I)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    seen, out = {}, []
    for line in raw.split("\n"):
        k = line.strip().lower()
        if not k:
            out.append(line); continue
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 2:                   # keep first two occurrences, drop the rest
            out.append(line)
    return "\n".join(out)

def pdf_to_text(data: bytes) -> str:
    import pymupdf                         # lazy: text-only sessions never pay the import
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "".join(p.get_text() for p in doc)

# ---------------- 3 · FORM, PROMPT, DEFENSIVE PARSER ----------------
SCHEMA_PROMPT = (
    "You are a precise resume-parsing engine. You copy facts from a resume into a fixed JSON form.\n"
    "Rules:\n"
    "1. Reply with ONE JSON object only. First character '{', last character '}'. No prose, no fences.\n"
    "2. Keys exactly: name, top_skills, years_experience, education, key_projects, summary.\n"
    "3. Types: name=str (the person's full name ONLY — never a city, company, email, or heading); "
    "top_skills=list[str] AT MOST 6 UNIQUE items, never repeat an item; years_experience=number (total years, 0 if unknown); "
    "education=str; key_projects=list[str] AT MOST 3 UNIQUE one-line items; summary=str ONE short sentence.\n"
    "4. Missing field? Use \"\" / [] / 0. Never omit a key, never write null. Be concise: entire JSON under 120 words.\n"
    "5. The resume may be noisy (tables, columns, repeated headers); ignore layout noise, extract each fact once.\n"
    "Example resume: \"Name: X. 2 years. Skills: Python, SQL. B.Tech CSE. Project: built ETL pipeline. Summary: data engineer.\"\n"
    "Example JSON: {\"name\":\"X\",\"top_skills\":[\"Python\",\"SQL\"],\"years_experience\":2,"
    "\"education\":\"B.Tech CSE\",\"key_projects\":[\"built ETL pipeline\"],\"summary\":\"data engineer.\"}")
REMINDER = ("\n---END OF RESUME---\nNow output the JSON object for the resume above. "
            "Start with { and end with }. Nothing else.")

class Profile(BaseModel):
    """The intake form. Lenient on purpose: over-listing and truncation must not fail validation."""
    name: str = ""
    top_skills: list[str] = []
    years_experience: float = 0.0
    education: str = ""
    key_projects: list[str] = []
    summary: str = ""

    @field_validator("top_skills", mode="before")
    @classmethod
    def _cap_skills(cls, v):
        out = []
        for x in (v if isinstance(v, list) else []):
            x = str(x).strip()
            if x and x not in out: out.append(x)
        return out[:6]

    @field_validator("key_projects", mode="before")
    @classmethod
    def _cap_projects(cls, v):
        return [str(x).strip() for x in (v if isinstance(v, list) else []) if str(x).strip()][:3]

    @field_validator("years_experience", mode="before")
    @classmethod
    def _years_num(cls, v):
        try: return float(str(v).replace("years", "").replace("yrs", "").strip() or 0)
        except Exception: return 0.0

    @field_validator("name", "education", "summary", mode="before")
    @classmethod
    def _as_str(cls, v):
        return str(v).strip() if v is not None else ""

def parse_json_defensive(raw: str):
    """Rescue 'almost-JSON': strip fences/prose, cut the first {...} block, and repair
    truncated outputs by closing open brackets/strings. Raises ValidationError if all fail."""
    s = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", raw.strip()))
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i: s = s[i:j + 1]
    elif i != -1:         s = s[i:]          # truncated: no closing brace at all
    last = None
    for closer in ("", "}", "]}", "\"}", "\"]}", "\"}]}"):
        try: return Profile.model_validate_json(s + closer)
        except ValidationError as e: last = e
    raise last

def fit_to_context(llm, text: str):
    """Token-level trim so prompt+output can NEVER exceed n_ctx (no ValueError crash)."""
    n = llm.n_ctx(); out = OUT_TOKENS
    toks = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    budget = n - SYS_TOKENS - out - 16
    if budget < 64:
        out = max(96, n - SYS_TOKENS - 96); budget = max(32, n - SYS_TOKENS - out - 16)
    if len(toks) > budget:
        toks = toks[:budget]; text = llm.detokenize(toks).decode("utf-8", "ignore")
    return text, out

# ---------------- 4 · MODEL SERVER (one shared instance, self-healing) ----------------
@st.cache_resource(show_spinner="Waking the clerk — first visitor pays ~30 s…")
def get_llm():
    import inspect
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    path = hf_hub_download(REPO, FILE, token=get_secret("HF_TOKEN"))
    journal(event="download_ok", file_mb=round(os.path.getsize(path) / 1e6))
    params = inspect.signature(Llama.__init__).parameters
    variants = []
    if "cache_type_k" in params: variants.append({"cache_type_k": "q8_0"})  # K-cache quant, safe without flash_attn
    if "type_k" in params:       variants.append({"type_k": 8})             # older naming, same effect
    variants.append({})                                                      # f16 KV last resort
    last = None
    for kw in variants:
        try:
            llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS,
                        n_batch=N_BATCH, verbose=False, **kw)
            journal(event="load_ok", kv=str(kw or "f16"))
            return llm
        except Exception as e:
            last = e; journal(event="load_retry", kv=str(kw or "f16"), err=str(e)[:200])
    raise last

if "_lock" not in st.session_state: st.session_state._lock = threading.Lock()
LOCK = st.session_state._lock

# ---------------- 5 · PIPELINE (clerk → checker → master) ----------------
def extract_resume(llm, text: str):
    """Clerk fills the form; checker validates; ONE corrective retry on failure.
    Returns (Profile|None, attempts, secs, raw_text_on_failure)."""
    text, out = fit_to_context(llm, text)
    msgs = [{"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user",   "content": text + REMINDER}]
    last_raw = ""
    for att in (1, 2):
        t0 = time.time()
        r = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=out,
                                       repeat_penalty=REPEAT_PENALTY)
        secs = time.time() - t0
        raw = r["choices"][0]["message"]["content"]; last_raw = raw
        try: return parse_json_defensive(raw), att, secs, None
        except ValidationError as e:
            msgs += [{"role": "assistant", "content": raw},
                     {"role": "user", "content":
                      f"Errors: {e.errors()[:2]}. Your previous reply repeated list items and/or was cut off. "
                      "Fix ONLY the errors: at most 6 UNIQUE skills, 3 UNIQUE projects, include ALL six keys, "
                      "and reply with the complete JSON object only, starting with { and ending with }."}]
    return None, 2, secs, last_raw

def judge_candidate(llm, prof: Profile, jd: str):
    """Master sees ONLY the ~300-token form + JD — never the raw resume (the 90% cost cut)."""
    r = llm.create_chat_completion(temperature=0.0, max_tokens=80, messages=[
        {"role": "system", "content": "Score 0-100 vs JD. Reply: SCORE: <int> REASON: <sentence>"},
        {"role": "user",   "content": jd + "\n" + prof.model_dump_json()}])
    txt = r["choices"][0]["message"]["content"]
    m = re.search(r"(\d{1,3})", txt)
    return (int(m.group(1)) if m else 0), txt.split("REASON:")[-1].strip()[:140]

# ---------------- 6 · CHAT UI (the front desk) ----------------
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
    st.caption("Fixed build: Qwen2.5-0.5B · q4_k_m · ctx4096 · thr2 · q8 K-cache · repeat-penalty 1.25. "
               "Fine-tuned student slots in later via two constants.")

st.session_state.setdefault("chat", [])
for msg in st.session_state.chat:                        # bounded history = bounded RAM
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🛰 agent trace"):
                for line in msg["trace"]: st.text(line)
        if msg.get("form"): st.json(msg["form"])

prompt = st.chat_input("Paste resume text here, or upload a PDF and say 'parse'…")

def run_agent(raw_text: str, source: str):
    """The agentic loop with a visible trace. Never stores PDF bytes or unbounded state."""
    trace = [f"source: {source}", f"raw chars: {len(raw_text)}"]
    text = clean_text(raw_text)[:MAX_CHARS]
    trace.append(f"cleaned+capped: {len(text)} chars")
    if not LOCK.acquire(blocking=False):
        return {"role": "assistant", "ok": False,
                "content": "🧑‍💼 The clerk is serving another visitor right now — resend in a few seconds."}
    try:
        llm = get_llm()
        trace.append("model: awake")
        prof, att, secs, fail_raw = extract_resume(llm, text)
        trace.append(f"extract: attempts={att}, {secs:.1f}s")
        if prof is None:
            trace.append("validate: FAILED twice")
            if fail_raw: trace.append(f"RAW MODEL OUTPUT:\n{fail_raw[:1000]}")
            journal(event="extract_fail", att=att, chars=len(text))
            return {"role": "assistant", "ok": False, "trace": trace,
                    "content": "😬 The form came back invalid twice. Expand the 'agent trace' below — "
                               "the RAW MODEL OUTPUT line shows exactly what the clerk wrote."}
        trace.append("validate: schema OK ✅")
        score, reason = judge_candidate(llm, prof, jd)
        trace.append(f"judge: {score}/100")
        journal(event="parse_ok", chars=len(text), att=att, score=score, secs=round(secs, 1))
        head = (f"### {prof.name or 'Candidate'} — **{score}/100**\n*{reason}*\n\n"
                f"**Experience:** {prof.years_experience} yrs · **Education:** {prof.education or '—'}\n"
                f"**Top skills:** {', '.join(prof.top_skills) or '—'}\n"
                f"**Projects extracted:** {len(prof.key_projects)}")
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
    file_key = f"{up.name}:{up.size}" if up is not None else None
    if file_key and st.session_state.get("parsed_file_name") != file_key:
        data = up.read()
        raw = pdf_to_text(data) if up.name.lower().endswith(".pdf") else data.decode("utf-8", "ignore")
        del data                                     # PDF bytes die here; only capped text lives on
        reply = run_agent(raw, up.name)
        if reply.get("ok"): st.session_state.parsed_file_name = file_key  # failure keeps it retryable
    elif len(prompt) > 200:
        reply = run_agent(prompt, "pasted text")
    else:
        reply = {"role": "assistant",
                 "content": "I parse resumes! **Upload a PDF** in the sidebar and send any message, "
                            "or **paste the resume text** (200+ chars) right here."}
    st.session_state.chat.append(reply)
    st.session_state.chat = st.session_state.chat[-12:]
    st.rerun()
