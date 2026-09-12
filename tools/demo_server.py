"""Demonstration interface — Anteproyecto Objetivo 4.

A zero-dependency local web demo: type a question, link entities to Wikidata
QIDs via a search dropdown (or load a DEV-200 preset), run all four pipeline
configurations, and see each answer beside the retrieved context supplied to
the model. Stdlib only — the frozen venv is never touched.

Run from the repo root (so `.env` resolves the same way every tool does):

    venv\\Scripts\\python tools\\demo_server.py [--port 8765] [--model ...]

⚠️ Do NOT run the demo concurrently with an evaluation run against the same
NVIDIA endpoint: the rate limiter is process-local, so two processes would
jointly exceed the provider limit.

Design contract (see the reviewed plan; all invariants have tests):
- No edits under src/. Every behavioural knob is a runtime assignment of a
  module global that the frozen code dereferences at call time.
- The git-tracked frozen caches are never written. All cache reads and writes
  go through a demo overlay at data/cache/demo/ (gitignored); before each
  retrieval run the frozen entry for every requested QID is copied over the
  demo entry — frozen always wins, a stale demo entry never shadows it.
- STRICT_RETRIEVAL stays True: a failed retrieval is an error row, never a
  silently partial pool.
- Import-safe: importing this module performs no server start, no cache
  mutation, no preset loading, no model warming. Side effects live in main().
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.llm_config as llm_config          # noqa: E402  (fires load_dotenv)
import src.pipelines.base_llm as base_llm    # noqa: E402
import src.pipelines.graph_rag as graph_rag  # noqa: E402
import src.pipelines.graph_rag_rerank as graph_rag_rerank  # noqa: E402
import src.pipelines.rag as rag              # noqa: E402
import src.prompts as prompts                # noqa: E402
import src.retrieval.embedding_retriever as embedding_retriever  # noqa: E402
import src.retrieval.wikidata as wikidata    # noqa: E402
import src.retrieval.wikidata_pool as wikidata_pool  # noqa: E402
import src.retrieval.wikipedia as wikipedia  # noqa: E402
import src.token_counter as token_counter    # noqa: E402
from src.eval import metrics                 # noqa: E402
from src.eval.parse_answers import load_answers      # noqa: E402
from src.eval.parse_questions import (       # noqa: E402
    ENTITY,
    EntityPackage,
    Question,
    load_questions,
)
from src.retrieval.entity_linking import _search_entity  # noqa: E402
from src.retrieval.wikidata import IncompleteRetrievalError  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
# The model NAME does not identify the provider: the same string is served by
# other OpenAI-compatible hosts, so the endpoint is part of the reference
# identity. Unset means the OpenAI default endpoint, which is not it either.
REFERENCE_BASE_URL = "https://integrate.api.nvidia.com/v1"

QID_RE = re.compile(r"^Q[1-9]\d*$")

CONFIGS = ("base_llm_abstain", "rag", "graph_rag", "rerank")
RETRIEVAL_CONFIGS = ("rag", "graph_rag", "rerank")

MAX_ENTITIES = 20
MAX_LABEL_CHARS = 300
MAX_QUESTION_CHARS = 2000
MAX_BODY_BYTES = 64 * 1024

DEV_SAMPLE = ROOT / "data" / "questions" / "mintaka_sample_dev_200.json"
DEMO_CACHE_ROOT = ROOT / "data" / "cache" / "demo"
# Append-only JSONL, one record per /api/run: the question, the QIDs it was
# given, and that configuration's stats. A demonstration leaves a record.
CALL_LOG_PATH = ROOT / "data" / "analysis" / "demo_calls.jsonl"

# Retrieval-incomplete gets an operator hint; every other exception renders
# its own type + message without the gloss (they are not endpoint weather).
INCOMPLETE_HINT = ("retrieval incomplete — endpoint likely struggling, "
                   "try again later")


class CacheSeedError(RuntimeError):
    """A frozen cache entry could not be copied into the demo overlay.

    Deliberately fatal for the affected configuration: silently falling
    through to a live fetch would serve data the frozen corpus disagrees
    with, without any signal.
    """


# ---------------------------------------------------------------------------
# Mutable server state — populated by main(); tests use their own instances
# ---------------------------------------------------------------------------

STATE: dict = {
    "frozen_wd": None,          # Path — frozen wikidata statement cache
    "frozen_wp": None,          # Path — frozen wikipedia article cache
    "demo_wd": None,
    "demo_wp": None,
    "presets": {},              # question id -> Question
    "golds": {},                # question id -> GoldAnswer
    "preset_order": [],         # question ids in file order
    "embedder_state": "loading",  # loading | ready | error
    "embedder_error": None,
    "non_reference_axes": [],
    "model_is_reference": True,
    "run_lock": threading.Lock(),
    # Path, or None to disable. Absent/None means log_call is a no-op, which is
    # what keeps the unit fixtures from ever touching a real file.
    "call_log": None,
}


# ---------------------------------------------------------------------------
# Pure helpers (import-safe, unit-tested)
# ---------------------------------------------------------------------------

def validate_run_request(body: object, presets: dict) -> tuple[dict | None, str | None]:
    """Validate a /api/run body. Returns (parsed, None) or (None, error).

    QIDs are validated BEFORE anything constructs a cache path from them.
    """
    if not isinstance(body, dict):
        return None, "body must be a JSON object"
    config = body.get("config")
    if config not in CONFIGS:
        return None, f"config must be one of {list(CONFIGS)}"

    preset_id = body.get("preset_id")
    if preset_id is not None:
        if not isinstance(preset_id, str) or preset_id not in presets:
            return None, "unknown preset_id"
        return {"config": config, "preset_id": preset_id}, None

    question = body.get("question")
    if not isinstance(question, str) or not (1 <= len(question.strip()) <= MAX_QUESTION_CHARS):
        return None, f"question must be 1–{MAX_QUESTION_CHARS} characters"

    entities = body.get("entities", [])
    if not isinstance(entities, list) or len(entities) > MAX_ENTITIES:
        return None, f"entities must be a list of at most {MAX_ENTITIES}"
    cleaned = []
    for ent in entities:
        if not isinstance(ent, dict):
            return None, "each entity must be an object"
        qid = ent.get("qid")
        label = ent.get("label")
        phrase = ent.get("phrase") or ""
        if not isinstance(qid, str) or not QID_RE.match(qid):
            return None, f"invalid QID {qid!r} (expected Q[1-9][0-9]*)"
        if not isinstance(label, str) or not (0 < len(label) <= MAX_LABEL_CHARS):
            return None, f"invalid label for {qid}"
        if not isinstance(phrase, str) or len(phrase) > MAX_LABEL_CHARS:
            return None, f"invalid phrase for {qid}"
        cleaned.append({"qid": qid, "label": label, "phrase": phrase.strip()})

    return {"config": config, "preset_id": None,
            "question": question.strip(), "entities": cleaned}, None


def build_manual_question(question: str, entities: list[dict]) -> Question:
    """Build a Question for manual linked-entity mode.

    NOT evaluation-exact: presets carry Mintaka's annotations (mentions,
    normalised literal packages); here the user's search phrase stands in for
    the mention. `first_mention` drives ranking; `mentions` is a DISPLAY list
    holding only forms that DIFFER from the label (the entity-block legend
    renders "mentioned as …" from it), so a phrase equal to the label
    contributes an empty tuple — same rule as the parser.
    """
    packages = []
    for ent in entities:
        phrase = ent["phrase"] or ent["label"]
        differs = phrase.strip().casefold() != ent["label"].strip().casefold()
        packages.append(EntityPackage(
            name=ent["qid"],
            entity_type=ENTITY,
            label=ent["label"],
            mentions=(phrase,) if differs else (),
            first_mention=phrase,
        ))
    return Question(id="demo-" + uuid.uuid4().hex[:8],
                    text=question, entities=tuple(packages))


def seed_cache_for(config: str, qids: list[str],
                   frozen_wd: Path, frozen_wp: Path,
                   demo_wd: Path, demo_wp: Path) -> None:
    """Copy frozen cache entries over the demo overlay — frozen always wins.

    Seeded per configuration: C1 touches no cache, C2 only Wikipedia entries,
    C3/C4 only Wikidata entries. An existing demo copy is atomically
    OVERWRITTEN whenever frozen has the QID (a stale demo entry must never
    shadow a tracked frozen entry); a demo-only entry is used only when
    frozen lacks the QID. A copy failure raises CacheSeedError — never a
    silent fall-through to a live fetch.
    """
    if config == "base_llm_abstain":
        return
    pairs = {"rag": [(frozen_wp, demo_wp)],
             "graph_rag": [(frozen_wd, demo_wd)],
             "rerank": [(frozen_wd, demo_wd)]}[config]
    for frozen_dir, demo_dir in pairs:
        for qid in qids:
            if not QID_RE.match(qid):  # defense in depth; validated upstream
                raise CacheSeedError(f"refusing cache path for invalid QID {qid!r}")
            src_path = frozen_dir / f"{qid}.json"
            if not src_path.exists():
                continue
            dst_path = demo_dir / f"{qid}.json"
            try:
                _atomic_copy(src_path, dst_path)
            except OSError as exc:
                raise CacheSeedError(
                    f"could not seed {qid} from the frozen cache: {exc}") from exc


def _atomic_copy(src_path: Path, dst_path: Path) -> None:
    """Copy src over dst atomically (write temp in place, then os.replace)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(dst_path.suffix + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    tmp.write_bytes(src_path.read_bytes())
    os.replace(tmp, dst_path)


def evidence_from_capture(config: str, capture: dict) -> dict:
    """Map a pipeline's capture dict to the UI evidence payload.

    All headings downstream say "retrieved context supplied to the model" —
    supplied context and claims actually grounded in it are different things
    (the thesis's own faithfulness results), so never "evidence used".
    """
    if config == "base_llm_abstain":
        return {}
    context = capture.get("context", "") or ""
    if config == "rag":
        # Chunks ARE double-newline-joined; `articles` lists CANDIDATE source
        # articles of the retrieval pool, not per-chunk provenance.
        return {"chunks": [c for c in context.split("\n\n") if c.strip()],
                "articles": capture.get("articles", [])}
    if config == "graph_rag":
        # Statement bullets are single-newline separated — split("\n\n")
        # would yield one blob. Render verbatim lines.
        return {"statements": [ln for ln in context.splitlines() if ln.strip()]}
    return {"top_facts": capture.get("top_facts", []),
            "condensed": context,
            "condense_finish_reason": capture.get("condense_finish_reason"),
            "condense_words": capture.get("condense_words"),
            "condense_fallback": bool(capture.get("condense_fallback", False))}


def error_payload(exc: BaseException) -> dict:
    """Exception → error row payload. Only IncompleteRetrievalError gets the
    endpoint gloss; anything else may be an LLM, embedding, permission or
    programming failure and must not be blamed on the endpoint."""
    payload = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, IncompleteRetrievalError):
        payload["hint"] = INCOMPLETE_HINT
    elif isinstance(exc, CacheSeedError):
        payload["hint"] = "frozen-cache seeding failed for this configuration"
    return payload


def badges(answer: str | None) -> dict:
    """Abstention flags. `abstained` uses the exact-sentinel predicate that
    defines the thesis's ABSTENTION bucket; the refusal-shaped badge must
    exclude compliant abstentions (the sentinel itself also opens with
    refusal language)."""
    abstained = metrics.is_compliant_abstention(answer)
    return {"abstained": abstained,
            "refusal_shaped": (not abstained) and metrics.opens_with_refusal(answer)}


def token_delta(before: dict, after: dict) -> dict:
    return {"calls": after["calls"] - before["calls"],
            "prompt": after["prompt_tokens"] - before["prompt_tokens"],
            "completion": after["completion_tokens"] - before["completion_tokens"]}


def pin_reference_configuration(model: str) -> list[str]:
    """Set/assert every frozen configuration axis on the live modules.

    `.env` can move LLM_MAX_TOKENS, VALUE_DESC_MODE and NOISE_FILTER_MODE,
    and C4 latches ANSWER_MAX_TOKENS at import — so pinning only the model
    would silently ship a non-reference configuration. Returns the axes that
    remain non-reference (empty = full reference compatibility).
    """
    non_reference: list[str] = []

    llm_config.MODEL = model
    if model != REFERENCE_MODEL:
        non_reference.append(f"model={model}")

    # Read from the environment, not from an argument: get_client() resolves the
    # endpoint from LLM_BASE_URL, so this is the value that will actually serve.
    base_url = (os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
    if base_url.lower() != REFERENCE_BASE_URL.rstrip("/").lower():
        non_reference.append(f"base_url={base_url or '(unset)'}")

    llm_config.MAX_TOKENS = 128
    graph_rag_rerank.ANSWER_MAX_TOKENS = 128     # latched at import from MAX_TOKENS
    graph_rag_rerank.CONDENSE_MAX_TOKENS = 384
    rag.TOP_K = 30
    graph_rag.TOP_K = 30
    graph_rag_rerank.EMBED_TOP_K = 30
    embedding_retriever.ALLOCATION = embedding_retriever.ALLOCATION_GLOBAL
    prompts.ENTITY_BLOCK_MODE = "all"
    wikidata_pool.VALUE_DESC_MODE = "off"
    wikidata.NOISE_FILTER_MODE = "property_type"

    # Assert-only axes (not env-driven; a mismatch means the code moved).
    if tuple(wikidata.RANK_POLICY) != ("preferred", "normal"):
        non_reference.append(f"rank_policy={wikidata.RANK_POLICY!r}")
    if prompts.PROMPT_VARIANT != "f":
        non_reference.append(f"prompt_variant={prompts.PROMPT_VARIANT!r}")

    return non_reference


# ---------------------------------------------------------------------------
# Run execution (uses STATE; serialized by the run lock in the handler)
# ---------------------------------------------------------------------------

def resolve_question(parsed: dict, state: dict) -> Question:
    if parsed.get("preset_id"):
        return state["presets"][parsed["preset_id"]]
    return build_manual_question(parsed["question"], parsed["entities"])


def dispatch(config: str, q: Question, capture: dict) -> str:
    """Mirror run_eval.run_config exactly (src/eval/run_eval.py:230-251)."""
    if config == "base_llm_abstain":
        return base_llm.answer(q.text)
    mod = {"rag": rag, "graph_rag": graph_rag, "rerank": graph_rag_rerank}[config]
    entity_block = q.prompt_block(prompts.ENTITY_BLOCK_MODE)
    return mod.answer(q.text, q.entity_mentions, qids=q.qids,
                      capture=capture, entity_block=entity_block)


def log_call(state: dict, parsed: dict, q: Question, row: dict) -> None:
    """Append one JSONL record for this call: question, QIDs, and stats.

    One record per /api/run, i.e. per CONFIGURATION — the page issues four
    requests for one question, so four lines share a question and differ in
    `config` and stats. Joining them is the reader's job; splitting them here
    keeps each line self-contained.

    ⚠️ Self-describing on purpose. `model` / `model_is_reference` /
    `non_reference_axes` are written into every line, so a record produced
    while `.env` pointed somewhere else can never be mistaken for a
    reference-configuration demonstration after the fact.

    Off unless `state["call_log"]` is set — the unit fixtures build STATE
    without the key, so they never touch a real file. A logging failure must
    not fail a run that already succeeded: it warns and returns.
    """
    path = state.get("call_log")
    if not path:
        return
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "preset" if parsed.get("preset_id") else "manual",
        "preset_id": parsed.get("preset_id"),
        "question": q.text,
        "qids": q.qids,
        "entity_names": q.entity_names,
        # Manual mode only: what the user actually typed and selected. The
        # search phrase stands in for Mintaka's mention and is NOT
        # evaluation-exact, so it is recorded separately from the resolved
        # labels rather than merged into them.
        "entities_requested": parsed.get("entities"),
        "config": row["config"],
        "model": llm_config.MODEL,
        "model_is_reference": state.get("model_is_reference"),
        "non_reference_axes": state.get("non_reference_axes"),
        "latency_s": row["latency_s"],
        "tokens": row["tokens"],
        "context_words": row["context_words"],
        "pool_size": row["pool_size"],
        "abstained": row["abstained"],
        "refusal_shaped": row["refusal_shaped"],
        "finish_reason": row["finish_reason"],
        "answer": row["answer"],
        "error": row["error"],
    }
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        print(f"  ⚠️ call-log write failed ({exc}) — the run itself was fine")


def execute_run(parsed: dict, state: dict) -> dict:
    """Run ONE configuration and assemble its row payload.

    Telemetry guards: the token delta is computed in `finally` so a mid-call
    exception still reports the tokens it burned; `last_finish_reason()` is
    read only after a SUCCESSFUL call (after a failure it may retain the
    preceding call's value); latency is observed wall time for this run —
    C3 warms C4's cache reads, so rows are not a performance experiment.
    """
    config = parsed["config"]
    q = resolve_question(parsed, state)

    capture: dict = {}
    answer_text: str | None = None
    error: dict | None = None
    finish_reason: str | None = None

    u0 = token_counter.get()
    t0 = time.perf_counter()
    try:
        if config in RETRIEVAL_CONFIGS:
            seed_cache_for(config, q.qids,
                           state["frozen_wd"], state["frozen_wp"],
                           state["demo_wd"], state["demo_wp"])
        answer_text = dispatch(config, q, capture)
        finish_reason = llm_config.last_finish_reason()
    except Exception as exc:  # noqa: BLE001 — every failure becomes an error row
        error = error_payload(exc)
    finally:
        latency = time.perf_counter() - t0
        tokens = token_delta(u0, token_counter.get())

    flags = badges(answer_text)
    is_retrieval = config in RETRIEVAL_CONFIGS
    context_words = (len((capture.get("context") or "").split())
                     if is_retrieval else None)

    row = {
        "config": config,
        "answer": answer_text,
        "abstained": flags["abstained"],
        "refusal_shaped": flags["refusal_shaped"],
        "finish_reason": finish_reason,
        "latency_s": round(latency, 2),
        "tokens": tokens,
        "usage_unavailable": (error is None and tokens["calls"] == 0),
        "pool_size": capture.get("pool_size"),
        "context_words": context_words,
        "evidence": evidence_from_capture(config, capture),
        "error": error,
    }
    # After the row is complete, so an error row is logged too — a failed
    # demonstration is exactly the one worth having a record of.
    log_call(state, parsed, q, row)
    return row


# ---------------------------------------------------------------------------
# Startup pieces (called from main() only)
# ---------------------------------------------------------------------------

def install_cache_overlay(state: dict) -> None:
    """Redirect both cache dirs to the demo overlay; refresh shared metadata.

    Must run before anything touches wikidata.item_properties() — its
    in-memory memo latches on first use. `_item_properties.json` is refreshed
    from frozen unconditionally (atomic overwrite, never only-if-absent).
    """
    state["frozen_wd"] = wikidata.STATEMENT_CACHE_DIR
    state["frozen_wp"] = wikipedia.CACHE_DIR
    state["demo_wd"] = DEMO_CACHE_ROOT / "wikidata_statements"
    state["demo_wp"] = DEMO_CACHE_ROOT / "wikipedia"
    state["demo_wd"].mkdir(parents=True, exist_ok=True)
    state["demo_wp"].mkdir(parents=True, exist_ok=True)
    wikidata.STATEMENT_CACHE_DIR = state["demo_wd"]
    wikipedia.CACHE_DIR = state["demo_wp"]

    props = state["frozen_wd"] / "_item_properties.json"
    if props.exists():
        _atomic_copy(props, state["demo_wd"] / "_item_properties.json")


def load_presets(state: dict) -> None:
    if not DEV_SAMPLE.exists():
        print(f"⚠️ presets disabled: {DEV_SAMPLE} not found (manual mode still works)")
        return
    questions = load_questions(DEV_SAMPLE)
    golds = load_answers(DEV_SAMPLE)
    state["presets"] = {q.id: q for q in questions}
    state["golds"] = golds
    state["preset_order"] = [q.id for q in questions]


def preset_payload(state: dict) -> list[dict]:
    out = []
    for qid_ in state["preset_order"]:
        q = state["presets"][qid_]
        gold = state["golds"].get(qid_)
        entities = [{"qid": p.name, "label": p.label, "mention": p.surface}
                    for p in q.entities
                    if p.entity_type == ENTITY and p.linked]
        out.append({
            "id": q.id, "question": q.text,
            "category": q.category, "complexity": q.complexity,
            "entities": entities,
            "gold": ({"mention": gold.mention,
                      "forms": list(gold.forms),
                      "answer_type": gold.answer_type} if gold else None),
        })
    return out


def warm_embedder(state: dict) -> None:
    try:
        embedding_retriever._get_model()
        state["embedder_state"] = "ready"
    except Exception as exc:  # noqa: BLE001
        state["embedder_state"] = "error"
        state["embedder_error"] = f"{type(exc).__name__}: {exc}"


def status_payload(state: dict) -> dict:
    return {
        "model": llm_config.MODEL,
        "model_is_reference": state["model_is_reference"],
        "non_reference_axes": state["non_reference_axes"],
        "rpm": llm_config.default_rpm(),
        "strict": bool(wikidata.STRICT_RETRIEVAL and wikipedia.STRICT_RETRIEVAL),
        "embedder_state": state["embedder_state"],
        "embedder_error": state["embedder_error"],
        "busy": state["run_lock"].locked(),
        "presets_available": bool(state["preset_order"]),
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class DemoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TFMGraphRAGDemo/1.0"

    # -- plumbing -----------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # quiet request log
        pass

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = (Path(__file__).parent / "demo.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif parsed.path == "/api/status":
                self._send_json(200, status_payload(STATE))
            elif parsed.path == "/api/presets":
                self._send_json(200, {"presets": preset_payload(STATE)})
            elif parsed.path == "/api/search":
                query = parse_qs(parsed.query).get("q", [""])[0].strip()
                if not (0 < len(query) <= MAX_LABEL_CHARS):
                    self._send_json(400, {"error": "q must be 1–300 characters"})
                    return
                try:
                    self._send_json(200, {"candidates": _search_entity(query)})
                except Exception as exc:  # noqa: BLE001 — network weather
                    self._send_json(502, {"error": f"search unavailable: {exc}"})
            else:
                self._send_json(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- POST ---------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/run":
                self._send_json(404, {"error": "not found"})
                return
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                self._send_json(400, {"error": "Content-Type must be application/json"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not (0 < length <= MAX_BODY_BYTES):
                self._send_json(400, {"error": f"body must be 1–{MAX_BODY_BYTES} bytes"})
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "body is not valid JSON"})
                return

            parsed, err = validate_run_request(body, STATE["presets"])
            if err:
                self._send_json(400, {"error": err})
                return
            if (parsed["config"] in RETRIEVAL_CONFIGS
                    and STATE["embedder_state"] != "ready"):
                self._send_json(503, {"error": "embedding model not ready "
                                      f"({STATE['embedder_state']})"})
                return

            # Non-blocking run lock: serializes INDIVIDUAL /api/run requests
            # (sound per-config token attribution); a second tab gets 409.
            # Held through telemetry and response construction, released in
            # the outermost finally — including client-disconnect paths.
            if not STATE["run_lock"].acquire(blocking=False):
                self._send_json(409, {"error": "a run is in progress"})
                return
            try:
                self._send_json(200, execute_run(parsed, STATE))
            finally:
                STATE["run_lock"].release()
        except (BrokenPipeError, ConnectionResetError):
            pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default: loopback only). Use "
                             "0.0.0.0 to reach the demo from another device on "
                             "the LAN — the server has no authentication, so "
                             "anyone who can route to this port can spend the "
                             "configured API key")
    parser.add_argument("--call-log", type=Path, default=CALL_LOG_PATH,
                        help="append-only JSONL record of every run "
                             "(question, QIDs, per-config stats)")
    parser.add_argument("--no-call-log", action="store_true",
                        help="run without recording calls")
    parser.add_argument("--model", default=REFERENCE_MODEL,
                        help="answering model (default: the thesis model of "
                             "record; overriding marks the demo non-reference)")
    args = parser.parse_args(argv)

    if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("No LLM_API_KEY / OPENAI_API_KEY in the environment.\n"
              "Run from the repo root so .env is found, or set the key first.")
        return 1

    STATE["non_reference_axes"] = pin_reference_configuration(args.model)
    # Derived from the axes so the badge and the axis list cannot disagree:
    # what answers is the (model, endpoint) pair, not the model name alone.
    STATE["model_is_reference"] = not any(
        a.startswith(("model=", "base_url="))
        for a in STATE["non_reference_axes"])

    STATE["call_log"] = None if args.no_call_log else args.call_log

    install_cache_overlay(STATE)
    llm_config.set_rpm_limit(llm_config.default_rpm())
    load_presets(STATE)
    threading.Thread(target=warm_embedder, args=(STATE,), daemon=True).start()

    print("TFM Graph-RAG demo — Anteproyecto Objetivo 4")
    print(f"  model:      {llm_config.MODEL}"
          + ("" if STATE["model_is_reference"]
             else "   ⚠️ NON-REFERENCE MODEL/ENDPOINT"))
    if STATE["non_reference_axes"]:
        print(f"  ⚠️ non-reference axes: {', '.join(STATE['non_reference_axes'])}")
    else:
        print("  configuration: all frozen axes pinned "
              "(k=30, global allocation, entity block all, value-desc off, "
              "property_type filter, answer cap 128, condense cap 384)")
    print(f"  rate limit: {llm_config.default_rpm() or 'off'} rpm (process-local — "
          "do not run alongside an evaluation on the same endpoint)")
    print("  retrieval:  strict (a failed retrieval is an error row)")
    print(f"  demo cache: {DEMO_CACHE_ROOT}  (frozen corpus is never written)")
    print(f"  presets:    {len(STATE['preset_order']) or 'disabled'}")
    print(f"  call log:   {STATE['call_log'] or 'disabled'}")
    print(f"\n  → http://127.0.0.1:{args.port}/\n")
    if args.host != "127.0.0.1":
        print(f"  ⚠️ bound to {args.host} — reachable from the network, "
              "unauthenticated\n")

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
