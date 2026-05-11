"""Mini-browse environment — v1 (Taskset / Harness / Env).

Reimplements the v0 ``MiniBrowseEnv`` (``StatefulToolEnv`` subclass) using the
verifiers v1 authoring API so that the same browser-agent toolset runs through
the ``Taskset -> Harness -> Env`` lifecycle.

Requires the ``mini-browse-env`` package to be installed (provides
``mini_browse_runtime``).

Usage (standalone)::

    # pyproject.toml points verifiers at local checkout or branch
    uv run vf-eval mini-browse-env-v1 \\
      -m openai/gpt-4.1-mini \\
      -a '{"backend": "browserbase", "dataset_name": "smoke"}' \\
      -n 1 -r 1

Usage (programmatic)::

    from verifiers.v1.packages.environments.mini_browse import load_environment
    env = load_environment(dataset_name="smoke", backend="browserbase")
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as v1
from verifiers.decorators import cleanup, metric, reward, setup

logger = logging.getLogger(__name__)

BackendName = Literal["browserbase", "perplexity"]
DatasetName = Literal["smoke", "webvoyager", "flights_no_year"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCREENSHOT_INDEX_SYSTEM_ADDENDUM = """\
<screenshot_indexing>
Every screenshot in this conversation is tagged with a unique label \
`[screenshot_N]` (e.g. `[screenshot_1]`, `[screenshot_2]`). The tag appears \
in the surrounding text describing the screenshot.

When you reference a specific screenshot in your reasoning, cite it by this \
tag — for example: "the flight in [screenshot_3] is the lowest fare". Older \
screenshot bytes may be pruned from context to save tokens, but the tag \
remains in the transcript so earlier observations are still referenceable.

When you call `submit_result`, you MUST cite the single screenshot whose \
visible content most directly supports your answer:
- `grounding_screenshot_id`: the integer N from `[screenshot_N]`.
- `grounding_claim`: one short, specific, factual statement that is \
verifiable just by looking at that screenshot — e.g. "the search results \
page shows a green Xbox Wireless Controller priced at $59.99".

How the verifier scores your submission:
- An independent vision verifier reads your `answer`, your \
`grounding_claim`, and the cited screenshot.
- It enumerates every concrete detail in your `answer` (specific flight \
number, exact price, exact duration, exact times, exact product title, \
etc.) and requires each to be verifiable in the cited screenshot.
- A trivially-true `grounding_claim` will NOT save a fabricated `answer`. \
The screenshot must show the actual specifics in your answer.

Anti-hallucination rules — your submission will score zero if your `answer` \
contains any of these:
- Hedging language: "typical", "approximately", "usually", "around", \
"would be", "estimated", "based on common data".
- Ranges instead of single observed values (e.g. "$2,500-$3,500", \
"15-16 hours").
- Placeholders or category labels (e.g. "[airline name]", "best nonstop \
option").
- Wording like "I could not access" / "based on typical data" / "I'll \
provide an estimate".
- Details that are not actually visible in the cited screenshot.

If you genuinely could not complete the task (e.g. tools failed, page \
blocked you), say so plainly in your answer instead of guessing — \
fabrication is punished harder than failure.
</screenshot_indexing>"""

JUDGE_PROMPT = (
    "You are grading a browser-automation agent's submission to a web task. "
    "The submission is correct only if the answer completes the task AND "
    "every concrete detail in the answer is visible in the cited screenshot."
    "\n\n"
    "You will receive:\n"
    "- TASK: what the user asked the agent to do.\n"
    "- ANSWER: the agent's submitted final answer.\n"
    "- GROUNDING_CLAIM: a short statement the agent says is verifiable in "
    "the screenshot. Informational only — never credit the answer just "
    'because the claim is trivially true (e.g. "this is the Google Flights '
    'homepage").\n'
    "- The cited SCREENSHOT.\n\n"
    "Procedure:\n"
    "1. Enumerate every concrete factual detail in the ANSWER that the task "
    "required (titles, prices, names, numbers, durations, dates, URLs, etc.).\n"
    "2. For each detail, find supporting evidence in the SCREENSHOT — either "
    "a literal verbatim quote or an unambiguous visual element. If even one "
    'required detail is missing, contradicted, or invisible, the verdict is "no".\n'
    "3. Use ONLY the attached screenshot — never prior knowledge of the site, "
    '"typical" values, or what content "would" be there.\n\n'
    'HALLUCINATION FAIL — verdict is "no" if the ANSWER contains any of:\n'
    '- Hedges: "typical", "approximately", "usually", "around", "based on '
    'common data", "would be", "on average", "estimated", "approx."\n'
    "- Ranges instead of single observed values.\n"
    "- Placeholders or category labels.\n"
    '- Apologies/refusals: "I could not access", "I was unable to", '
    '"based on typical data".\n\n'
    'EMPTY-PAGE FAIL — verdict is "no" if the screenshot shows only a search '
    "form, an empty homepage, an error page, a CAPTCHA, a login wall, or a "
    '"no results" state.\n\n'
    'WRONG-PAGE FAIL — verdict is "no" if the screenshot shows a different '
    "page or different item than the answer describes.\n\n"
    'OFF-TASK FAIL — verdict is "no" if the answer addresses a different task '
    "than asked, is empty, or says the agent could not complete the task.\n\n"
    'Submit your decision via the `submit_verdict` tool: set `verdict` to "yes" '
    'or "no", and write 2-4 sentences in `verdict_reasoning`.'
)

_VERDICT_TOOL: dict[str, Any] = {
    "name": "submit_verdict",
    "description": (
        "Return your verdict on whether the agent's answer is correct AND "
        "grounded in the cited screenshot."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["yes", "no"],
                "description": (
                    "yes if the answer is correct and every concrete detail "
                    "is visible in the screenshot; no otherwise."
                ),
            },
            "verdict_reasoning": {
                "type": "string",
                "description": (
                    "2-4 sentences listing the concrete details checked, the "
                    "visible evidence found or missing, and any triggers."
                ),
            },
        },
        "required": ["verdict", "verdict_reasoning"],
    },
}

_HALLUCINATION_MARKER_RE = re.compile(
    r"\b(typical(?:ly)?|approximately|usually|around\b|on average|estimat\w+|"
    r"would be|based on (?:typical|common)|approx\.?|i could not|i was unable)\b",
    re.IGNORECASE,
)

SMOKE_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "prime_intellect_latest_blog",
        "url": "https://www.primeintellect.ai/blog",
        "description": (
            "Go to the Prime Intellect blog at "
            "https://www.primeintellect.ai/blog, identify the most recent "
            "blog post, and summarize it in a few sentences. "
            "Call submit_result with your summary."
        ),
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_backend(
    backend_name: BackendName, backend_kwargs: dict[str, Any] | None = None
) -> Any:
    from mini_browse_runtime.bcu.backends import (  # type: ignore[import-not-found]
        BrowserbaseBackend,
        PerplexityBackend,
    )

    kwargs = dict(backend_kwargs or {})
    if backend_name == "browserbase":
        return BrowserbaseBackend(**kwargs)
    if backend_name == "perplexity":
        return PerplexityBackend(**kwargs)
    raise ValueError(f"Unknown backend: {backend_name!r}")


def _next_screenshot_id(state: dict[str, Any]) -> int:
    n = int(state.get("_mb_screenshot_count", 0)) + 1
    state["_mb_screenshot_count"] = n
    return n


def _archive_screenshot(
    state: dict[str, Any], screenshot_id: int, b64_data: str, media_type: str
) -> None:
    archive = state.setdefault("_mb_screenshot_archive", {})
    archive[screenshot_id] = {"data": b64_data, "media_type": media_type}


def _extract_task_text(state: dict[str, Any]) -> str:
    prompt = state.get("prompt") or []
    for msg in reversed(prompt):
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text", ""))
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    return text
    return ""


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        result = json.loads(s)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _record_verdict(state: dict[str, Any] | None, verdict: str, reasoning: str) -> None:
    if state is None:
        return
    state["_mb_judge_verdict"] = verdict
    state["_mb_judge_reasoning"] = reasoning


# ---------------------------------------------------------------------------
# Tool implementations — standalone async functions for v1 Toolset.
#
# In v1, tools are plain functions. ``state`` is injected via the Toolset
# binding mechanism (hidden from the model's tool schema).
# ---------------------------------------------------------------------------


async def _tool_exec(
    tool_name: str, args: dict[str, Any], state: dict[str, Any]
) -> str | list[dict[str, Any]]:
    """Run a mini-browse-runtime tool, return multipart content when images exist.

    Returns a list of content-part dicts (text + image_url) when the tool
    produces screenshots or system reminders.  The v1 ``base_program`` loop
    checks ``is_valid_tool_content_parts(result)`` and forwards the list
    directly as the ``ToolMessage`` content, so the model sees the images.
    Falls back to a plain JSON string when there are no images.
    """
    from mini_browse_runtime.runtime import (  # type: ignore[import-not-found]
        MiniBrowseContext,
        execute_tool,
    )

    if state.get("_mb_setup_error") or state.get("_mb_ctx") is None:
        err = state.get("_mb_setup_error") or "browser unavailable"
        return json.dumps({"error": f"Browser unavailable: {err}"}, ensure_ascii=False)
    ctx: MiniBrowseContext = state["_mb_ctx"]
    result = await execute_tool(tool_name, args, ctx)

    if result.images or result.system_reminder:
        text_lines = [f"Updated state after `{tool_name}`."]
        if result.system_reminder:
            text_lines.append(
                f"<system-reminder>{result.system_reminder}</system-reminder>"
            )
        image_ids: list[int] = []
        for image in result.images:
            sid = _next_screenshot_id(state)
            _archive_screenshot(state, sid, image.data, image.media_type)
            image_ids.append(sid)
        if image_ids:
            tags = ", ".join(f"[screenshot_{i}]" for i in image_ids)
            text_lines.append(
                "Use the attached screenshot(s) as the latest browser "
                f"state. Tagged: {tags}."
            )
        # Include the JSON payload as text so the model still sees it.
        text_lines.append(json.dumps(result.payload, ensure_ascii=False))
        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": "\n\n".join(text_lines)}
        ]
        for image in result.images:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.media_type};base64,{image.data}"
                    },
                }
            )
        return content_parts

    return json.dumps(result.payload, ensure_ascii=False)


async def navigate(
    url: str,
    user_description: str = "",
    tab_id: int | None = None,
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Navigate the current tab to a URL, or use 'forward'/'back' for history."""
    return await _tool_exec(
        "navigate",
        {"url": url, "user_description": user_description, "tab_id": tab_id},
        state,
    )


async def computer(
    actions: list[Any],
    user_description: str = "",
    tab_id: int | None = None,
    save_to_workspace: bool = False,
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Execute a batched sequence of low-level browser actions."""
    return await _tool_exec(
        "computer",
        {
            "actions": actions,
            "user_description": user_description,
            "tab_id": tab_id,
            "save_to_workspace": save_to_workspace,
        },
        state,
    )


async def read_page(
    user_description: str = "",
    depth: int | None = None,
    filter: str = "all",
    ref_id: str | None = None,
    tab_id: int | None = None,
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Accessibility-tree view. filter: 'interactive'|'all'|'viewport'."""
    return await _tool_exec(
        "read_page",
        {
            "user_description": user_description,
            "depth": depth,
            "filter": filter,
            "ref_id": ref_id,
            "tab_id": tab_id,
        },
        state,
    )


async def form_input(
    ref: str,
    value: str,
    user_description: str = "",
    tab_id: int | None = None,
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Set a form field using a ref_id from read_page."""
    return await _tool_exec(
        "form_input",
        {
            "ref": ref,
            "value": value,
            "user_description": user_description,
            "tab_id": tab_id,
        },
        state,
    )


async def get_page_text(
    user_description: str = "",
    tab_id: int | None = None,
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Get the current page's visible text."""
    return await _tool_exec(
        "get_page_text",
        {"user_description": user_description, "tab_id": tab_id},
        state,
    )


async def find(
    query: str,
    user_description: str = "",
    tab_id: int | None = None,
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Keyword-rank elements matching a natural-language query."""
    return await _tool_exec(
        "find",
        {"query": query, "user_description": user_description, "tab_id": tab_id},
        state,
    )


async def tabs_context(*, state: dict[str, Any]) -> str | list[dict[str, Any]]:
    """List current tabs."""
    return await _tool_exec("tabs_context", {}, state)


async def tabs_create(
    url: str = "about:blank",
    user_description: str = "",
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Open a new tab, optionally at a URL."""
    return await _tool_exec(
        "tabs_create",
        {"url": url, "user_description": user_description},
        state,
    )


async def wait_for_download(
    user_description: str = "",
    guid: str | None = None,
    timeout: int = 30,
    *,
    state: dict[str, Any],
) -> str | list[dict[str, Any]]:
    """Wait for a file download to finish."""
    return await _tool_exec(
        "wait_for_download",
        {"user_description": user_description, "guid": guid, "timeout": timeout},
        state,
    )


async def submit_result(
    answer: str,
    grounding_screenshot_id: int,
    grounding_claim: str,
    *,
    state: v1.State,
) -> str:
    """Submit your final answer with a screenshot citation.

    answer: your final answer to the user's task.
    grounding_screenshot_id: the integer N from the [screenshot_N] tag of the
      single screenshot whose visible content most directly supports your
      answer.
    grounding_claim: one short, specific, factual statement about what is
      visible in that screenshot that supports your answer.
    """
    state["submitted_result"] = {
        "answer": answer,
        "grounding_screenshot_id": grounding_screenshot_id,
        "grounding_claim": grounding_claim,
    }
    state.stop("submitted")
    return json.dumps({"submitted": True})


# ---------------------------------------------------------------------------
# Toolset: browser tools + lifecycle
# ---------------------------------------------------------------------------


def build_browse_toolset(
    backend: BackendName = "browserbase",
    backend_kwargs: dict[str, Any] | None = None,
    browser_start_max_retries: int = 5,
    browser_start_base_backoff_seconds: float = 1.5,
    browser_start_max_backoff_seconds: float = 30.0,
    browser_start_backoff_jitter_seconds: float = 0.5,
    browser_start_min_interval_seconds: float = 0.0,
    browser_start_jitter_seconds: float = 0.5,
    browser_start_max_in_flight: int = 0,
) -> v1.Toolset:
    """Build a Toolset packaging all browser tools with setup/cleanup."""
    browser_start_kwargs = {
        "browser_start_max_retries": browser_start_max_retries,
        "browser_start_base_backoff_seconds": browser_start_base_backoff_seconds,
        "browser_start_max_backoff_seconds": browser_start_max_backoff_seconds,
        "browser_start_backoff_jitter_seconds": browser_start_backoff_jitter_seconds,
        "browser_start_min_interval_seconds": browser_start_min_interval_seconds,
        "browser_start_jitter_seconds": browser_start_jitter_seconds,
        "browser_start_max_in_flight": browser_start_max_in_flight,
    }

    @setup
    async def setup_browser(task: dict[str, Any], state: dict[str, Any]) -> None:
        """Create a browser session and prime the initial prompt."""
        from mini_browse_runtime.bcu.session import (  # type: ignore[import-not-found]
            SCREENSHOT_MEDIA_TYPE,
            BrowserSession,
        )
        from mini_browse_runtime.prompts import (  # type: ignore[import-not-found]
            build_bcu_system_prompt,
            format_user_context,
        )
        from mini_browse_runtime.runtime import (  # type: ignore[import-not-found]
            MiniBrowseContext,
        )

        workspace = Path(tempfile.mkdtemp(prefix="mini-browse-v1-"))
        state["_mb_workspace"] = str(workspace)

        info = task.get("info") or {}
        if not isinstance(info, dict):
            info = {}

        jitter = random.uniform(1.0, 5.0)
        state["_mb_browser_create_jitter_seconds"] = jitter
        await asyncio.sleep(jitter)

        browser_backend = _make_backend(backend, backend_kwargs)
        browser = BrowserSession(
            browser_backend=browser_backend, **browser_start_kwargs
        )
        state["_mb_browser_create_started_at"] = time.monotonic()

        try:
            await browser.start()
        except Exception as exc:
            elapsed = time.monotonic() - state["_mb_browser_create_started_at"]
            state["browser_create_attempts"] = browser.create_attempts
            state["browser_create_failures"] = browser.create_failures
            state["browser_create_succeeded"] = 0
            state["browser_create_last_status"] = browser.last_create_error_status
            state["browser_create_elapsed_seconds"] = elapsed
            state["_mb_setup_traceback"] = traceback.format_exc()
            state["_mb_setup_error_type"] = type(exc).__name__
            logger.exception("MINIBROWSE_V1_BROWSER_START_FAILED backend=%s", backend)
            try:
                await browser.close()
            except Exception:
                pass
            error_str = f"{type(exc).__name__}: {exc}"
            state["_mb_setup_error"] = error_str
            state["_mb_browser"] = None
            state["_mb_ctx"] = None
            state["browser_session_id"] = None
            state["browser_backend"] = backend
            state["_mb_task_text"] = _extract_task_text(state)
            state["submitted_result"] = {
                "answer": (
                    "Could not complete task: browser session failed to "
                    f"initialize. {error_str}"
                ),
                "grounding_screenshot_id": -1,
                "grounding_claim": (
                    "Browser was never initialized; no screenshot exists."
                ),
            }
            state.stop("setup_failed")  # type: ignore[union-attr]
            return

        elapsed = time.monotonic() - state["_mb_browser_create_started_at"]
        state["browser_create_attempts"] = browser.create_attempts
        state["browser_create_failures"] = browser.create_failures
        state["browser_create_succeeded"] = 1
        state["browser_create_last_status"] = browser.last_create_error_status
        state["browser_create_elapsed_seconds"] = elapsed
        state["_mb_browser"] = browser
        state["_mb_ctx"] = MiniBrowseContext(browser=browser, workspace_root=workspace)
        state["browser_session_id"] = browser.browser_session_id
        state["browser_backend"] = backend

        logger.info(
            "MINIBROWSE_V1_BROWSER_SESSION_CREATED backend=%s session_id=%s",
            backend,
            browser.browser_session_id,
        )

        # -- Initial navigation, screenshot, and prompt priming --
        url = info.get("url")
        nav_error: str | None = None
        if url:
            try:
                await browser.navigate(url)
            except Exception as nav_exc:
                nav_error = (
                    f"Initial navigation to {url} failed: {nav_exc}. "
                    "Use navigate to recover."
                )

        screenshot_bytes: bytes | None = None
        try:
            screenshot_bytes = await browser.screenshot()
        except Exception:
            screenshot_bytes = None
        try:
            tab_context = browser.tabs.get_tab_context().to_dict()
        except Exception:
            tab_context = {}

        task_text = _extract_task_text(state)
        state["_mb_task_text"] = task_text

        tab_ctx_json = json.dumps({"tab_context": tab_context}, ensure_ascii=False)
        screenshot_id: int | None = None
        if screenshot_bytes is not None:
            screenshot_id = _next_screenshot_id(state)
            screenshot_line = (
                "Attached is a screenshot of the current tab "
                f"(tab_id={tab_context.get('current_tab_id', 0)}), "
                f"tagged [screenshot_{screenshot_id}]."
            )
        else:
            screenshot_line = "Screenshot of the current tab was unavailable."

        text = (
            f"{format_user_context()}\n\n"
            f"Task: {task_text}\n\n"
            f"{tab_ctx_json}\n\n"
            f"{screenshot_line}"
        )
        if nav_error:
            text = f"[SYSTEM: {nav_error}]\n\n{text}"

        user_content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if screenshot_bytes is not None and screenshot_id is not None:
            encoded = base64.b64encode(screenshot_bytes).decode("ascii")
            _archive_screenshot(state, screenshot_id, encoded, SCREENSHOT_MEDIA_TYPE)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (f"data:{SCREENSHOT_MEDIA_TYPE};base64,{encoded}")
                    },
                }
            )

        model_name = None
        runtime = state.get("runtime")
        if isinstance(runtime, dict):
            model_name = runtime.get("model")

        system_text = build_bcu_system_prompt(
            model_name,
            has_workspace_tools=False,
            has_todo_tools=False,
            use_local_browser_context=False,
        )
        system_text = f"{system_text}\n\n{SCREENSHOT_INDEX_SYSTEM_ADDENDUM}"

        state["system_prompt"] = [{"role": "system", "content": system_text}]
        state["prompt"] = [{"role": "user", "content": user_content}]

    @cleanup
    async def cleanup_browser(task: dict[str, Any], state: dict[str, Any]) -> None:
        """Close browser session and clean up state."""
        browser = state.get("_mb_browser")
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                logger.exception("mini_browse_v1.browser_close_failed")
        import shutil

        state.pop("_mb_browser", None)
        workspace = state.pop("_mb_workspace", None)
        if workspace is not None:
            try:
                shutil.rmtree(workspace, ignore_errors=True)
            except Exception:
                pass
        state.pop("_mb_ctx", None)

    return v1.Toolset(
        tools=[
            navigate,
            computer,
            read_page,
            form_input,
            get_page_text,
            find,
            tabs_context,
            tabs_create,
            wait_for_download,
            submit_result,
        ],
        write=True,
        scope="rollout",
        setups=[setup_browser],
        cleanups=[cleanup_browser],
    )


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _build_smoke_rows() -> list[dict[str, Any]]:
    return [
        {
            "prompt": [{"role": "user", "content": task["description"]}],
            "answer": "",
            "task": task["task_id"],
            "info": {"url": task["url"]},
        }
        for task in SMOKE_TASKS
    ]


def _build_webvoyager_rows(
    num_examples: int = -1,
    web_filter: str | None = None,
    datasets_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    if datasets_dir is not None:
        path = Path(datasets_dir) / "WebVoyager_data_clean.jsonl"
    else:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "mini_browse_env"
            / "datasets"
            / "WebVoyager_data_clean.jsonl"
        )
    if not path.exists():
        raise FileNotFoundError(f"WebVoyager dataset not found at {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            if web_filter and item.get("web_name") != web_filter:
                continue
            question = (
                f"{item['ques']}\n\n"
                f"Start at: {item['web']}\n\n"
                "When you have the answer, call submit_result with your "
                "final answer."
            )
            rows.append(
                {
                    "prompt": [{"role": "user", "content": question}],
                    "answer": "",
                    "task": item["id"],
                    "info": {"url": item["web"], "website": item["web_name"]},
                }
            )
    if num_examples > 0:
        rows = rows[:num_examples]
    return rows


def _build_flights_no_year_rows(
    num_examples: int = -1,
    datasets_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    if datasets_dir is not None:
        path = Path(datasets_dir) / "flights_no_year.jsonl"
    else:
        path = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "mini_browse_env"
            / "datasets"
            / "flights_no_year.jsonl"
        )
    if not path.exists():
        raise FileNotFoundError(f"flights_no_year dataset not found at {path}")
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            question = (
                f"{item['ques']}\n\n"
                f"Start at: {item['web']}\n\n"
                "When you have the answer, call submit_result with your "
                "final answer."
            )
            rows.append(
                {
                    "prompt": [{"role": "user", "content": question}],
                    "answer": "",
                    "task": item["id"],
                    "info": {"url": item["web"], "website": item["web_name"]},
                }
            )
    if num_examples > 0:
        rows = rows[:num_examples]
    return rows


# ---------------------------------------------------------------------------
# Rewards & metrics
# ---------------------------------------------------------------------------


def build_judge_reward(
    judge_model: str = "claude-opus-4-7",
    judge_base_url: str | None = None,
    judge_api_key_var: str = "ANTHROPIC_API_KEY",
    judge_max_tokens: int = 600,
) -> Any:
    """Build a weight-1 reward function that calls a vision LLM judge."""
    judge_provider = "anthropic" if "claude" in judge_model.lower() else "openai"

    api_key = os.environ[judge_api_key_var]

    anthropic_client: Any = None
    openai_client: Any = None
    if judge_provider == "anthropic":
        from anthropic import AsyncAnthropic

        anthropic_client = (
            AsyncAnthropic(
                api_key=api_key,
                base_url=judge_base_url,
            )
            if judge_base_url
            else AsyncAnthropic(api_key=api_key)
        )
    else:
        from openai import AsyncOpenAI

        openai_client = (
            AsyncOpenAI(
                api_key=api_key,
                base_url=judge_base_url,
            )
            if judge_base_url
            else AsyncOpenAI(api_key=api_key)
        )

    @reward(weight=1.0)
    async def judge_reward_func(task: dict[str, Any], state: dict[str, Any]) -> float:
        """1.0 iff every concrete detail in the answer is visible in the
        cited screenshot."""
        submitted = state.get("submitted_result")
        if not isinstance(submitted, dict):
            _record_verdict(state, "no", "submission missing or not a dict")
            return 0.0
        answer_text = str(submitted.get("answer") or "").strip()
        sid = submitted.get("grounding_screenshot_id")
        claim = (submitted.get("grounding_claim") or "").strip()
        if not answer_text:
            _record_verdict(state, "no", "submission has no answer")
            return 0.0
        if not isinstance(sid, int):
            _record_verdict(state, "no", "submission missing grounding_screenshot_id")
            return 0.0
        if not claim:
            _record_verdict(state, "no", "submission missing grounding_claim")
            return 0.0
        archive = state.get("_mb_screenshot_archive") or {}
        image = archive.get(sid) or archive.get(str(sid))
        if not isinstance(image, dict) or not image.get("data"):
            _record_verdict(state, "no", f"cited screenshot id {sid} not in archive")
            return 0.0

        task_text = state.get("_mb_task_text") or ""
        user_text = (
            f"TASK:\n{task_text}\n\nANSWER:\n{answer_text}\n\nGROUNDING_CLAIM:\n{claim}"
        )

        try:
            if anthropic_client is not None:
                response = await anthropic_client.messages.create(
                    model=judge_model,
                    max_tokens=judge_max_tokens,
                    system=JUDGE_PROMPT,
                    tools=[_VERDICT_TOOL],  # type: ignore[arg-type]
                    tool_choice={
                        "type": "tool",
                        "name": "submit_verdict",
                    },
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image["media_type"],
                                        "data": image["data"],
                                    },
                                },
                                {"type": "text", "text": user_text},
                            ],
                        }
                    ],
                )
                payload = next(
                    (
                        block.input
                        for block in response.content
                        if getattr(block, "type", None) == "tool_use"
                        and getattr(block, "name", None) == "submit_verdict"
                    ),
                    None,
                )
                if not isinstance(payload, dict):
                    _record_verdict(
                        state,
                        "no",
                        "judge returned no tool_use block; "
                        f"stop_reason={response.stop_reason}",
                    )
                    return 0.0
            elif openai_client is not None:
                data_url = f"data:{image['media_type']};base64,{image['data']}"
                response = await openai_client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_text},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                            ],
                        },
                    ],
                    max_completion_tokens=judge_max_tokens,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content or ""
                payload = _parse_json_payload(text)
                if payload is None:
                    _record_verdict(
                        state,
                        "no",
                        f"judge JSON parse failed; raw={text[:200]}",
                    )
                    return 0.0
        except Exception as exc:
            logger.exception(
                "mini_browse_v1.judge_failed model=%s err=%s",
                judge_model,
                exc,
            )
            _record_verdict(state, "no", f"judge call failed: {type(exc).__name__}")
            return 0.0

        verdict = str(payload.get("verdict", "")).strip().lower()
        reasoning = (
            str(payload.get("verdict_reasoning", "")).strip()
            or "(no reasoning provided)"
        )
        if verdict not in ("yes", "no"):
            _record_verdict(
                state,
                "no",
                f"invalid verdict {verdict!r}; reasoning={reasoning[:200]}",
            )
            return 0.0
        _record_verdict(state, verdict, reasoning)
        logger.info(
            "mini_browse_v1.judge verdict=%s reasoning=%s",
            verdict,
            reasoning[:500],
        )
        return 1.0 if verdict == "yes" else 0.0

    return judge_reward_func


def build_metric_funcs() -> list[Any]:
    """Build weight-0 metric reward functions for eval diagnostics."""

    @metric
    async def browser_create_attempts(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        return float(state.get("browser_create_attempts", 0))

    @metric
    async def browser_create_retries(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        attempts = int(state.get("browser_create_attempts", 0))
        return float(max(0, attempts - 1))

    @metric
    async def browser_create_succeeded(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        return float(state.get("browser_create_succeeded", 0))

    @metric
    async def browser_create_elapsed_seconds(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        val = state.get("browser_create_elapsed_seconds")
        return float(val) if val is not None else -1.0

    @metric
    async def tool_call_error_count(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        return float(state.get("tool_call_error_count", 0))

    @metric
    async def screenshot_count(task: dict[str, Any], state: dict[str, Any]) -> float:
        return float(state.get("_mb_screenshot_count", 0))

    @metric
    async def submission_present(task: dict[str, Any], state: dict[str, Any]) -> float:
        return 1.0 if state.get("submitted_result") else 0.0

    @metric
    async def grounding_screenshot_id_valid(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        submitted = state.get("submitted_result")
        if not isinstance(submitted, dict):
            return 0.0
        sid = submitted.get("grounding_screenshot_id")
        if not isinstance(sid, int) or sid < 1:
            return 0.0
        archive = state.get("_mb_screenshot_archive") or {}
        return 1.0 if (sid in archive or str(sid) in archive) else 0.0

    @metric
    async def answer_has_hallucination_markers(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        submitted = state.get("submitted_result")
        if not isinstance(submitted, dict):
            return 0.0
        answer = str(submitted.get("answer") or "")
        return 1.0 if _HALLUCINATION_MARKER_RE.search(answer) else 0.0

    @metric
    async def cdp_disconnect_count(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        return float(state.get("cdp_disconnect_count", 0))

    @metric
    async def cdp_reconnect_count(task: dict[str, Any], state: dict[str, Any]) -> float:
        return float(state.get("cdp_reconnect_count", 0))

    @metric
    async def cdp_alive_seconds(task: dict[str, Any], state: dict[str, Any]) -> float:
        return float(state.get("cdp_alive_seconds", 0))

    @metric
    async def cdp_last_close_code(task: dict[str, Any], state: dict[str, Any]) -> float:
        code = state.get("cdp_last_close_code")
        return float(code) if isinstance(code, (int, float)) else -1.0

    @metric
    async def cdp_pending_at_close(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        val = state.get("cdp_pending_at_close")
        return float(val) if val is not None else 0.0

    @metric
    async def cdp_seconds_since_last_command_at_close(
        task: dict[str, Any], state: dict[str, Any]
    ) -> float:
        val = state.get("cdp_seconds_since_last_command_at_close")
        return float(val) if isinstance(val, (int, float)) else -1.0

    return [
        browser_create_attempts,
        browser_create_retries,
        browser_create_succeeded,
        browser_create_elapsed_seconds,
        tool_call_error_count,
        screenshot_count,
        submission_present,
        grounding_screenshot_id_valid,
        answer_has_hallucination_markers,
        cdp_disconnect_count,
        cdp_reconnect_count,
        cdp_alive_seconds,
        cdp_last_close_code,
        cdp_pending_at_close,
        cdp_seconds_since_last_command_at_close,
    ]


# ---------------------------------------------------------------------------
# Public API: load_environment
# ---------------------------------------------------------------------------


def load_environment(
    dataset_name: DatasetName = "smoke",
    num_examples: int = -1,
    web_filter: str | None = None,
    backend: BackendName = "browserbase",
    max_turns: int = 40,
    judge_model: str = "claude-opus-4-7",
    judge_base_url: str | None = None,
    judge_api_key_var: str = "ANTHROPIC_API_KEY",
    judge_max_tokens: int = 600,
    backend_kwargs: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    datasets_dir: str | Path | None = None,
    browser_start_max_retries: int = 5,
    browser_start_base_backoff_seconds: float = 1.5,
    browser_start_max_backoff_seconds: float = 30.0,
    browser_start_backoff_jitter_seconds: float = 0.5,
    browser_start_min_interval_seconds: float = 0.0,
    browser_start_jitter_seconds: float = 0.5,
    browser_start_max_in_flight: int = 0,
    enable_braintrust_tracing: bool = True,
    **kwargs: Any,
) -> v1.Env:
    """Entry point for ``vf-eval``.  Returns a v1 ``Env``.

    Set ``BROWSERBASE_API_KEY`` + ``BROWSERBASE_PROJECT_ID`` for
    ``backend="browserbase"``, or ``MINI_BROWSE_BROWSER_API_URL`` for
    ``backend="perplexity"``.

    Set ``BRAINTRUST_API_KEY`` and optionally ``VF_BRAINTRUST_PROJECT``
    to enable Braintrust tracing.
    """
    # Optionally enable Braintrust v1 tracing.
    if enable_braintrust_tracing:
        try:
            from verifiers.v1.experimental.braintrust_tracing import (  # type: ignore[import-not-found]
                setup_v1_tracing,
            )
        except ImportError:
            logger.debug("Braintrust v1 tracing not available; skipping.")
        else:
            setup_v1_tracing()

    if output_schema is None:
        import verifiers as vf

        vf.ensure_keys([judge_api_key_var])

    # ---- Dataset rows ----
    if dataset_name == "smoke":
        rows = _build_smoke_rows()
    elif dataset_name == "webvoyager":
        rows = _build_webvoyager_rows(
            num_examples=num_examples,
            web_filter=web_filter,
            datasets_dir=datasets_dir,
        )
    elif dataset_name == "flights_no_year":
        rows = _build_flights_no_year_rows(
            num_examples=num_examples, datasets_dir=datasets_dir
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name!r}")

    # ---- Toolset ----
    browse_toolset = build_browse_toolset(
        backend=backend,
        backend_kwargs=backend_kwargs,
        browser_start_max_retries=browser_start_max_retries,
        browser_start_base_backoff_seconds=browser_start_base_backoff_seconds,
        browser_start_max_backoff_seconds=browser_start_max_backoff_seconds,
        browser_start_backoff_jitter_seconds=browser_start_backoff_jitter_seconds,
        browser_start_min_interval_seconds=browser_start_min_interval_seconds,
        browser_start_jitter_seconds=browser_start_jitter_seconds,
        browser_start_max_in_flight=browser_start_max_in_flight,
    )

    # ---- Rewards & metrics ----
    rewards: list[Any] = []
    if output_schema is None:
        rewards.append(
            build_judge_reward(
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                judge_api_key_var=judge_api_key_var,
                judge_max_tokens=judge_max_tokens,
            )
        )
    metrics = build_metric_funcs()

    # ---- Taskset ----
    taskset = v1.Taskset(
        source=rows,
        taskset_id="mini-browse-env-v1",
        toolsets=[browse_toolset],
        rewards=rewards,
        metrics=metrics,
    )

    # ---- Harness (uses default base_program tool loop) ----
    harness = v1.Harness(max_turns=max_turns)

    # ---- Env ----
    return v1.Env(taskset=taskset, harness=harness)
