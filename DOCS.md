# Tool Reference

What each tool does, in plain terms, and when the agent reaches for it. This
mirrors the descriptions the model itself sees, minus the schema noise.

The agent has 11 tools. One (`browser`) covers direct page interaction; the
rest handle output, scale, and verification.

---

## `browser`

The core tool — drives the live Playwright page. Takes an `action` plus
whatever parameters that action needs (see below).

**Navigation & waiting**
- `navigate` — go to a URL, or `"back"` / `"forward"` through history. Takes a screenshot automatically.
- `wait` — pause for N seconds (0–100). The prompted escape hatch for pages slower than the built-in settle delay.

**Reading the page — the primary way the agent understands what's on screen**
- `read_page` — the DOM tree with element references (`ref_N`), optionally filtered to interactive elements only.
- `get_page_text` — all visible text content.
- `find` — semantic search: describe what you're looking for, get back matching element refs. Doesn't mark anything visually — pair with `outline` for that.
- `execute_js` — run arbitrary JavaScript in the page, returns the last expression's value. Also the way to reach content `read_page` can't see (e.g. same-origin iframes).

Every DOM-mutating action's result includes a diff against the page's last
known state, so the model sees what changed without a separate `read_page`.

**Interacting with elements**
- `left_click` / `right_click` / `middle_click` / `double_click` / `triple_click` — target by `ref` (preferred) or `coordinate`.
- `hover` — move the cursor without clicking (tooltips, dropdown reveals).
- `left_click_drag` — drag from `start_coordinate` to `coordinate`.
- `left_mouse_down` / `left_mouse_up` — raw press/release, for drag sequences the click-drag shorthand can't express.
- `type` — type text at the current cursor position.
- `key` / `hold_key` — press or hold a key combination.
- `form_input` — set a form field's value directly by `ref`, without simulating keystrokes.

**Scrolling & visual capture**
- `scroll` — directional scroll by N units. Meant for short hops when you're already close to the target — not for exploring a page.
- `scroll_to` — jump straight to an element by `ref`. The preferred way to reach something specific; avoids the old failure mode of scrolling blindly in small increments.
- `screenshot` — capture the current viewport (`full_page: true` for the whole scrollable page in one image, instead of scrolling-and-recapturing repeatedly).
- `zoom` — a zoomed screenshot of one rectangular `region`.
- `outline` — draw a bounding box around one element (by `ref`) and screenshot it. Use right before treating that element as evidence for a specific claim, to visually confirm it's the right one.

**Network inspection** — for content that never reaches the rendered DOM (the JSON behind an XHR call, same-origin iframe data):
- `network_list_types` — a first-glance breakdown of captured traffic this session, by resource type and host.
- `network_list` — list captured responses (most recent first), optionally filtered by a substring/status-code match.
- `network_inspect` — full detail (status, content-type, body) for one response, by the id shown in `network_list`.

---

## `save_file`

Writes a text deliverable (CSV, JSON, markdown report, …) to the run's
output directory, where it shows up in the Streamlit sidebar for download.

Exists because triggering a browser download saves the file inside the
containerized Chromium's own filesystem, not somewhere the user can
actually find it — this is the correct way to hand back a generated file.

---

## `dispatch_subagents`

Fans out a batch of **3+ similar items** to independent workers running
concurrently, each with its own browser tab and the exact same shared
instructions — so results are consistent across the batch. Returns one
result (or error) per item, in input order, validated against a schema you
provide.

Use for items that need real per-item judgment or interaction (a page whose
layout varies, a multi-step flow). For pure data extraction it's more
expensive than `batch_extract` or `run_script` — those don't spin up an LLM
per item.

---

## `batch_extract`

Pulls the same structured data from **many URLs at once**, without visiting
them one at a time. Fetches all URLs concurrently — reusing the current
session's cookies, so authenticated and cross-origin URLs work — and runs
your `extract_js` against each page's raw HTML.

Only sees what's in the page's **initial HTML**. If the data you need is
rendered client-side by JavaScript after load, `batch_extract` won't find
it — reach for `run_script` (with a `navigate` step) or `network_inspect`
instead. Backs off and retries automatically if the target starts
rate-limiting you.

---

## `run_script`

Writes one fixed recipe of browser-tool-shaped steps
(`navigate` / `screenshot` / `get_page_text` / `execute_js` / `wait` /
`scroll_to` / the `network_*` actions) and replays it across every item in
a list, concurrently, without spinning up a fresh LLM turn per item.

If the script's first step isn't `navigate`, it runs in "fast mode" —
fetching each item as a URL over raw HTTP instead of rendering it, the same
mechanism `batch_extract` uses. This is the tool to reach for whenever a
task involves several similar items and the same handful of fixed steps
would otherwise mean many manual tool calls one at a time. Not for items
that need real interaction or judgment — that's `dispatch_subagents`.

---

## `verify_finding`

Gets an **independent, fresh re-derivation** of one specific claim before
the agent relies on it — a separate session with no visibility into how the
original answer was reached, so it can't just rubber-stamp the same
reasoning.

Meant to be used sparingly, only when a finding is all three of:
**specific** (a discrete value — a number, name, date, id), **checkable**
(there's a concrete way to confirm it independently), and **consequential**
(getting it wrong would make the final deliverable wrong). Budgeted per
session — not for routine intermediate steps.

---

## Sub-browser queue: `queue_screenshots`, `queue_status`, `queue_clear`, `queue_pause`, `queue_resume`

A queue of independent browser tabs for **visual reconnaissance across many
pages at once** — navigate, scroll to something, capture a screenshot. Not
for interaction or reasoning (no clicking, no forms); for that, use
`dispatch_subagents`.

- **`queue_screenshots`** — add one or more instruction sequences to the queue. Returns immediately with queued item ids; does not wait for completion. Items run concurrently (bounded by a configurable fan-out) and paced by a throttle interval, each in its own page. The user can watch them working live in a panel next to the chat.
- **`queue_status`** — check progress: pending/running counts, plus the most recently completed items (screenshots, a step-by-step log, and the error if one failed).
- **`queue_clear`** — drop all pending and completed items. In-progress items finish normally, not force-cancelled.
- **`queue_pause`** / **`queue_resume`** — halt new items from starting (in-progress ones still finish), then continue.

---

## The safety layer

Every `browser` call is checked against a deterministic pattern-match
(`browser_use_demo/guardrails.py`) before it runs — not an LLM judge, so
there's nothing to prompt-inject or convince. It looks for actions that are
plausibly irreversible or destructive: navigating to a delete/cancel/
unsubscribe URL, clicking a button labeled "Delete Account," "Place Order,"
"Confirm Purchase," and similar. A blocked action returns an error instead
of executing.

The restriction level is a sidebar toggle:

- **None** — no blocking (default).
- **All** — matched actions are blocked outright.
- **Manual** — matched actions are blocked, but the chat shows a prompt to
  bypass that specific block if the user wants to proceed anyway.
