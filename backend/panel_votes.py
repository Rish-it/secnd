"""Registered S3R Amendment 4 panel: one independent vote per model per row; never sees the key."""

import argparse
import csv
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
PANEL = ("openai/gpt-6-luna", "google/gemini-3.5-flash-lite", "qwen/qwen3.8-flash")
# Amendment 5. Muse is reached through the Meta Model API at list price; the rest report their cost.
SECOND = ("muse-wonder-1.0", "openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5")
META_URL = "https://api.meta.ai/v1/chat/completions"
WONDER_PER_TOKEN = (4e-6, 20e-6)
BOTH, NEITHER, UNSURE = "Both fit equally", "Neither fits", "insufficient information"
PROMPTS = {
    "mtop": "A person typed the message below to a phone assistant. Pick the option that best "
    "describes what they want the assistant to do.",
    "banking77": "A customer sent the message below to their bank. Pick the option that best "
    "describes what the customer needs.",
    "ad": "Below is the state of an ad campaign. Pick what its manager should do next.",
    "inbox": "Below is a message a business received. Pick how the business should handle it.",
    "account": "Below is a change on a business account. Pick how a reviewer should treat it.",
}
RULE = (
    " Judge only the text given. Reply with the exact words of one option and nothing else.\n\n"
    "Text: {text}\n\nOptions:\n{options}"
)


def choices(row):
    if "option_a" in row:
        return [row["option_a"], row["option_b"], BOTH, NEITHER]
    return [*row["options"].split(" | "), UNSURE]


def request_for(row, model, domain):
    options = choices(row)
    seed = hashlib.sha256(f"{model}:{row['id']}".encode()).digest()
    random.Random(seed).shuffle(options)
    prompt = PROMPTS[domain or row["decision_type"]] + RULE.format(
        text=row.get("message") or row["text"], options="\n".join(f"- {o}" for o in options)
    )
    return prompt, options


def parse(reply, options):
    text = (reply or "").strip().strip("\"'`.*- ").casefold()
    return next((o for o in options if o.casefold() == text), None)


def overturned(votes, gold, candidate):
    """Amendment 3: a label changes only when at least 2 of 3 pick the other option."""
    assert gold != candidate
    return Counter(votes)[candidate] >= 2


def majority(votes):
    answer, count = Counter(v for v in votes if v).most_common(1)[0] if any(votes) else (None, 0)
    return answer if count >= 2 else None


def tally(rows, votes, key, panel=PANEL):
    """Disagreements and controls under Amendment 3; `key` is read here and never sent."""
    report = dict(changed=[], controls=0, controls_overturned=0)
    for row in rows:
        entry = key[row["id"]]
        cast = [votes.get((m, row["id"])) for m in panel]
        hit = overturned(cast, entry["gold"], entry["candidate"])
        if entry["kind"] == "control":
            report["controls"] += 1
            report["controls_overturned"] += hit
        if hit:
            report["changed"].append(
                dict(id=row["id"], kind=entry["kind"], old=entry["gold"], votes=cast)
            )
    report["biased"] = report["controls_overturned"] > 0.02 * report["controls"]
    return report


def call(client, model, prompt, key, meta_key):
    """One vote and its cost; Muse needs a low reasoning effort to leave room to answer."""
    body = dict(model=model, messages=[dict(role="user", content=prompt)], temperature=0)
    if model.startswith("muse-"):
        response = client.post(
            META_URL,
            headers={"Authorization": f"Bearer {meta_key}"},
            json=dict(body, max_tokens=2048, reasoning_effort="low"),
        )
    else:
        response = client.post(
            URL,
            headers={"Authorization": f"Bearer {key}"},
            json=dict(body, seed=20261030, max_tokens=1000, usage=dict(include=True)),
        )
    response.raise_for_status()
    reply = response.json()
    usage = reply.get("usage", {})
    if model.startswith("muse-"):
        cost = (usage.get("prompt_tokens", 0) * WONDER_PER_TOKEN[0]) + (
            usage.get("completion_tokens", 0) * WONDER_PER_TOKEN[1]
        )
    else:
        cost = float(usage.get("cost") or 0)
    return reply, cost


def run(rows, out, domain, cap, key, workers=16, panel=PANEL, meta_key=""):
    out = Path(out)
    done = {}
    if out.exists():
        done = {(r["model"], r["id"]): r for r in map(json.loads, out.open())}
    spent = sum(r["cost"] for r in done.values())
    lock = threading.Lock()
    client = httpx.Client(timeout=120)

    def vote(job):
        nonlocal spent
        model, row = job
        prompt, options = request_for(row, model, domain)
        answer, cost = None, 0.0
        for attempt in range(3):
            time.sleep(attempt * 5)
            with lock:
                if spent >= cap:
                    return
            try:
                body, used = call(client, model, prompt, key, meta_key)
            except (httpx.HTTPError, ValueError):
                continue
            cost += used
            with lock:
                spent += used
            message = (body.get("choices") or [{}])[0].get("message") or {}
            answer = parse(message.get("content"), options)
            if answer:
                break
        with lock:
            with out.open("a") as handle:
                handle.write(
                    json.dumps(dict(model=model, id=row["id"], answer=answer, cost=cost)) + "\n"
                )

    jobs = [(m, r) for m in panel for r in rows if (m, r["id"]) not in done]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(vote, jobs))
    return spent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--domain", choices=("mtop", "banking77"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cap", type=float, default=1.0)
    parser.add_argument("--second", action="store_true", help="the Amendment 5 panel")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.input, encoding="utf-8", newline="")))[: args.limit]
    spent = run(
        rows,
        args.out,
        args.domain,
        args.cap,
        os.environ["OPENROUTER_API_KEY"],
        panel=SECOND if args.second else PANEL,
        meta_key=os.environ.get("SECND_META_API_KEY", ""),
    )
    print(f"rows={len(rows)} spent=${spent:.4f}")


if __name__ == "__main__":
    main()
