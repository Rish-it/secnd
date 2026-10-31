"""Fixed-count, fixed-arrival-rate HTTP probe. No calls occur on import."""

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import httpx


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


async def run(url, payload, count, rate, token="", timeout=30):
    if count < 1 or rate <= 0:
        raise ValueError("Count and rate must be positive.")
    latencies, errors = [], 0
    started = time.perf_counter()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
    ) as client:

        async def send():
            nonlocal errors
            begin = time.perf_counter()
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            except httpx.HTTPError:
                errors += 1
            finally:
                latencies.append((time.perf_counter() - begin) * 1000)

        tasks = []
        for i in range(count):
            await asyncio.sleep(max(0, started + i / rate - time.perf_counter()))
            tasks.append(asyncio.create_task(send()))
        await asyncio.gather(*tasks)
    return {
        "requests": count,
        "rate_per_second": rate,
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "error_count": errors,
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="SECND_READER_BEARER_TOKEN")
    args = parser.parse_args()
    report = asyncio.run(
        run(
            args.url,
            json.loads(args.request.read_text()),
            args.count,
            args.rate,
            os.environ.get(args.token_env, ""),
        )
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
