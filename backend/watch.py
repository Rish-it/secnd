"""Watch a CPU learning simulation: `uv run python -m backend.watch --dataset mtop`."""

import argparse
import re
import sys
from collections import deque

from backend import sim
from backend.config import Settings
from backend.encoders import fetch


def text(value):
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(value))


def percent(value):
    return "unmeasured" if value is None else f"{value:.2%}"


class LiveView:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.row, self.result, self.loop, self.original = {}, {}, None, None
        self.feed = deque(maxlen=6)

    def __call__(self, event):
        loop = event.get("loop")
        if loop is not None and loop is not self.loop:
            self.close()
            self.loop = loop
            if callable(getattr(loop, "retrain", None)):
                self.original = loop.retrain

                def observed(*args, **kwargs):
                    result = self.original(*args, **kwargs)
                    self.result = result
                    return result

                # Curve rows omit reasons; observe return values without changing training.
                loop.retrain = observed
        kind = event.get("kind")
        if kind == "round":
            self.row = event["row"]
            self.result = event.get("result", self.result)
        elif kind == "answer":
            item = event["item"].iloc[0]
            decision = event["decision"]
            source = "spot-check" if decision.get("spot_check") else "answer"
            self.feed.append(
                f"{source}: {text(item.get('text', ''))[:55]} -> {text(item['label'])} "
                f"(predicted {text(decision['label'])})"
            )
        elif kind == "auto":
            if event["decisions"]:
                self.feed.append(f"Automatic: {len(event['decisions'])} items")
        else:
            return
        if self.row and (kind == "round" or self.stream.isatty()):
            self.render()

    def render(self):
        row = self.row
        guarded = bool(getattr(self.loop, "guarded", row.get("guarded", False)))
        coverage = 0.0 if guarded else row.get("coverage", 0)
        count = round(20 * coverage)
        bar = "█" * count + "░" * (20 - count)
        precision = row.get("precision")
        target = (
            "not certified"
            if precision is None
            else ("meets 99% target" if precision >= 0.99 else "BELOW 99% target")
        )
        promoted = row.get("promoted")
        status = "Promoted" if promoted else "Refused"
        reasons = self.result.get("reasons", row.get("reasons", []))
        if not promoted:
            status += ": " + ("; ".join(map(text, reasons)) or "reason unavailable")
        labels = len(self.loop.y) if hasattr(self.loop, "y") else row.get("labels", 0)
        lines = [
            f"Ren | round {row.get('round', 0)} | {text(row.get('model_version', 'untrained'))}",
            f"Coverage [{bar}] {percent(coverage)}",
            f"Test precision (last scored): {percent(precision)} | {target}",
            f"Accuracy: {percent(row.get('accuracy'))} | Labels: {labels}",
            f"{status} | guarded: {str(guarded).lower()}",
        ]
        if guarded:
            lines.append("Automatic answers paused; all new items go to a person.")
        lines.extend(["Recent answers:", *self.feed])
        if self.stream.isatty():
            self.stream.write("\x1b[H\x1b[2J")
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()

    def close(self):
        if self.loop is not None and self.original is not None:
            self.loop.retrain = self.original
        self.loop, self.original = None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mtop")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", choices=sim.STRATEGIES, default="random")
    parser.add_argument("--featurizer", default=Settings().featurizer)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.batch < 1:
        parser.error("rounds and batch must be positive")
    if args.featurizer == "pt-minilm":
        fetch(args.featurizer, Settings().data_dir)
    view = LiveView()
    try:
        sim.simulate(
            args.dataset,
            args.rounds,
            args.batch,
            args.seed,
            args.strategy,
            on_step=view,
            featurizer=args.featurizer,
        )
    finally:
        view.close()


if __name__ == "__main__":
    main()
