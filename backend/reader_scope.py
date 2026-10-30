"""Keep the measured full-history scope distinct from arbitrary workspace history."""

import hashlib
import json


def scope_digest(labels, history):
    # IDs and arrival order do not affect the support definition; duplicate rows do count.
    rows = sorted((row["text"], row["label"]) for row in history)
    return hashlib.sha256(json.dumps([sorted(labels), rows]).encode()).hexdigest()
