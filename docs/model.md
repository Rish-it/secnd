# The Secnd decision model

## Why

Filled in as each part ships.

## Where

Secnd learns one organisation's past decisions and repeats them only where it can prove it is right.
The first decision it learns is which cost type an expense claim belongs to.

**Main dataset: UK MP business costs.** Every claim the Independent Parliamentary Standards Authority
published for 2022–23 to 2024–25: 283,682 claims, 62 cost types, with the claim date, the amount, the
free-text details and the MP office that filed it. Real decisions by a real organisation, under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Classes are imbalanced: the largest has 55,690 claims, the smallest a handful.

**Two more sets** check that results are not specific to one source:

- 68,000 synthetic US bank-statement descriptions in 17 spending categories (MIT).
- BANKING77: 13,083 customer-support questions in 77 intents (CC-BY-4.0).

`uv run python -m backend.data` downloads all three, checks each file's SHA-256 and writes the split
manifests.

**How the data is split** ([manifests](results/s1/)):

- **Test** is the newest 15% of claims (2 Jan 2025 to 7 May 2026). Nothing after the training period is
  ever trained on.
- **Calibration** is the 15% of earlier claims just before test, so the confidence threshold is set on
  data that looks like what comes next.
- **Seed** is 200 labelled claims from the oldest data, the starting point for learning.
- Undated sets split by payee instead, so no payee appears in more than one part.

**What leaks and what doesn't** ([audit](results/s1/leakage-ipsa.json)):

- The claim's "short description" is picked from a list tied to the cost type and predicts it 97.8% of
  the time. It is never used as an input, and neither is the parent category.
- 75% of test claims come from a payee seen in training, but remembering a payee's last cost type is
  right only 64% of the time on those claims: the same payee is filed under different cost types.
  Every result is reported separately for payees seen before and payees never seen.

**Starting point** ([eval cards](results/s1/)): with 200 labelled claims, a character n-gram text
classifier gets 37.5% of test claims right (88.6% on known payees, 21.7% on new ones). With 5,000 it
gets 44.7%. None of the baselines can yet answer a useful share of claims at 99% precision.

## How

Filled in as each part ships.

## Numbers

Filled in as each part ships.

## Limits

Filled in as each part ships.
