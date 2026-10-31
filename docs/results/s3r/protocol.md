# S3R preregistration

Registered 2026-10-30 before attack inference. Baseline: merge c543ca4 (#29).
Checkpoint: ic-ettin-150m-s6000-near. No training or threshold tuning.

Use the official English MTOP history and test, and the 1,203 clean calibration
rows obtained by the existing evaluate() training-overlap filter. Seed 0 splits
calibration into temperature-fit and certification halves. Recompute the existing
choose_threshold rule only on calibration, freeze temperature and threshold, and
apply them unchanged to every attack. Evaluate 0, 8-nearest and 32-nearest with
full history. Save split, checkpoint and protocol hashes with results.

Primary sample: first 500 test rows in a seed-20261030 permutation, chosen before
inference. Each variant retains the source row ID; variants are paired, not
independent evidence. Report baseline alongside each attack:

- Paraphrase: prefix each message with “Please help me with this request: ”.
  This tests a narrow, meaning-preserving wrapper, not broad semantic paraphrases.
- Typo: transpose the first two internal letters of the first alphabetic word
  of length at least five; unchanged rows are counted and reported separately.
- Option order: independent seed-20261030 permutation per source row; map output
  slots back to label IDs before scoring.
- Duplicates: exact repeat and a lowercase/whitespace-normalized variant. Report
  prediction and acceptance flips per original, never inflate sample size by
  treating repeated texts as independent observations. Retrieve from history only.
- Missing intent: remove the gold intent from options and remove history examples
  of that intent. Every accepted answer is an error; report false-accept rate.

Re-encode changed texts for retrieval. Do not add test variants to history.
Record accepted count, errors, coverage, accepted precision (null if none), exact
binomial 95% interval, and difference from 99%. Intervals over unique source rows
are descriptive; do not pool correlated attacks. Zero acceptance establishes no
precision claim. Report every condition, including failures and truncation counts.

CPU latency: same checkpoint, FP32 eval/inference mode, batch size one, CPU only,
113 options, 0/8/32 examples, one process, four torch threads. Twenty warmups then
100 sequential distinct test requests per condition. Measure tokenization,
packing, forward and softmax together; precomputed nearest examples are excluded
and that exclusion must accompany results. Record p50/p95, raw durations, token
lengths, process peak RSS, model-load time, CPU/OS/torch versions. No GPU latency
may substitute for CPU measurements. No shipping decision without a latency SLA.

The committed #29 card currently certifies zero rows in all these conditions.
The handoff's 23–37% figure has no supporting artifact in this checkout. If a
newer gate is supplied, register its identity and any protocol amendment before
running it; do not silently replace this baseline.

## Amendment 1 (before inference)

The owner supplied pushed commit e2581ee on stage/s2f-mtop-calib. Use that
held_out() implementation instead: seed-0 4,000-row MTOP calibration from official
train, leaving 11,667 history rows; certification has 2,000 rows. All reader
training excluded these MTOP train texts. The earlier 602-row result is superseded.
Keep rule A unchanged, fit temperature and threshold only on the new calibration
split. Full-history 0/8/32 conditions remain the registered primary audit.

## Implementation clarification (before attack results)

Attack inference uses the same CUDA BF16 autocast and calibration batch size 32
as the supplied evaluation; CPU measurements use FP32. An initial GPU process
was interrupted before any attack outcome was observed to enforce this match.
No thresholds are adjusted to attack outcomes. The wrapper paraphrase is limited;
a separately registered set of assistant-authored test-message rewrites will supplement
it. Max-length hits are reported as an upper bound on actual truncations.

## Supplement: natural paraphrases (registered before their inference)

Evaluate the first 50 primary source rows using the frozen rewrites in
`paraphrases.json`, SHA-256 `acf30e35d78c2b742674f994ef6065cecdad539db39762403587005232d7424e`.
They were written from the source texts without inspecting per-source predictions or
gold intent labels. Preserve entities, requested actions and source ambiguity.
No post-inference edits or exclusions. Report this small, assistant-authored sample
separately from the primary 500-row attack suite. Re-run paired original messages
and re-encode changed messages for nearest-history retrieval. The same three
full-history shot conditions and unchanged calibration rule apply.

## CPU comparison extension (before latency results)

Also run `ic-ettin-32m-s6000-near` on the identical CPU allocation and request
sample with the same warmups, FP32, four threads and three shot counts. Report
these as separate processes, including CPU identity; cloud scheduling can assign
different hosts. This supplies the requested smaller-model alternative without
using GPU timing to infer CPU speed.

## Authorship clarification

The natural rewrites were written by an AI coding assistant, not by a human annotator or an
independent reviewer. This wording correction changes no source IDs, rewrites,
thresholds, exclusions or measurements. The frozen manifest hash above is unchanged.

The supplement was registered after the primary suite had started, when aggregate
zero-shot counts were visible. Per-source predictions and gold labels were not
inspected when authoring rewrites. The primary protocol preceded the primary run;
the supplement preceded its own run. This clarification changes no experiments.

## Amendment 2: support floor for missing intents (registered before its inference)

The frozen audit accepted 4/9/9 wrong answers per 500 when the gold intent was absent.
Add one rule on top of the unchanged gate. Support is the highest MiniLM cosine
similarity between the message and any history row carrying the predicted intent.
An answer is accepted only when confidence clears the frozen threshold and support
clears the floor.

The floor is fixed per shot count on calibration only, never on the attack: on the
2,000 certification rows, remove each row's gold intent from the options and from
retrieved history, run the reader, and keep the support of every answer the frozen
gate would accept. The floor is the lowest value on a 0.01 grid at which at most
0.2% of those rows (4 of 2,000) would still be accepted. Then apply temperature,
threshold and floor unchanged to every registered attack on the same 500 test rows.
Report, beside the unchanged gate: accepted, errors and coverage with the floor, the
calibration coverage the floor costs on clean rows, and the floor itself. Target:
at most 1 wrong accept per 500 missing-intent rows at every shot count.

## Amendment 3: human audit of calibration labels (registered before any labeller result)

Reason: the frozen gate needs zero errors among its first 427 answers, and the rows that break it
for the 400M reader look like wrong labels. That is a hypothesis, so people test it.

Audit only calibration rows; test labels are never audited or changed. Rows: every confident
disagreement between a reader and its label, plus random confident rows where they agree, as
controls. Three independent labellers per row see the message and two options in random order, and
are not told which is the label. A label changes only when at least 2 of 3 pick the other option.

Agreeing rows are audited because fixing only disagreements could only flatter the model. If the
majority overturns more than 2% of controls, the audit is reported as biased and not used. All
labels the audit changes are published with their votes. The gate is then refit on calibration,
and the result is reported on both label sets, including precision against unaudited test labels,
which can fall below 99% because those labels keep their errors. Any certified claim states this.

## Amendment 4: model panel replaces people in the audit (registered before any vote)

Reason: no human labellers are available before the freeze. Amendment 3 stays as written except
that three language models replace the three people. This is a model audit, not a human audit, and
every result from it says so. Agreement between models is not proof that a label is right.

Panel: `openai/gpt-6-luna`, `google/gemini-3.5-flash-lite`, `qwen/qwen3.8-flash`, through one
routing API, with one independent request per row and no shared conversation. None of them wrote
the Meta-app tasks below, and none is the teacher used in training. Each sees the message and the
options, with option order shuffled per model, and is not told which is the label. Answers are
option A, option B, "Both fit equally" or "Neither fits". A row that still has no valid answer
after three tries counts as no vote.

The 2 of 3 rule and the 2% control rule from Amendment 3 are unchanged. Test labels are still never
audited or changed. Inputs are fixed by SHA-256:

- MTOP, 26 disagreements and 100 controls:
  `3163ac7eea02b64ec67d6b89798eb5eaee9dd2e3dadc27d9d1e98cb54351e27b`
- BANKING77, 13 disagreements and 60 controls:
  `92e62e30efb07ac334b354ed6486f5d5ddb6864c0490bbdca14d772c3441d619`

The same panel labels a blind 150-row sample of Meta-app tasks, 50 per decision type, with sources
removed: `74b28c16db32712b6837ed54d2a0feabf79895134c5d15514b53e3ed86194053`. Answers are the task's
options or "insufficient information". A row gets a label when at least 2 of 3 agree; otherwise it
has none. These tasks are written, not collected, and these labels are model labels, so no accuracy
measured on them is reported as accuracy on real data.

Spend is capped at $1, checked before every request.

### Support floors after the model audit (registered before their inference)

Thresholds change with the audited labels, so floors are refit by the Amendment 2 method, unchanged,
on the audited MTOP calibration labels, and the 500-row attack set is rerun once. Test labels stay
unchanged. Only values from this run ship with the refit gate.

## Amendment 5: second BANKING77 model audit (registered before any vote)

Reason: the BANKING77 audit under Amendment 4 overturned 2 of 60 controls, so it was reported as
biased and not used. That result stands and is reported next to anything from this audit. This is
a second, separate attempt with a different panel. If it fails its control rule, BANKING77 stays
unaudited and there is no third attempt.

Panel: `muse-wonder-1.0`, `openai/gpt-6.1-sol`, `anthropic/claude-sonnet-5.5`, one independent
request per row, each from a different lab, none used as a teacher in training. Reader:
`ic-ettin-400m-s6000-near-wide-clean`, 8 nearest examples, full history. Rows come only from
calibration: every disagreement between the reader and its label among the 600 most confident rows
of the gate's certification half, plus 100 random rows from those 600 where they agree, drawn with
seed 5. Everything else in Amendments 3 and 4 is unchanged: the same options and shuffling, the
2 of 3 rule, the 2% control rule, publishing every changed label with its votes, and reporting on
both label sets. Test labels are never audited or changed, so precision against them can fall
below 99%. Every result says it comes from a model audit. Audit outputs are never used for
training. Spend is capped at $6.

Inputs are fixed by SHA-256 in a commit made before the first vote.

### Amendment 5, reader revised (registered before any vote)

No vote had been cast. The reader named above lost to `ic-ettin-400m-s6000-near-wide-fine-seed1`, chosen
because it certified the most calibration rows at 8 nearest examples, so the audit uses that reader
instead. Nothing else changes. Its 600 most confident rows of the certification half hold 9
disagreements and the 100 controls, with the first wrong answer at rank 200 of the 427 the gate needs.
Controls offer the reader's second choice as the other option. Inputs:
`59b981d6f5fda7139edea8bb62b38bc9c54ff6a1b50779fd9fff9c3fa75e53ee`
