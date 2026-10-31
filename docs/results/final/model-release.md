# Model release preparation draft

These steps cover the classifier-head bundle. The 400M reader and Meta checkpoints are public at
tag `v1.1.0`, uploaded by `backend/modal_hf_reader.py` with [this card](../../hf/README.md).

No upload is performed by the build or by this document. The helper defaults
to a local preview; publication requires a separate approved invocation.

Prepare a dedicated directory containing exactly four reviewed files:

- `model.pkl`: the saved classifier/calibration artifact for the chosen version.
- `card.json`: an EvalCard with the same `ren-1.<n>` version and measured provenance.
- `runtime.json`: `model_id: "secnd-ren-1"`, `model_version`, `labels` in classifier
  order, `featurizer`, and `merchant_memory` mapping normalised keys to label
  indices. Include the encoder revision, preprocessing and seed needed to load it.
- `README.md`: the completed [model card](../../hf/README.md), with an approved
  release license and results for those exact weights. No DRAFT/RELEASE_TODO markers.

The registry currently stores a candidate head/calibration dictionary. Final
packaging still needs the matching ordered labels, memory and encoder identity;
this helper does not infer them or export live workspace data. Verify a fresh
CPU load predicts the same labels/probabilities as the original version before
publication. Fine-tuned encoder packaging is pending the final model choice.

Preview locally (no network or credentials required):

```sh
uv run python -m backend.release --folder .data/release/ren-1.0 \
  --repo-id YOUR_NAMESPACE/secnd-ren-1
```

It prints the model/version and SHA-256 of the four files, not their contents.
Extra files, symlinks, mismatched versions or invalid runtime labels are refused.
The preview accepts a draft README so the bundle can be reviewed first.

Only after the human approves that exact bundle, use the existing Hub login or
`HF_TOKEN` in the backend process environment and run the same command with
`--upload`. It creates a public model repository if absent and uploads only the
four allowlisted files. It refuses a draft or a model card without a license.
Do not source a provider `.env` in a shell. Keep tokens out of commands and logs.

Use the installed `huggingface_hub` client; its local API signatures and the
[official upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
were checked on Oct 29, 2026. No model files or Hub repository are published as
part of this draft.

The committed BANKING77 reference at 2,000 labels and seed 0 answers **36.9%**
automatically at **99.03%** test precision ([card](../s3/card-banking77-minilm+lr-b2000.json));
the claim holds in **98.95%** of 2,000 re-splits
([robustness run](../s3/robustness-banking77-minilm+lr-b2000.json)).
These are reference measurements, not results for an unpublished release bundle.
