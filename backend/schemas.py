"""Stable request and response types for the workspace decision API."""

from datetime import date as Date
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]
WorkspaceId = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]
Label = Annotated[str, Field(min_length=1, max_length=128)]


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: Identifier
    text: Annotated[str, Field(min_length=1, max_length=10000)]
    amount: Annotated[float, Field(allow_inf_nan=False)] | None = None
    date: Date | None = None


class Neighbour(BaseModel):
    item_id: Identifier
    label: Label
    similarity: Probability
    text: str


class Reasons(BaseModel):
    neighbours: list[Neighbour] = Field(default_factory=list)
    merchant_memory: bool = False


class ReaderDecision(BaseModel):
    model_id: str
    revision: str
    device: str
    examples: int
    support: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None
    support_floor: Probability | None
    temperature: NonNegative | None
    policy_complete: bool
    scope_authorized: bool


class ReaderBenchmark(BaseModel):
    dataset: str
    history_rows: int
    accuracy: Probability | None
    with_floor_coverage: Probability | None
    with_floor_precision: Probability | None
    with_floor_sample: int
    with_floor_accepted: int | None
    with_floor_errors: int | None
    with_floor_low95: Probability | None
    calibration_audit: Literal["model"]
    audited_calibration_labels: int | None
    evaluation_file: str
    floor_result_file: str
    policy_file: str


class ServingStatus(BaseModel):
    backend: Literal["head", "reader"]
    model_id: str
    device: str
    revision: str | None = None
    examples: int | None = None
    history_rows: int | None = None
    temperature: NonNegative | None = None
    threshold: Probability | None = None
    support_floor: Probability | None = None
    policy_complete: bool | None = None
    scope_matches: bool | None = None
    guarded: bool = False
    scope_note: str
    benchmark: ReaderBenchmark | None = None


class Decision(BaseModel):
    item_id: Identifier
    label: Label
    probabilities: dict[Label, Probability] = Field(min_length=1)
    prob: Probability
    auto: bool
    threshold: Probability | None
    model_version: Identifier
    reasons: Reasons
    item: Item | None = None
    spot_check: bool = False
    reader: ReaderDecision | None = None

    @model_validator(mode="after")
    def check_probabilities(self):
        if self.label not in self.probabilities:
            raise ValueError("The predicted label must appear in probabilities.")
        if abs(self.probabilities[self.label] - self.prob) > 1e-9:
            raise ValueError("prob must match the predicted label's probability.")
        if self.prob < max(self.probabilities.values()):
            raise ValueError("The predicted label must have the highest probability.")
        if sum(self.probabilities.values()) > 1 + 1e-9:
            raise ValueError("Top-k probabilities must sum to at most one.")
        if self.auto and (self.threshold is None or self.prob < self.threshold):
            raise ValueError("Automatic decisions require a threshold and enough confidence.")
        if self.auto and self.reader is not None:
            r = self.reader
            if (
                not r.policy_complete
                or not r.scope_authorized
                or r.support is None
                or r.support_floor is None
                or r.support < r.support_floor
            ):
                raise ValueError("Automatic reader decisions require authorized support evidence.")
        return self


class Latency(BaseModel):
    p50: NonNegative
    p99: NonNegative


class SliceMetrics(BaseModel):
    count: Annotated[int, Field(ge=0)]
    accuracy: Probability | None = None
    coverage: Probability | None = None
    precision: Probability | None = None


class EvalCard(BaseModel):
    """Null metrics mean unmeasured; zero coverage means no certified automatic traffic."""

    model_version: Identifier
    accuracy: Probability | None = None
    macro_f1: Probability | None = None
    nll: NonNegative | None = None
    brier: NonNegative | None = None
    ece: Probability | None = None
    classwise_ece: dict[str, Probability] = Field(default_factory=dict)
    aurc: Probability | None = None
    threshold: Probability | None = None
    coverage: Probability
    precision: Probability | None = None
    cp_lower_bound: Probability | None = None
    robustness_pass_rate: Probability | None = None
    seen_merchant: SliceMetrics | None = None
    unseen_merchant: SliceMetrics | None = None
    worst_classes: dict[str, SliceMetrics] = Field(default_factory=dict)
    featurize_ms: Latency | None = None
    predict_ms: Latency | None = None
    retrain_ms: NonNegative | None = None
    peak_rss_bytes: Annotated[int, Field(ge=0)] | None = None
    artifact_bytes: Annotated[int, Field(ge=0)] | None = None
    image_bytes: Annotated[int, Field(ge=0)] | None = None
    cost_per_1000_usd: NonNegative | None = None
    data_hash: str | None = None
    split_hash: str | None = None
    featurizer_id: str | None = None
    seed: int | None = None
    git_sha: str | None = None


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: Identifier
    label: Label


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


class Error(BaseModel):
    code: str
    message: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Item] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def unique_item_ids(self):
        if len({item.item_id for item in self.items}) != len(self.items):
            raise ValueError("Each item_id in a batch must be unique.")
        return self


class ImportResult(BaseModel):
    workspace_id: WorkspaceId
    row_count: Annotated[int, Field(ge=0)]
    target_column: str
    card: EvalCard


class AnswerResult(BaseModel):
    item_id: Identifier
    coverage: Probability
    model_version: Identifier
    retrain: Literal["queued", "fixture", "history_updated"]


class ColdStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[Label] = Field(min_length=2, max_length=128)
    paid: Literal[True]

    @model_validator(mode="after")
    def distinct_labels(self):
        self.labels = [label.strip() for label in self.labels]
        if any(not label for label in self.labels) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Use at least two distinct, nonempty label names.")
        return self


class InventedExample(BaseModel):
    id: Identifier
    label: Label
    text: Annotated[str, Field(min_length=1, max_length=500)]


class ExampleEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Label
    text: Annotated[str, Field(min_length=1, max_length=500)]


class ColdStartState(BaseModel):
    workspace_id: WorkspaceId
    labels: list[Label]
    status: Literal["queued", "generating", "ready", "stopped"]
    error: str | None
    history: list[InventedExample]
    spent_usd: str
    cap_usd: str
    demo: bool


class HistoryPoint(BaseModel):
    timestamp: datetime
    model_version: Identifier
    coverage: Probability
    precision: Probability | None = None


class ReliabilityBin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence: Probability
    accuracy: Probability
    count: Annotated[int, Field(ge=0)]


class CoveragePrecisionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: Probability
    coverage: Probability
    precision: Probability | None


class EvalCurves(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reliability: list[ReliabilityBin] = Field(default_factory=list)
    coverage_precision: list[CoveragePrecisionPoint] = Field(default_factory=list)


class MetricsResult(BaseModel):
    dataset: Literal["mtop", "banking77", "ipsa"] | None = None
    curves: EvalCurves = Field(default_factory=EvalCurves)
    card: EvalCard
    history: list[HistoryPoint]
    guarded: bool = False
    labels: Annotated[int, Field(ge=0)] = 0


class ModelVersion(BaseModel):
    model_version: Identifier
    current: bool
    card: EvalCard


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, Field(min_length=1, max_length=4000)]


class ChatResult(BaseModel):
    answer: str
    model_version: Identifier | None = None
