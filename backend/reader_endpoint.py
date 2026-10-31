"""Validated wire protocol shared by the GPU endpoint and its application client."""

import hmac
from typing import Literal

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.schemas import Identifier, Label, Probability


class Example(BaseModel):
    id: Identifier
    text: str = Field(max_length=16000)
    label: Label


class ReaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: Identifier
    text: str = Field(max_length=16000)
    decision_type: Literal["ad", "inbox", "account"]
    keys: list[Label] = Field(min_length=2, max_length=128)
    meanings: list[str] = Field(min_length=2, max_length=128)
    history: list[Example] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def options_match(self):
        if len(set(self.keys)) != len(self.keys) or len(self.keys) != len(self.meanings):
            raise ValueError("Supply one meaning per distinct key.")
        if any(not 1 <= len(m) <= 1000 for m in self.meanings):
            raise ValueError("Keep option meanings short.")
        if any(row.label not in self.keys for row in self.history):
            raise ValueError("History labels must be decision options.")
        return self

    def decision(self):
        from backend.typed import Decision

        return Decision(
            case=self.item_id,
            kind="choice",
            text=f"choice Meta {self.decision_type}: Choose the person's preferred action.",
            keys=self.keys,
            meanings=self.meanings,
            state=self.text,
            gold=np.zeros(len(self.keys)),
            label=0,
        )


class ReaderReply(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=200)
    probabilities: dict[Label, Probability] = Field(min_length=2, max_length=128)


def infer(reader, body):
    rows = reader.predict(
        body.keys,
        [r.model_dump() for r in body.history],
        pd.DataFrame([dict(id=body.item_id, text=body.text)]),
        decision=body.decision(),
        allow_auto=False,
    )
    return ReaderReply(
        model_id=reader.options.model_id,
        revision=reader.artifact.revision,
        probabilities=rows[0]["probabilities"],
    )


def create_reader_api(reader, token):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/", response_model=ReaderReply)
    def decide(body: ReaderRequest, authorization: str = Header(default="")):
        if not token or not hmac.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Reader authentication required.")
        return infer(reader, body)

    return app


class RemoteReader:
    def __init__(self, settings, transport=None):
        from types import SimpleNamespace

        self.settings, self.transport = settings, transport
        self.options = SimpleNamespace(model_id=settings.meta_decision_model_id)
        self.artifact = SimpleNamespace(revision=settings.reader_endpoint_revision or "unresolved")

    def predict(self, labels, history, frame, *, decision, allow_auto=False):
        results = []
        for row in frame.itertuples():
            body = ReaderRequest(
                item_id=str(row.id),
                text=row.text,
                decision_type="ad"
                if "Meta ad:" in decision.text
                else "inbox"
                if "Meta inbox:" in decision.text
                else "account",
                keys=labels,
                meanings=decision.meanings,
                history=history[-1000:],
            )
            error = None
            try:
                token = self.settings.reader_endpoint_token.get_secret_value()
                if not token:
                    raise ValueError("Missing reader token")
                with httpx.Client(
                    transport=self.transport,
                    follow_redirects=False,
                    timeout=self.settings.reader_endpoint_timeout_seconds,
                ) as client:
                    response = client.post(
                        self.settings.reader_endpoint_url,
                        headers={"Authorization": f"Bearer {token}"},
                        json=body.model_dump(),
                    )
                    response.raise_for_status()
                    reply = ReaderReply.model_validate(response.json())
                if (
                    reply.model_id != self.options.model_id
                    or set(reply.probabilities) != set(labels)
                    or abs(sum(reply.probabilities.values()) - 1) > 1e-6
                    or (
                        self.settings.reader_endpoint_revision
                        and reply.revision != self.settings.reader_endpoint_revision
                    )
                ):
                    raise ValueError(
                        "Reader response does not match the requested model and options"
                    )
                self.artifact.revision = reply.revision
                probabilities = reply.probabilities
            except (httpx.HTTPError, ValueError, TypeError):
                error = "Reader unavailable. Please choose an answer."
                probabilities = {label: 1 / len(labels) for label in labels}
            label = max(probabilities, key=probabilities.get)
            results.append(
                dict(
                    item_id=str(row.id),
                    label=label,
                    probabilities=probabilities,
                    prob=probabilities[label],
                    auto=False,
                    spot_check=False,
                    threshold=None,
                    model_version="meta-reader-" + self.artifact.revision[:32],
                    model_id=self.options.model_id,
                    model_revision=self.artifact.revision,
                    reasons={},
                    inference_error=error,
                )
            )
        return results
