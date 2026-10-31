"""Redact credentials in access logs and outgoing HTTP request URLs."""

import logging
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SECRETS = {
    "hub.verify_token",
    "access_token",
    "token",
    "code",
    "state",
    "client_secret",
    "fb_exchange_token",
    "signed_request",
    "password",
    "authorization",
}


def redact_url(value):
    url = urlsplit(value)
    query = [
        (k, "[redacted]" if k.lower() in SECRETS else v)
        for k, v in parse_qsl(url.query, keep_blank_values=True)
    ]
    return urlunsplit((url.scheme, url.netloc, url.path, urlencode(query), url.fragment))


class RedactQuerySecrets(logging.Filter):
    def filter(self, record):
        if isinstance(record.args, tuple) and len(record.args) == 5:
            args = list(record.args)
            if isinstance(args[2], str):
                args[2] = redact_url(args[2])
                record.args = tuple(args)
        # HTTPX INFO logs contain URLs. Transport DEBUG traces can contain headers/bodies.
        if record.name.startswith(("httpx", "httpcore")):
            if record.levelno < logging.INFO:
                return False
            record.msg = re.sub(
                r"https?://[^\s\"]+", lambda m: redact_url(m[0]), record.getMessage()
            )
            record.args = ()
        return True


def redact_access_logs():
    for name in (
        "uvicorn.access",
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.connection",
        "httpcore.proxy",
    ):
        logger = logging.getLogger(name)
        if not any(isinstance(item, RedactQuerySecrets) for item in logger.filters):
            logger.addFilter(RedactQuerySecrets())
