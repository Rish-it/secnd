"""Keep callback verification credentials out of the HTTP access log."""

import logging
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class RedactQuerySecrets(logging.Filter):
    def filter(self, record):
        if isinstance(record.args, tuple) and len(record.args) == 5:
            args = list(record.args)
            if isinstance(args[2], str):
                url = urlsplit(args[2])
                query = [
                    (
                        key,
                        "[redacted]"
                        if key in {"hub.verify_token", "access_token", "token"}
                        else value,
                    )
                    for key, value in parse_qsl(url.query, keep_blank_values=True)
                ]
                args[2] = urlunsplit(
                    (url.scheme, url.netloc, url.path, urlencode(query), url.fragment)
                )
                record.args = tuple(args)
        return True


def redact_access_logs():
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, RedactQuerySecrets) for item in logger.filters):
        logger.addFilter(RedactQuerySecrets())
