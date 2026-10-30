# Decision inbox browser check

Run `npm ci`, `uv sync --locked`, `npx playwright install chromium`, then
`npm run test:inbox`. The browser test starts an isolated app with real CPU
`char_tfidf` inference and a fixed synthetic calibration holdout. No provider,
outbound messaging or ad API calls occur. Sources are illustrative badges;
these screenshots do not demonstrate live Meta delivery or a Meta-traffic evaluation.

[Recorded browser assertions and synthetic workspace card](browser.json).

- [Desktop, pending](desktop.png)
- [Desktop, automatic answer with saved evidence](automatic-evidence.png)
- [Desktop, correction](overridden.png)
- [390px, automatic answer](mobile-390.png)
- [390px, pending](mobile-pending-390.png)

The check confirms an answer, undoes it, saves a different label, views and undoes
the correction, navigates status tabs by keyboard, expands nearest-example evidence,
loads the next page and switches workspaces. The recorded result includes viewport
and document widths. Displayed certification belongs only to synthetic calibration;
it is not a claim about real incoming messages. Muse writes no text in this check.
