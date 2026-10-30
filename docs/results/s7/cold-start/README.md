# Label-only onboarding browser check

The screenshots use simulated Muse Wonder transport, clearly marked **Demo examples · no
provider charge**, and the real `ic-ettin-150m-s6000-near` reader and MiniLM retrieval on CPU.
They show illustrative messages, not an accuracy evaluation or real Meta API delivery.

- [Desktop, 1440px](desktop.jpg)
- [Mobile, 390px](mobile-390.jpg)

Measured browser flow: two labels produced 20 simulated starting messages; one edit and one
deletion left 19 examples. A real reader prediction suggested Travel for an illustrative train
request. It remained a question for a person, with a null threshold. Confirming the answer
added a twentieth history row. Neither predicting nor answering called the teacher transport.
The 390px viewport had a 390px document width, with no horizontal overflow.

Reproduce after installing the reader checkpoint as described in the root README:

```sh
npm run build
uv run python -m backend.tests.cold_start_demo --port 8028
```

Open http://127.0.0.1:8028, enter Software and Travel on separate lines and generate examples.
Edit/save an example, delete another, ask Ren about a train ticket and confirm or correct its
suggestion. Repeat at desktop and 390px. The helper uses a separate demo database, an isolated
simulated ledger, and a mocked transport that cannot make a paid model call.

Automated coverage uses mocked transport: canonical/shared cache hits, durable corrections,
restart, attempted-request protection, schema/import validation, model availability before
spend, and a partial-history run stopped by the cumulative $1 cap. Public spend ledgers are
untouched by the checks.
