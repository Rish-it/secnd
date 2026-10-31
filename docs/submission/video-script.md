# Secnd — three-minute product video

Record the running local product. Keep the synthetic banner visible. Start a fresh
`npm run demo --workspace @secnd/frontend`, then open
`http://127.0.0.1:8000/?setup=1` at a desktop viewport. No real sign-in, account,
message, ad change or provider call is used. All numerical business examples are
synthetic story props described in `README.md`, not performance measurements.

| Time | Exact on-screen actions | Voiceover |
| --- | --- | --- |
| 0:00–0:15 | Show the setup steps and “Meet your second.” Click **Sign in to synthetic business**. | “Running a small business means switching between ads, messages and decisions all day. Secnd runs your Meta business with you. Meet Ren: a second pair of eyes that learns your standards from your past decisions.” |
| 0:15–0:35 | Click **Connect synthetic Meta**. Select the four synthetic accounts. Click **Use selected accounts**. Select **95%**, click **Save my bar**, then **Open my inbox**. | “Sign in, connect Meta, choose your accounts, and set your bar. This is a clearly labelled synthetic walkthrough. No live accounts are connected. You decide how much evidence Ren needs before it earns permission to act.” |
| 0:35–0:58 | In **Needs you**, show the WhatsApp icon and the studio-hours question. Open **Why · nearest past decisions** on the first card. Point to the saved examples. Click its **Confirm Share opening hours** button. | “Here is the decision inbox. Ren brings the situation, a proposed action, and how sure it is. Why opens the nearest decisions you made before. One tap confirms. You spend your attention on a choice, instead of hunting through another app.” |
| 0:58–1:18 | Click **Undo last answer**. On the same card, click **Ask a person** in its other actions. Open **Corrected** and show the original suggestion and correction. Return to **Needs you**, press **j**, then **y**, then **u**. | “You can correct it just as quickly, and undo a review. Your answer becomes part of the business’s history. The keyboard works too. Ren is the decision model; Muse never makes the decision. The synthetic demo uses scripted suggestions so this story can run entirely offline.” |
| 1:18–1:43 | Open **Autonomy**. Show **Ads**, its **Act** selection, exact counts and progress. Show **Inbox**, its progress and remaining confirmations. Show **Account** with current level **ask** and its correction in **Permission & correction history**. | “Permission is earned separately. Ads has reached this business’s chosen bar. Inbox is still building evidence. A correction has lowered Account’s bound, so it returns to asking. Watch, Ask and Act remain your controls. Sensitive actions still need separate permission, and account changes are not performed in this version.” |
| 1:43–2:03 | Open **Inbox → Done by Ren**. Show the proof paragraph on the first card. Expand **Why**. | “Every automatic decision carries its basis in plain words. This one uses reviews from this synthetic business. An audited public-history certificate is a separate source of evidence. Secnd keeps those two distinct; a benchmark does not become a claim about your business.” |
| 2:03–2:28 | Open **Activity**. Scroll to **What happened after**. Show the linen tote ad’s **Before** and **After 7 days** spend, cost per result and click-through. Scroll back to **Agent log**. | “The story continues after a decision. Activity shows what Ren did and the ad results observed a week later. These are synthetic observations, not proof that a decision caused a change. Your review of an outcome is what becomes a new example.” |
| 2:28–2:46 | Scroll to **Stop Ren**. Click **Stop all automatic actions**. Show the inline confirmation. Click **Confirm stop** and show the stopped state. | “And there is an obvious stop. Confirm once, and automatic actions stop across the business. Already submitted requests can finish. Restarting needs an operator reset. You stay in charge.” |
| 2:46–3:00 | Open **About**. Show **synthetic-scripted-demo** and the disclosure that the 400M model is not running here. End on the footer with both required credits. | “Ren is our decision model, trained once and reading your history as examples. About tells you what is actually running. Secnd is built for Meta’s platform, with Muse Spark Open. Your standards, with a little more headspace.” |

Ren's 400M architecture is documented in `docs/results/s3r/README.md`; the demo's
active model identifier and limitations are documented in `README.md`. The video
makes no model-quality, speed, customer-impact or live-integration claims.

If a measured-result caption is required, use only:
“Fixed MTOP public-history evaluation: 21.04% coverage, 1 error in 923; 99.40% lower confidence
bound. Eight examples, full fixed history, model-audited calibration.”
Source filename on screen: `docs/results/s3r/floor-full-test-400m-fine-seed1.json`.
Keep the limitations on screen too: unchanged test labels, no new-business
certification, and 2 wrong acceptances per 500 in the separate missing-intent audit
(`docs/results/s3r/README.md`). Do not put this caption beside synthetic business
review counts.

Final video URL: **TBD**. Measured end-to-end 400M latency for this product build:
**TBD**. Live OAuth: verified, see `verification.md`. Do not substitute estimates.
