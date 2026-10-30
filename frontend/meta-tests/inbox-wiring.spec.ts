import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { createHmac } from "node:crypto";

test("connector source groups appear in the decision inbox when it is available", async ({
  page,
  request,
}) => {
  await page.goto("/");
  const heading = page.getByRole("heading", { name: "Your decision inbox." });
  test.skip(
    (await heading.count()) === 0,
    "Decision inbox ships separately; run both app changes to verify source groups.",
  );
  const api = "/api/v1/workspaces/default";
  for (const platform of ["whatsapp", "messenger", "instagram"]) {
    const bytes = readFileSync(
      `backend/tests/fixtures/meta/${platform}-text.json`,
    );
    const signature = createHmac("sha256", "demo-only")
      .update(bytes)
      .digest("hex");
    expect(
      (
        await request.post(api + "/integrations/meta/webhook", {
          data: bytes,
          headers: {
            "Content-Type": "application/json",
            "X-Hub-Signature-256": `sha256=${signature}`,
          },
        })
      ).ok(),
    ).toBeTruthy();
  }
  const date = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  expect(
    (
      await request.post(api + "/integrations/ads/insights:sync", {
        data: { level: "adset", since: date, until: date },
      })
    ).ok(),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Refresh inbox" }).click();
  const pending = page.getByRole("tabpanel", { name: /^Pending/ });
  for (const source of ["WhatsApp", "Messenger", "Instagram", "Ads"]) {
    await expect(
      pending.getByRole("region", { name: `${source} decisions` }),
    ).toBeVisible();
  }
  await expect(pending.getByRole("article")).toHaveCount(4);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
});
