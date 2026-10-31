import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { createHmac } from "node:crypto";

test("connector source groups appear in the decision inbox", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Your decision inbox." }),
  ).toBeVisible();
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
  const pending = page.getByRole("tabpanel", { name: /^Needs you/ });
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
