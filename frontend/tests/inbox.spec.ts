import { test, expect } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const folder = "docs/results/c1-inbox";
const api = "/api/v1/workspaces/default";

test("inbox decisions, evidence, undo, source groups, keyboard, pagination and workspace isolation", async ({
  page,
  request,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Your decision inbox." }),
  ).toBeVisible();
  const pending = page.getByRole("tabpanel", { name: /^Needs you/ });
  await expect(pending.getByRole("article")).toHaveCount(4);
  for (const name of ["WhatsApp", "Messenger", "Instagram", "Ads"]) {
    await expect(
      pending.getByRole("region", { name: `${name} decisions` }),
    ).toBeVisible();
  }
  mkdirSync(folder, { recursive: true });
  await page.screenshot({ path: `${folder}/desktop.png`, fullPage: true });
  const original = pending.getByRole("article").first();
  const confirm = original.getByRole("button", { name: /^Confirm / });
  await confirm.click();
  await expect(pending.getByRole("article")).toHaveCount(3);
  await expect(
    page.getByRole("status").filter({ hasText: /Saved .* to Ren/ }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Undo last answer" }).click();
  await expect(pending.getByRole("article")).toHaveCount(4);
  await original.getByRole("button", { name: "Pick another label" }).click();
  await original.getByRole("combobox").selectOption("Escalate");
  // The synthetic ambiguous input predicts Reply or Archive, never Escalate for this seed.
  const modelLabel = (await request.get(api + "/queue").then((r) => r.json()))
    .items[0].label;
  if (modelLabel === "Escalate")
    await original.getByRole("combobox").selectOption("Archive");
  await original.getByRole("button", { name: "Save correction" }).click();
  await page.getByRole("tab", { name: /^Corrected/ }).click();
  const overridden = page.getByRole("tabpanel", { name: /^Corrected/ });
  await expect(overridden.getByRole("article")).toHaveCount(1);
  await expect(
    overridden.getByText(/Your correction is in its history/),
  ).toBeVisible();
  await page.screenshot({ path: `${folder}/overridden.png`, fullPage: true });
  await page.getByRole("button", { name: "Undo last answer" }).click();
  await expect(overridden.getByRole("article")).toHaveCount(0);
  const pendingTab = page.getByRole("tab", { name: /^Needs you/ });
  await pendingTab.focus();
  await page.keyboard.press("ArrowRight");
  const automaticTab = page.getByRole("tab", {
    name: /^Done by Ren/,
  });
  await expect(automaticTab).toBeFocused();
  await expect(automaticTab).toHaveAttribute("aria-selected", "true");
  const automatic = page.getByRole("tabpanel", {
    name: /^Done by Ren/,
  });
  await expect(automatic.getByRole("article")).toHaveCount(4);
  await expect(
    automatic.getByText("Precision lower bound", { exact: true }),
  ).toHaveCount(4);
  const evidence = automatic.getByRole("article").first();
  await evidence.getByText("Evidence · nearest past decisions").click();
  await expect(evidence.getByRole("listitem")).toHaveCount(3);
  await page.screenshot({
    path: `${folder}/automatic-evidence.png`,
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: `${folder}/mobile-390.png`, fullPage: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  await pendingTab.click();
  await expect(pending.getByRole("article")).toHaveCount(4);
  await page.screenshot({
    path: `${folder}/mobile-pending-390.png`,
    fullPage: true,
  });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );

  // More than one page, followed by a durable answer and refresh. No duplicates or lost cursor.
  const additional = Array.from({ length: 23 }, (_, i) => ({
    item_id: `pagination-${i}`,
    text: "Hello, a question",
    source_app: "instagram",
  }));
  expect(
    (
      await request.post(api + "/predict", { data: { items: additional } })
    ).ok(),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Refresh inbox" }).click();
  await expect(pending.getByRole("article")).toHaveCount(20);
  await page.getByRole("button", { name: "Load more decisions" }).click();
  await expect(pending.getByRole("article")).toHaveCount(27);
  await page.getByRole("textbox", { name: "Workspace", exact: true }).fill("other");
  await page.getByRole("button", { name: "Open workspace" }).click();
  await expect(page).toHaveURL(/workspace=other/);
  await expect(
    page.getByRole("tabpanel", { name: /^Needs you/ }).getByRole("article"),
  ).toHaveCount(1);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  const metrics = await request.get(api + "/metrics").then((r) => r.json());
  writeFileSync(
    `${folder}/browser.json`,
    JSON.stringify(
      {
        scenario:
          "Synthetic messages; real char_tfidf CPU inference and fixed synthetic calibration. No providers or live Meta calls.",
        viewports: [
          { width: 1440, height: 1100 },
          { width: 390, height: 844, document_width: 390 },
        ],
        pending: 4,
        automatic: 4,
        nearest_examples: 3,
        pagination_items: 27,
        checked: [
          "confirm",
          "correct",
          "undo confirm",
          "undo correction",
          "override view",
          "source groups",
          "tab arrow keys",
          "evidence",
          "pagination",
          "workspace isolation",
          "no horizontal overflow",
        ],
        synthetic_workspace_card: metrics.card,
      },
      null,
      2,
    ) + "\n",
  );
});
