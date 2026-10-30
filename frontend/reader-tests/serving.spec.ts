import { expect, test } from "@playwright/test";
import { mkdir } from "node:fs/promises";

test("reader evidence, history scope and audited policies at desktop and 390px", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByRole("button", { name: /^Metrics/ }).click();
  const card = page.getByRole("region", { name: "Ren · Reader" });
  await expect(card).toBeVisible();
  await expect(card).toContainText("ic-ettin-150m-s6000-near");
  await expect(card).toContainText(
    "Human confirmation required for this history.",
  );
  await expect(card).toContainText("96.226%");
  await expect(card).toContainText("0.92");
  await expect(card).toContainText("11,667-row history");
  await expect(card).toContainText("92.9%");
  await expect(card).toContainText("10.85%");
  await expect(card).toContainText("4,386 test messages");
  await expect(card).toContainText("99.23%");
  await expect(card).toContainText("model audit, not a human audit");
  await mkdir("docs/results/c3-serving", { recursive: true });
  await page.screenshot({
    path: "docs/results/c3-serving/desktop-audited.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(card).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  await page.screenshot({
    path: "docs/results/c3-serving/mobile-audited-390.png",
    fullPage: true,
  });
  await page.goto("/?workspace=gpu");
  await page.getByRole("button", { name: /^Metrics/ }).click();
  await expect(card).toContainText("ic-ettin-400m-s6000-near");
  await expect(card).toContainText("CUDA · 0 examples");
  await expect(card).toContainText("96.552%");
  await expect(card).toContainText("14.36%");
  await expect(card).toContainText("model audit, not a human audit");
  await expect(card).toContainText(
    "Human confirmation required for this history.",
  );
  expect(errors).toEqual([]);
});
