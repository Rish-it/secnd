import { expect, test } from "@playwright/test";
import { mkdir } from "node:fs/promises";

test("reader evidence, history scope and null policies at desktop and 390px", async ({
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
  await expect(card).toContainText("99.563%");
  await expect(card).toContainText("0.80");
  await expect(card).toContainText("11,667-row history");
  await expect(card).toContainText("92.9%");
  await expect(card).toContainText("22.0%");
  await mkdir("docs/results/c3-serving", { recursive: true });
  await page.screenshot({
    path: "docs/results/c3-serving/desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(card).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  await page.screenshot({
    path: "docs/results/c3-serving/mobile-390.png",
    fullPage: true,
  });
  await page.goto("/?workspace=gpu");
  await page.getByRole("button", { name: /^Metrics/ }).click();
  await expect(card).toContainText("ic-ettin-400m-s6000-near");
  await expect(card).toContainText("CUDA · 0 examples");
  await expect(card).toContainText("Unavailable");
  await expect(card).toContainText(
    "Human confirmation required for this history.",
  );
  expect(errors).toEqual([]);
});
