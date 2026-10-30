import { test, expect } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHmac } from "node:crypto";

const folder = "docs/results/c2-meta-apps";
const api = "/api/v1/workspaces/default";

test("mocked connectors, typed Ads proposal, opt-in, default-off execution, and responsive UI", async ({
  page,
  request,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto("/");
  await page.getByRole("button", { name: /^Teach/ }).click();
  const connections = page.getByRole("region", {
    name: "Connect your decisions.",
  });
  await expect(connections.getByText("Ready for signed messages")).toHaveCount(
    3,
  );
  await expect(
    connections.getByText("Execution switch off", { exact: true }),
  ).toBeVisible();
  const toggle = connections.getByRole("checkbox");
  await expect(toggle).not.toBeChecked();
  for (const platform of ["whatsapp", "messenger", "instagram"]) {
    const bytes = readFileSync(
      `backend/tests/fixtures/meta/${platform}-text.json`,
    );
    const signature = createHmac("sha256", "demo-only")
      .update(bytes)
      .digest("hex");
    const response = await request.post(api + "/integrations/meta/webhook", {
      data: bytes,
      headers: {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": `sha256=${signature}`,
      },
    });
    expect(response.ok()).toBeTruthy();
    expect((await response.json()).items[0].status).toBe("queued");
    const again = await request.post(api + "/integrations/meta/webhook", {
      data: bytes,
      headers: {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": `sha256=${signature}`,
      },
    });
    expect((await again.json()).items[0].duplicate).toBe(true);
  }
  await connections
    .getByRole("button", { name: "Fetch sandbox insights" })
    .click();
  await expect(
    connections.getByText("Synthetic launch ad set", { exact: true }),
  ).toBeVisible();
  await expect(
    connections.getByText("USD 42.00", { exact: true }),
  ).toBeVisible();
  await expect(
    connections.getByText("USD 7.00", { exact: true }),
  ).toBeVisible();
  await expect(connections.getByText("1.75%", { exact: true })).toBeVisible();
  await expect(
    connections.getByText("Ren proposes pause", { exact: true }),
  ).toBeVisible();
  await toggle.check();
  await connections
    .getByRole("button", { name: "Save auto-act preference" })
    .click();
  await expect(connections.getByRole("status")).toContainText(
    "server execution switch",
  );
  await connections.getByText("Action log · proposed and executed").click();
  await expect(
    connections.getByText("auto act changed", { exact: true }),
  ).toBeVisible();
  const status = await request
    .get(api + "/integrations/ads/status")
    .then((r) => r.json());
  expect(status.auto_act).toBe(true);
  expect(status.server_send_enabled).toBe(false);
  const queue = await request.get(api + "/queue").then((r) => r.json());
  expect(queue.items).toHaveLength(4);
  const log = await request
    .get(api + "/integrations/ads/log")
    .then((r) => r.json());
  expect(
    log.items.some((row: { event: string }) =>
      row.event.startsWith("execution"),
    ),
  ).toBe(false);
  mkdirSync(folder, { recursive: true });
  await connections.screenshot({ path: `${folder}/desktop.png` });
  await page.setViewportSize({ width: 390, height: 844 });
  await connections.screenshot({ path: `${folder}/mobile-390.png` });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  await page.reload();
  await page.getByRole("button", { name: /^Teach/ }).click();
  await expect(connections.getByRole("checkbox")).toBeChecked();
  writeFileSync(
    `${folder}/browser.json`,
    JSON.stringify(
      {
        scope:
          "Synthetic sandbox insights, deterministic Ren test fixture, httpx.MockTransport. No live Meta API delivery, model evaluation, provider calls or account changes.",
        viewports: [
          { width: 1440, height: 1100 },
          { width: 390, height: 844, document_width: 390 },
        ],
        incoming_platforms: ["whatsapp", "messenger", "instagram"],
        pending_decisions: queue.items.length,
        duplicate_delivery_checked: true,
        ads_auto_act_persisted: status.auto_act,
        ads_server_switch: status.server_send_enabled,
        execution_attempts: 0,
        log_events: log.items.map((row: { event: string }) => row.event),
      },
      null,
      2,
    ) + "\n",
  );
});
