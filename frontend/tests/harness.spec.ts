import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// Serve the built product through request interception: no listening socket or live network.
test.beforeEach(async ({ page }) => {
  await page.route("**/*", async route => {
    const path = new URL(route.request().url()).pathname;
    if (path !== "/" && !/^\/assets\/[a-zA-Z0-9._-]+$/.test(path)) {
      await route.fulfill({ status: 404, body: "Synthetic test route unavailable" });
      return;
    }
    const file = fileURLToPath(new URL(path === "/" ? "../dist/index.html" : `../dist${path}`, import.meta.url));
    await route.fulfill({ body: await readFile(file), contentType: path.endsWith(".js") ? "text/javascript" : path.endsWith(".css") ? "text/css" : "text/html" });
  });
});

const root = "/api/v1/workspaces/default";
const proposal = (id: string, source = "whatsapp") => ({
  item_id: id, source_app: source, label: "Reply", prob: .96,
  probabilities: { Reply: .96, Escalate: .03, Archive: .01 }, auto: false,
  threshold: .9, model_version: "synthetic-test-model", spot_check: false,
  item: { item_id: id, text: `Synthetic · Delivery question ${id}`, source_app: source },
  reasons: { merchant_memory: false, neighbours: [{ item_id: `past-${id}`, text: "Synthetic · You shared a delivery update", label: "Reply", similarity: .94 }] },
});
const policy = (decision_type: string) => ({
  decision_type, level: "ask", effective_level: "ask", bar: 95, confidence_cut: .9,
  allow_raise_budget: false, allow_customer_messages: false,
  reviewed: 40, confirmed: 40, corrected: 0, lower_bound: .9278,
  unlocked: false, confirmations_left: 19, window: 1000,
});

async function mockBusiness(page: Page) {
  let queue = [proposal("one"), proposal("two", "instagram")];
  let corrected: ReturnType<typeof proposal>[] = [];
  const answers = new Map<string, { row: ReturnType<typeof proposal>; label: string }>();
  let policies = ["ad", "inbox", "account"].map(policy);
  let agent = { enabled: false, share_counts: false, global_killed: false, worker_enabled: true, activity: [] };
  let kills = 0;
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname;
    const method = route.request().method();
    let data: unknown = {};
    if (path === "/api/v1/auth/session") data = { signed_in: false, demo: true };
    else if (path === "/api/v1/workspaces") data = { items: ["default"] };
    else if (path === "/api/health") data = { status: "ok", database: "ready" };
    else if (path === root + "/serving") data = { model_id: "synthetic-test-model", synthetic: true };
    else if (path === root + "/labels") data = ["Reply", "Escalate", "Archive"];
    else if (path === root + "/queue") data = { items: queue, next_cursor: null };
    else if (path === root + "/feed") data = { items: [
      { ...proposal("own"), auto: true, autonomy: { confirmed: 61, reviewed: 61, lower_bound: .952, effective_level: "act" } },
      { ...proposal("public"), auto: true, basis: { kind: "audited_public_history", lower_bound: .994 }, autonomy: { confirmed: 2, reviewed: 3, lower_bound: .1, effective_level: "ask" } },
      { ...proposal("explicit-own"), auto: true, basis: { kind: "personal_reviews", confirmed: 61, reviewed: 61, lower_bound: .952 }, cp_lower_bound: .994 },
    ], next_cursor: null };
    else if (path === root + "/overridden") data = { items: corrected, next_cursor: null };
    else if (path === root + "/answers" && method === "POST") {
      const { item_id, label } = route.request().postDataJSON();
      const row = queue.find(item => item.item_id === item_id)!;
      answers.set(route.request().headers()["idempotency-key"], { row, label });
      queue = queue.filter(item => item.item_id !== item_id);
      if (label !== row.label) corrected.push({ ...row, ...{ answer_label: label } });
      data = { item_id, coverage: 0, model_version: "synthetic-test-model", retrain: "history_updated" };
    } else if (path === root + "/answers:undo") {
      const { idempotency_key } = route.request().postDataJSON();
      const last = answers.get(idempotency_key)!;
      queue.push(last.row); corrected = corrected.filter(row => row.item_id !== last.row.item_id);
      data = { item_id: last.row.item_id, model_version: "synthetic-test-model", coverage: 0, retrain: "history_updated" };
    } else if (path === root + "/autonomy") data = policies;
    else if (path.startsWith(root + "/autonomy/") && method === "PUT") {
      const kind = path.split("/").at(-1);
      policies = policies.map(p => p.decision_type === kind ? { ...p, ...route.request().postDataJSON(), effective_level: route.request().postDataJSON().level === "act" ? "ask" : route.request().postDataJSON().level } : p);
      data = policies.find(p => p.decision_type === kind);
    } else if (path === "/api/v1/track-record") data = { counts: [] };
    else if (path === root + "/agent") {
      if (method === "PUT") agent = { ...agent, ...route.request().postDataJSON() };
      data = agent;
    } else if (path === root + "/outcomes") data = [];
    else if (path === "/api/v1/agent/kill") { kills++; agent.global_killed = true; data = { stopped: true }; }
    else if (path.endsWith("/meta/status")) data = { accounts: { whatsapp: [], messenger: [], instagram: [] }, model_ready: false, signing_configured: false, whatsapp_send_enabled: false };
    else if (path.endsWith("/ads/status")) data = { sandbox_allowed: false };
    else if (path.endsWith("/meta/connection")) data = { connected: false, assets: [], selected: [] };
    else { await route.fulfill({ status: 404, json: { message: "Synthetic test route unavailable" } }); return; }
    await route.fulfill({ json: data, headers: { "X-Secnd-Demo": "synthetic", "X-Secnd-Mode": "live" } });
  });
  return { kills: () => kills };
}

test("first run: sign in, connect Meta, pick accounts, set bar, done", async ({ page }) => {
  await mockBusiness(page);
  let signedIn = false;
  let connected = false;
  let selected: string[] = [];
  const asset = { key: "messenger:20", kind: "messenger", id: "20", name: "Synthetic Studio Page" };
  await page.route("**/api/v1/auth/session", route => route.fulfill({ json: { signed_in: signedIn, demo: false } }));
  await page.route("**/api/v1/auth/login", route => { signedIn = true; return route.fulfill({ json: { signed_in: true, demo: false } }); });
  await page.route("**/integrations/meta/connection", route => route.fulfill({ json: { connected, assets: connected ? [asset] : [], selected } }));
  await page.route("**/integrations/meta/oauth/start", route => { connected = true; return route.fulfill({ status: 302, headers: { location: "/?connected=true" } }); });
  await page.route("**/integrations/meta/accounts", route => { selected = route.request().postDataJSON().assets; return route.fulfill({ json: { connected, assets: [asset], selected } }); });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("list", { name: "Set up your business" })).toBeVisible();
  await page.getByLabel("Sign-in name").fill("synthetic-owner");
  await page.getByLabel("Password", { exact: true }).fill("synthetic-password");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await page.getByRole("link", { name: "Connect Meta", exact: true }).click();
  const account = page.getByRole("checkbox", { name: /Synthetic Studio Page/ });
  await account.click();
  await expect(account).toBeChecked();
  await page.getByRole("button", { name: "Use selected accounts" }).click();
  const bar = page.getByRole("radio", { name: /^98%/ });
  await bar.click(); await expect(bar).toBeChecked();
  await page.getByRole("button", { name: "Save my bar" }).click();
  await expect(page.getByText("Your bar:", { exact: false })).toContainText("98%");
  await page.getByRole("button", { name: "Open my inbox" }).click();
  await expect(page.getByRole("heading", { name: "Your decision inbox." })).toBeVisible();
  expect(selected).toEqual(["messenger:20"]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
  await page.reload();
  await expect(page.getByRole("heading", { name: "Your decision inbox." })).toBeVisible();
});

test("inbox keyboard moves, confirms, corrects, undoes and ignores typing", async ({ page }) => {
  await mockBusiness(page); await page.goto("/");
  const panel = page.getByRole("tabpanel", { name: /^Needs you/ });
  await expect(panel.getByRole("article")).toHaveCount(2);
  await expect(page.getByText(/move.*confirm.*correct.*undo/)).toBeVisible();
  await page.keyboard.press("j");
  await expect(panel.getByRole("article").first()).toBeFocused();
  await page.keyboard.press("j");
  await expect(panel.getByRole("article").last()).toBeFocused();
  await page.keyboard.press("k");
  await page.keyboard.press("y");
  await expect(panel.getByRole("article")).toHaveCount(1);
  await page.keyboard.press("u");
  await expect(panel.getByRole("article")).toHaveCount(2);
  await page.keyboard.press("j"); await page.keyboard.press("1");
  await expect(panel.getByRole("article")).toHaveCount(1);
  await page.getByRole("tab", { name: /^Corrected/ }).click();
  await expect(page.getByRole("tabpanel", { name: /^Corrected/ }).getByText(/Your correction is in its history/)).toBeVisible();
  await page.keyboard.press("u");
  await page.getByRole("tab", { name: /^Needs you/ }).click();
  await expect(panel.getByRole("article")).toHaveCount(2);
  await panel.getByText("Why · nearest past decisions").first().click();
  await expect(panel.getByText("Synthetic · You shared a delivery update").first()).toBeVisible();
  await page.getByText("Switch business", { exact: true }).click();
  const business = page.getByRole("textbox", { name: "Business", exact: true });
  await business.fill(""); await page.keyboard.type("jkyu1234");
  await expect(panel.getByRole("article")).toHaveCount(2);
});

test("automatic decisions distinguish personal reviews and audited public evidence", async ({ page }) => {
  await mockBusiness(page); await page.goto("/");
  await page.getByRole("tab", { name: /^Done by Ren/ }).click();
  const panel = page.getByRole("tabpanel", { name: /^Done by Ren/ });
  await expect(panel.getByText("Measured on your own reviews: 61 of 61 confirmed, at least 95%", { exact: true })).toHaveCount(2);
  const publicCard = panel.getByRole("article", { name: /question public/ });
  await expect(publicCard.getByText("Certified at 99% on audited public history", { exact: true })).toBeVisible();
  await expect(publicCard.getByText(/Measured on your own reviews/)).toHaveCount(0);
});

test("kill switch requires one inline confirmation and can be cancelled", async ({ page }) => {
  const mocked = await mockBusiness(page); await page.goto("/");
  await page.getByRole("button", { name: "Activity", exact: true }).click();
  const stop = page.getByRole("button", { name: "Stop all automatic actions", exact: true });
  await stop.click(); expect(mocked.kills()).toBe(0);
  await expect(page.getByRole("alertdialog")).toBeVisible();
  await page.getByRole("button", { name: "Keep running" }).click();
  expect(mocked.kills()).toBe(0);
  await stop.click(); await page.getByRole("button", { name: "Confirm stop", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "All automatic actions stopped." })).toBeVisible();
  expect(mocked.kills()).toBe(1);
  await expect(page.getByRole("button", { name: "Automatic actions stopped", exact: true })).toBeDisabled();
});

test("390px: no overflow, saved controls update, light and dark", async ({ page }) => {
  await mockBusiness(page); await page.setViewportSize({ width: 390, height: 844 }); await page.goto("/");
  await expect(page.getByRole("heading", { name: "Your decision inbox." })).toBeVisible();
  for (const view of ["Inbox", "Autonomy", "Activity", "Connections", "Past decisions", "About"]) {
    await page.getByRole("button", { name: view, exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth), view).toBe(390);
  }
  await page.getByRole("button", { name: "Autonomy", exact: true }).click();
  const ads = page.getByRole("region", { name: "Ads autonomy" });
  await expect(ads.getByRole("progressbar")).toHaveAttribute("value", "0.9278");
  await expect(ads.getByText(/19 more consecutive confirmations/)).toBeVisible();
  const watch = ads.getByRole("radio", { name: /^Watch/ });
  await watch.click();
  await expect(page.getByRole("status").filter({ hasText: /Current level: watch/ })).toBeVisible();
  await expect(watch).toBeChecked();
  await page.getByRole("button", { name: "Activity", exact: true }).click();
  const share = page.getByRole("checkbox", { name: /Share anonymous aggregate/ });
  await share.click();
  await expect(page.getByRole("status").filter({ hasText: "Preference saved." })).toBeVisible();
  await expect(share).toBeChecked();
  await page.getByRole("button", { name: /Use dark appearance/ }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test("sign-in has useful errors at 390px", async ({ page }) => {
  await mockBusiness(page);
  await page.route("**/api/v1/auth/session", route => route.fulfill({ json: { signed_in: false, demo: false } }));
  await page.route("**/api/v1/auth/login", route => route.fulfill({ status: 401, json: { message: "Sign-in name or password is incorrect." } }));
  await page.setViewportSize({ width: 390, height: 844 }); await page.goto("/");
  await page.getByLabel("Sign-in name").fill("synthetic-person");
  await page.getByLabel("Password", { exact: true }).fill("synthetic-password");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("alert")).toHaveText("Sign-in name or password is incorrect.");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});
