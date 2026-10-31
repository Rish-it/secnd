import { test, expect } from "@playwright/test";

// All Meta-facing requests remain in the local synthetic app or these explicit fixtures.
test("autonomy and activity remain keyboard accessible at 390px", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByText(/Synthetic demo · fictional/)).toBeVisible();
  await page.getByRole("button", { name: /^Autonomy/ }).click();
  await expect(
    page.getByRole("heading", { name: "How much can Ren do?" }),
  ).toBeVisible();
  const ads = page.getByRole("region", { name: "Ads autonomy" });
  await expect(ads.getByText("0", { exact: true })).toHaveCount(3);
  await expect(
    ads.getByText("299 more consecutive confirmations", { exact: false }),
  ).toBeVisible();
  await ads.getByLabel("Permission", { exact: true }).selectOption("ask");
  await expect(
    page.getByRole("status").filter({ hasText: /Current level: ask/ }),
  ).toBeVisible();
  await expect(ads.getByLabel(/Allow budget increases/)).not.toBeChecked();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
  await page.getByRole("button", { name: /^Activity/ }).click();
  await expect(
    page.getByRole("heading", { name: "What happened after" }),
  ).toBeVisible();
  await expect(page.getByLabel(/Share anonymous aggregate/)).not.toBeChecked();
  await page.getByLabel(/Share anonymous aggregate/).check();
  await expect(
    page.getByRole("status").filter({ hasText: "Preference saved." }),
  ).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
});

test("signed-in account picker only submits selected assets and explains role access", async ({
  page,
}) => {
  await page.route("**/api/v1/auth/session", (route) =>
    route.fulfill({ json: { signed_in: true, demo: false } }),
  );
  await page.route("**/api/v1/workspaces", (route) =>
    route.fulfill({ json: { items: ["default"] } }),
  );
  const connection = {
    connected: true,
    assets: [
      {
        key: "messenger:20",
        kind: "messenger",
        id: "20",
        name: "Synthetic Page",
      },
    ],
    selected: [],
    access_note: "App role required",
  };
  await page.route("**/integrations/meta/connection", (route) =>
    route.fulfill({ json: connection }),
  );
  await page.route("**/integrations/meta/accounts", async (route) => {
    expect(route.request().postDataJSON()).toEqual({
      assets: ["messenger:20"],
    });
    await route.fulfill({
      json: { ...connection, selected: ["messenger:20"] },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: /^Connections/ }).click();
  await expect(
    page.getByText(/Until the Meta app has Advanced Access/),
  ).toBeVisible();
  await page.getByLabel(/Synthetic Page/).check();
  await page.getByRole("button", { name: "Use selected accounts" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: /Accounts saved/ }),
  ).toBeVisible();
});

test("sign-in has labelled inputs, no horizontal overflow and a useful error", async ({
  page,
}) => {
  await page.route("**/api/v1/auth/session", (route) =>
    route.fulfill({ json: { signed_in: false, demo: false } }),
  );
  await page.route("**/api/v1/auth/login", (route) =>
    route.fulfill({
      status: 401,
      json: { message: "Sign-in name or password is incorrect." },
    }),
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByLabel("Sign-in name").fill("synthetic-person");
  await page.getByLabel("Password", { exact: true }).fill("synthetic-password");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("alert")).toHaveText(
    "Sign-in name or password is incorrect.",
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  );
});
