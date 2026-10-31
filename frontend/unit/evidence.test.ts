import assert from "node:assert/strict";
import { test } from "node:test";
import { decisionBasis } from "../src/decisionBasis.ts";
import type { Decision } from "../src/api.ts";

const saved: Decision = {
  item_id: "synthetic", label: "keep", probabilities: { keep: 1 }, prob: 1,
  auto: true, threshold: .9, model_version: "synthetic", spot_check: false,
  cp_lower_bound: .994, reasons: { merchant_memory: false, neighbours: [] },
};
test("own review proof uses the lower bound, rounded down", () => {
  assert.equal(decisionBasis({ ...saved, autonomy: { reviewed: 61, confirmed: 61, lower_bound: .952, effective_level: "act" } }), "Measured on your own reviews: 61 of 61 confirmed, at least 95%");
});
test("explicit public basis wins over unrelated business reviews", () => {
  assert.equal(decisionBasis({ ...saved, basis: { kind: "audited_public_history", lower_bound: .994 }, autonomy: { reviewed: 3, confirmed: 2, lower_bound: .1, effective_level: "ask" } }), "Certified at 99% on audited public history");
});
test("explicit own basis can use legacy counts but never public certification", () => {
  assert.equal(decisionBasis({ ...saved, basis: { kind: "personal_reviews" }, autonomy: { reviewed: 61, confirmed: 61, lower_bound: .952, effective_level: "act" } }), "Measured on your own reviews: 61 of 61 confirmed, at least 95%");
  assert.match(decisionBasis({ ...saved, basis: { kind: "personal_reviews" } }), /not available/);
});
test("plain calibration is never presented as audited public history", () => {
  assert.match(decisionBasis(saved), /^Measured on saved calibration history:/);
  assert.doesNotMatch(decisionBasis(saved), /audited public/);
  assert.match(decisionBasis({ ...saved, cp_lower_bound: null }), /unavailable/);
});
