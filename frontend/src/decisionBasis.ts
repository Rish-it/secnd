import type { Decision } from "./api";

const atLeast = (value: number) => `${Math.floor(value * 100)}%`;
export function decisionBasis(item: Decision) {
  const basis = item.basis;
  if (typeof basis === "string" && /\s/.test(basis)) return basis;
  const details = typeof basis === "object" ? basis : null;
  const description = details?.display_text ?? details?.text ?? details?.summary;
  if (description) return description;
  const kind = (
    typeof basis === "string"
      ? basis
      : details?.kind ?? details?.type ?? details?.source ?? ""
  ).toLowerCase();
  const publicHistory = /public|audited|benchmark/.test(kind);
  const own = !publicHistory && /personal|own|review|business|workspace/.test(kind);

  // An explicit basis wins. Legacy responses can prove business reviews through autonomy.
  if (own || (!basis && item.autonomy)) {
    const evidence = { ...item.autonomy, ...details };
    if (evidence.confirmed != null && evidence.reviewed != null && evidence.lower_bound != null) {
      return `Measured on your own reviews: ${evidence.confirmed} of ${evidence.reviewed} confirmed, at least ${atLeast(evidence.lower_bound)}`;
    }
    return "Your own review evidence is not available for this saved decision.";
  }
  const bound = details?.lower_bound ?? details?.cp_lower_bound ?? item.cp_lower_bound;
  const legacyPublic = !basis && item.reader?.scope_authorized && item.reader?.policy_complete;
  if ((publicHistory || legacyPublic) && bound != null) {
    return `Certified at ${atLeast(bound)} on audited public history`;
  }
  return bound == null
    ? "Certification details unavailable. Check Autonomy before allowing Act."
    : `Measured on saved calibration history: at least ${atLeast(bound)}. This is separate from your own reviews.`;
}
