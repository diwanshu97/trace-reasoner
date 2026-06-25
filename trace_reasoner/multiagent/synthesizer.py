"""The synthesizer: reconcile specialist findings into a ranked Prediction (Checkpoint 5).

This is the hub of the hub-and-spoke. It is the only role that reads every specialist's
finding, and it is deliberately plain Python — the "decision maker / controller as
deterministic Python, not a model" from Checkpoint 4 — so the reconciliation is reproducible
and adds zero hidden LLM calls to the iso-token budget.

Two jobs:

  reconcile(...)        fold the latest finding per role into a ranked top-3 Prediction with a
                        calibrated confidence, applying the Checkpoint 3 rule that verified
                        evidence (a confirmed baseline anomaly) outweighs advisory evidence.

  redispatch_targets(...)  decide whether to send one targeted question back to a specialist.
                        Fires when the analysts disagree on which span is the cause and the
                        dependency/pattern lens points somewhere the latency lens never
                        baseline-checked — exactly the Redis-vs-pool-saturation reframe.
"""

from __future__ import annotations

from trace_reasoner.eval.metrics import Hypothesis, Prediction
from trace_reasoner.multiagent.state import Finding, latest_per_role

# Each lens is independent evidence that a span is the cause. A confirmed-anomalous latency
# finding is verified evidence and carries the most weight; the structural and precedent lenses
# are advisory. `_VERIFIED` is an extra evidence term contributed when baseline confirms anomaly.
_ROLE_WEIGHT = {"latency": 1.0, "dependency": 0.8, "pattern": 0.6}
_VERIFIED = 0.5


def reconcile(trace_id: str, findings: list[Finding]) -> Prediction:
    """Combine the specialists' findings into a ranked, calibrated Prediction.

    Evidence is pooled per implicated span across the lenses that named it (keeping each role's
    strongest finding for a span, so a re-dispatched re-check never erases the original verified
    finding for a different span). Two Checkpoint-3/4 rules decide the outcome:
      - a confirmed-anomalous latency finding is verified evidence and contributes an extra term;
      - a latency finding reporting a span NOT anomalous is a negative tool result, which
        hard-prunes that span regardless of how confidently a structural lens named it.

    Confidence combines the per-lens evidence by noisy-OR — 1 - prod(1 - w*conf) — which is
    bounded in [0,1] and calibration-sound: one verified lens alone reads confident, agreement
    raises it toward 1, and a lone advisory guess stays low. This is the calibration property
    from Checkpoint 1 (confident-and-wrong must hurt more than honestly-uncertain).
    """
    # Per (role, span) keep the highest-confidence finding, so one role can't double-vote a span.
    best: dict[tuple[str, str], Finding] = {}
    pruned: set[str] = set()
    for f in findings:
        if not f.span_id:
            continue
        if f.role == "latency" and f.anomalous is False:
            pruned.add(f.span_id)  # baseline says normal — hard prune (CP4)
        key = (f.role, f.span_id)
        if key not in best or f.confidence > best[key].confidence:
            best[key] = f

    terms: dict[str, list[float]] = {}  # span_id -> evidence strengths to combine by noisy-OR
    families: dict[str, str | None] = {}
    evidence: dict[str, list[str]] = {}
    verified_family: dict[str, str | None] = {}  # family from a confirmed latency finding (wins)

    for (role, span_id), f in best.items():
        if span_id in pruned:
            continue
        terms.setdefault(span_id, []).append(_ROLE_WEIGHT.get(role, 0.5) * f.confidence)
        families.setdefault(span_id, f.fault_family)
        evidence.setdefault(span_id, []).append(f"{role}: {f.evidence[0] if f.evidence else 'no rationale'}")
        if role == "latency" and f.anomalous is True:
            terms[span_id].append(_VERIFIED)
            verified_family[span_id] = f.fault_family

    if not terms:
        # No lens implicated a (surviving) span — honest "inconclusive within budget" (Checkpoint 1).
        return Prediction(trace_id=trace_id, ranked=[])

    ranked = [
        Hypothesis(
            span_id=span_id,
            confidence=round(_noisy_or(strengths), 3),
            # Prefer the family from a confirmed (verified) latency finding over an advisory lens.
            fault_family=verified_family.get(span_id) or families.get(span_id),
            evidence=evidence.get(span_id, []),
        )
        for span_id, strengths in terms.items()
    ]
    ranked.sort(key=lambda h: h.confidence, reverse=True)
    return Prediction(trace_id=trace_id, ranked=ranked[:3])


def _noisy_or(strengths: list[float]) -> float:
    """Combine independent evidence strengths: 1 - prod(1 - s). Bounded in [0, 1]."""
    product = 1.0
    for s in strengths:
        product *= 1.0 - min(1.0, max(0.0, s))
    return 1.0 - product


def redispatch_targets(findings: list[Finding]) -> list[str]:
    """One targeted re-dispatch when verification never reached the structurally-implicated span.

    If the dependency or pattern lens confidently points at a span the latency analyst has never
    baselined — the subtle branch the loud signal hid — ask the latency analyst to check it. A
    span the latency lens has already examined is not re-requested, so a settled span is not
    re-litigated; the hard convergence guarantee is the synthesizer's `rounds < max_rounds` cap
    (one re-dispatch with the default max_rounds=2). Returns [] when nothing is worth another round.
    """
    latest = latest_per_role(findings)
    examined = {f.span_id for f in findings if f.role == "latency" and f.span_id}
    structural = [latest.get("dependency"), latest.get("pattern")]

    for f in structural:
        if f and f.span_id and f.span_id not in examined and f.confidence >= 0.5:
            return ["latency"]
    return []


def redispatch_hint(findings: list[Finding]) -> str:
    """The targeted question handed to the re-dispatched latency analyst."""
    latest = latest_per_role(findings)
    leads = [f for f in (latest.get("dependency"), latest.get("pattern")) if f and f.span_id]
    leads.sort(key=lambda f: f.confidence, reverse=True)
    if not leads:
        return "Re-examine the trace against baseline."
    top = leads[0]
    return (
        f"Another analyst implicates span {top.span_id} "
        f"({top.fault_family or 'unknown family'}). Baseline-check it and report whether it is "
        f"genuinely anomalous — it may be the real cause one hop from the obvious slow span."
    )
