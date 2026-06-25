"""The Checkpoint 6 safety / control system: guardrails, trust-risk router, safety metrics.

CP6's thesis is that strong task performance is not enough — an autonomous agent needs guardrails,
calibrated evaluation, and human-intervention points. The three task conditions (A/B/C) supply the
performance; this package supplies the control system around them, composed by `SafeLocalizer`:

    trace -> [guardrails.check_input] -> inner localizer -> [guardrails.verify_prediction]
          -> [router.route] -> AUTO | REVIEW | ESCALATE

`SafeLocalizer` is itself a Localizer, so it scores on the existing harness (an ESCALATE is an
abstention, counted by escalation_rate and on the accuracy-coverage curve); `safety.evaluation`
adds the router-specific metrics (escalation precision, autonomy rate, lane distribution).
"""

from trace_reasoner.safety.evaluation import (
    SafetyReport,
    autonomy_rate,
    escalation_precision,
    lane_distribution,
    safety_report,
)
from trace_reasoner.safety.guardrails import (
    Authorization,
    InputReport,
    ToolPolicy,
    check_input,
    groundedness,
    redact_secrets,
    verify_prediction,
)
from trace_reasoner.safety.router import (
    Lane,
    RoutedDecision,
    RouterThresholds,
    SafeLocalizer,
    ood_score,
    route,
)

__all__ = [
    "SafeLocalizer",
    "RoutedDecision",
    "RouterThresholds",
    "Lane",
    "route",
    "ood_score",
    "check_input",
    "verify_prediction",
    "groundedness",
    "redact_secrets",
    "ToolPolicy",
    "Authorization",
    "InputReport",
    "safety_report",
    "SafetyReport",
    "escalation_precision",
    "autonomy_rate",
    "lane_distribution",
]
