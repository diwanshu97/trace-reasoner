import importlib.util
import json

import pytest

from trace_reasoner.datasets.synthetic import normal_traces
from trace_reasoner.mcp.session import TraceReasonerSession
from trace_reasoner.tools.baseline import LatencyBaseline

mcp_installed = importlib.util.find_spec("mcp") is not None


def make_session() -> TraceReasonerSession:
    return TraceReasonerSession(baseline=LatencyBaseline.from_traces(normal_traces(60, seed=0)))


# --- session logic (no mcp dependency) ----------------------------------
def test_requires_a_loaded_trace_first():
    with pytest.raises(ValueError):
        TraceReasonerSession().survey()


def test_load_then_survey_walk_and_baseline():
    s = make_session()

    loaded = json.loads(s.load_synthetic_trace(seed=42, fault="latency"))
    assert loaded["n_spans"] > 0
    assert loaded["survey"]["hottest_by_self_time"]

    sv = json.loads(s.survey())
    assert sv["n_spans"] == loaded["n_spans"]

    root_id = sv["critical_path"][0]["span_id"]
    children = json.loads(s.walk_tree(root_id, "children"))
    assert isinstance(children, list)

    hottest = sv["hottest_by_self_time"][0]["span_id"]
    verdict = json.loads(s.baseline_latency(hottest))
    assert verdict["span_id"] == hottest
    assert "is_anomalous" in verdict


def test_walk_unknown_span_raises():
    s = make_session()
    s.load_synthetic_trace(seed=1)
    with pytest.raises(KeyError):
        s.walk_tree("no-such-span", "children")


# --- server registration (only when the mcp SDK is installed) -----------
@pytest.mark.skipif(not mcp_installed, reason="mcp SDK not installed")
def test_server_module_constructs():
    from trace_reasoner.mcp import server

    assert server.mcp is not None
    assert callable(server.main)
