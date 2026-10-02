"""Regression: real model must search before accepting a turn-one FINAL."""

import json

from arena.model import ModelResponse
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import ReActAgent
from tests.fixtures_briefs import BRIEF_SLA, CORPUS


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)

    def complete(self, messages):
        return ModelResponse(text=next(self.turns), prompt_tokens=7, completion_tokens=7)


def test_premature_final_corrected_once():
    final = 'FINAL: {"answer": "Không đủ căn cứ", "claims": [], "citations": [], "abstain": true}'
    action = 'ACTION: {"tool": "search", "args": {"query": "giao hàng SLA"}}'
    trace = Trace(run_id="premature-final", seed=42)
    tools = Tools(CORPUS, trace, seed=42, flaky=False)
    agent = ReActAgent(ScriptedModel([final, action, final]), tools, trace, corpus=CORPUS)
    agent.run(BRIEF_SLA)
    events = [json.loads(line) for line in trace.to_jsonl().splitlines()]
    assert sum(e["event"] == "model_call" for e in events) == 3
    assert sum(e["event"] == "tool_call" and e["name"] == "search" for e in events) == 1


def test_repeated_premature_final_stops_without_padding():
    final = 'FINAL: {"answer": "Không đủ căn cứ", "claims": [], "citations": [], "abstain": true}'
    trace = Trace(run_id="repeated-final", seed=42)
    tools = Tools(CORPUS, trace, seed=42, flaky=False)
    agent = ReActAgent(ScriptedModel([final, final]), tools, trace, corpus=CORPUS)
    agent.run(BRIEF_SLA)
    events = [json.loads(line) for line in trace.to_jsonl().splitlines()]
    assert agent.last_context.stop_reason == "final"
    assert sum(e["event"] == "model_call" for e in events) == 2
