"""Tests for services/reasoning/planner.py – Planner."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from services.reasoning.planner import StepType, PlanStep, ExecutionPlan, Planner


# ---------------------------------------------------------------------------
# DataClasses
# ---------------------------------------------------------------------------

class TestStepType:
    def test_values(self):
        assert StepType.EXECUTE_TOOL.value == "execute_tool"
        assert StepType.ASK_USER.value == "ask_user"
        assert StepType.USER_SELECT.value == "user_select"
        assert StepType.CONFIRM.value == "confirm"


class TestPlanStep:
    def test_defaults(self):
        s = PlanStep(step_number=1, step_type=StepType.EXECUTE_TOOL)
        assert s.tool_name is None
        assert s.parameters == {}
        assert s.question is None
        assert s.reason == ""


class TestExecutionPlan:
    def test_defaults(self):
        p = ExecutionPlan(understanding="test", is_simple=True, has_all_data=True)
        assert p.missing_data == []
        assert p.steps == []
        assert p.direct_response is None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def _make_planner():
    ms = MagicMock()
    ms.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com"
    ms.AZURE_OPENAI_API_KEY = "test-key"
    ms.AZURE_OPENAI_API_VERSION = "2024-02-15"
    ms.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"

    with patch("services.reasoning.planner.settings", ms):
        with patch("services.reasoning.planner.AsyncAzureOpenAI"):
            p = Planner()
    return p


@pytest.fixture
def planner():
    return _make_planner()


def _ctx(person_id="00000000-0000-0000-0000-000000000001", vehicle=True):
    ctx = {"person_id": person_id, "tenant_id": "t1"}
    if vehicle:
        ctx["vehicle"] = {"id": "v1", "plate": "ZG-123", "name": "Golf"}
    return ctx


class TestSummarizeContext:
    def test_with_person_and_vehicle(self, planner):
        result = planner._summarize_context(_ctx())
        assert "person_id" in result
        assert "ZG-123" in result
        assert "Golf" in result

    def test_person_only(self, planner):
        result = planner._summarize_context(_ctx(vehicle=False))
        assert "person_id" in result

    def test_empty(self, planner):
        result = planner._summarize_context({})
        assert "Nema" in result

    def test_with_display_name(self, planner):
        ctx = _ctx()
        ctx["display_name"] = "Igor"
        result = planner._summarize_context(ctx)
        assert "Igor" in result


class TestSummarizeTools:
    def test_basic(self, planner):
        tools = [
            {"name": "get_MasterData", "score": 0.95, "schema": {
                "description": "Get master data",
                "parameters": {"properties": {"VehicleId": {}}, "required": ["VehicleId"]}
            }}
        ]
        result = planner._summarize_tools(tools)
        assert "get_MasterData" in result
        assert "0.95" in result
        assert "VehicleId" in result

    def test_empty(self, planner):
        assert planner._summarize_tools([]) == ""

    def test_no_required(self, planner):
        tools = [{"name": "t", "score": 0.5, "schema": {"description": "d", "parameters": {"properties": {}, "required": []}}}]
        result = planner._summarize_tools(tools)
        assert "nema" in result


class TestParsePlanResponse:
    def test_simple_plan(self, planner):
        response = {
            "understanding": "Dohvati km",
            "is_simple": True,
            "has_all_data": True,
            "steps": [
                {"step": 1, "type": "execute_tool", "tool": "get_MD", "reason": "best"}
            ],
            "direct_response": None
        }
        plan = planner._parse_plan_response(response, [])
        assert plan.understanding == "Dohvati km"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "get_MD"
        assert plan.steps[0].step_type == StepType.EXECUTE_TOOL

    def test_multi_step(self, planner):
        response = {
            "understanding": "Booking",
            "is_simple": False,
            "has_all_data": False,
            "missing_data": ["from", "to"],
            "steps": [
                {"step": 1, "type": "ask_user", "question": "Od kada?"},
                {"step": 2, "type": "execute_tool", "tool": "get_AV"},
                {"step": 3, "type": "user_select"},
                {"step": 4, "type": "confirm"},
            ]
        }
        plan = planner._parse_plan_response(response, [])
        assert len(plan.steps) == 4
        assert plan.steps[0].step_type == StepType.ASK_USER
        assert plan.steps[2].step_type == StepType.USER_SELECT
        assert plan.steps[3].step_type == StepType.CONFIRM

    def test_invalid_step_type(self, planner):
        response = {
            "understanding": "test",
            "is_simple": True,
            "has_all_data": True,
            "steps": [{"step": 1, "type": "INVALID"}]
        }
        plan = planner._parse_plan_response(response, [])
        assert plan.steps[0].step_type == StepType.EXECUTE_TOOL

    def test_empty(self, planner):
        plan = planner._parse_plan_response({}, [])
        assert plan.understanding == ""
        assert plan.steps == []

    def test_direct_response(self, planner):
        response = {
            "understanding": "Greeting",
            "is_simple": True,
            "has_all_data": True,
            "steps": [],
            "direct_response": "Pozdrav!"
        }
        plan = planner._parse_plan_response(response, [])
        assert plan.direct_response == "Pozdrav!"


class TestCreateFallbackPlan:
    def test_no_tools(self, planner):
        plan = planner._create_fallback_plan("test", [])
        assert plan.direct_response is not None
        assert "pomoći" in plan.direct_response or "pojasniti" in plan.direct_response

    def test_with_tools(self, planner):
        tools = [{"name": "get_MD", "score": 0.8}]
        plan = planner._create_fallback_plan("km?", tools)
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "get_MD"


class TestCreatePlan:
    @pytest.mark.asyncio
    async def test_llm_success(self, planner):
        llm_response = {
            "understanding": "Dohvati km",
            "is_simple": True,
            "has_all_data": True,
            "steps": [{"step": 1, "type": "execute_tool", "tool": "get_MD"}]
        }
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(llm_response)
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        planner.openai.chat.completions.create = AsyncMock(return_value=mock_resp)

        plan = await planner.create_plan("km?", _ctx(), [], [{"name": "get_MD", "score": 0.9, "schema": {"description": "d", "parameters": {"properties": {}, "required": []}}}])
        assert plan.understanding == "Dohvati km"
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, planner):
        planner.openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("fail"))

        tools = [{"name": "get_MD", "score": 0.8}]
        plan = await planner.create_plan("km?", _ctx(), [], tools)
        assert plan.steps[0].tool_name == "get_MD"

    @pytest.mark.asyncio
    async def test_llm_failure_no_tools(self, planner):
        planner.openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("fail"))

        plan = await planner.create_plan("test", _ctx(), [], [])
        assert plan.direct_response is not None
