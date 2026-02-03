"""
Tests for SchemaSanitizer - OpenAI function schema validation and generation.
"""

import pytest
from services.schema_sanitizer import SchemaSanitizer
from services.tool_contracts import UnifiedToolDefinition, ParameterDefinition, DependencySource


@pytest.fixture
def sanitizer():
    return SchemaSanitizer()


def _make_tool(params=None):
    """Helper to build a tool definition."""
    param_defs = {}
    if params:
        for name, ptype, required in params:
            param_defs[name] = ParameterDefinition(
                name=name,
                param_type=ptype,
                required=required,
                description=f"The {name}",
            )
    return UnifiedToolDefinition(
        operation_id="get_TestEndpoint",
        method="GET",
        path="/api/test",
        description="Test endpoint description",
        parameters=param_defs,
        service_name="test_service",
        service_url="https://api.example.com",
        swagger_name="test",
    )


class TestSanitizeToolSchema:
    def test_basic_schema_structure(self, sanitizer):
        tool = _make_tool([("VehicleId", "string", True)])
        schema = sanitizer.sanitize_tool_schema(tool)

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "get_TestEndpoint"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_required_params_included(self, sanitizer):
        tool = _make_tool([
            ("VehicleId", "string", True),
            ("Note", "string", False),
        ])
        schema = sanitizer.sanitize_tool_schema(tool)
        params = schema["function"]["parameters"]

        assert "VehicleId" in params["properties"]
        assert "VehicleId" in params.get("required", [])

    def test_context_only_params_excluded(self, sanitizer):
        tool = _make_tool()
        tool.parameters["hidden"] = ParameterDefinition(
            name="hidden",
            param_type="string",
            required=False,
            description="Hidden param",
            dependency_source=DependencySource.FROM_CONTEXT,
        )
        tool.parameters["visible"] = ParameterDefinition(
            name="visible",
            param_type="string",
            required=False,
            description="Visible param",
            dependency_source=DependencySource.FROM_USER,
        )
        schema = sanitizer.sanitize_tool_schema(tool)
        props = schema["function"]["parameters"]["properties"]

        assert "hidden" not in props
        assert "visible" in props

    def test_description_truncated(self, sanitizer):
        long_desc = "x" * 2000
        tool = _make_tool()
        tool.description = long_desc
        schema = sanitizer.sanitize_tool_schema(tool)

        assert len(schema["function"]["description"]) <= 1024


class TestBuildParamSchema:
    def test_string_type(self, sanitizer):
        param = ParameterDefinition(name="Name", param_type="string", required=True, description="The name")
        result = sanitizer._build_param_schema(param, "Name")
        assert result["type"] == "string"

    def test_integer_type(self, sanitizer):
        param = ParameterDefinition(name="Count", param_type="integer", required=True, description="Count")
        result = sanitizer._build_param_schema(param, "Count")
        assert result["type"] == "integer"

    def test_array_type_with_items(self, sanitizer):
        param = ParameterDefinition(
            name="Tags", param_type="array", required=False,
            description="Tag list", items_type="string"
        )
        result = sanitizer._build_param_schema(param, "Tags")
        assert result["type"] == "array"
        assert "items" in result

    def test_invalid_type_defaults_to_string(self, sanitizer):
        param = ParameterDefinition(name="X", param_type="INVALID", required=False, description="Bad type")
        result = sanitizer._build_param_schema(param, "X")
        assert result["type"] == "string"

    def test_enum_values_included(self, sanitizer):
        param = ParameterDefinition(
            name="Status", param_type="string", required=False,
            description="Status", enum_values=["active", "inactive"]
        )
        result = sanitizer._build_param_schema(param, "Status")
        assert "enum" in result
        assert "active" in result["enum"]


class TestValidateOpenAISchema:
    def test_valid_schema_passes(self, sanitizer):
        schema = {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name"}
                    },
                    "required": ["name"]
                }
            }
        }
        assert sanitizer.validate_openai_schema(schema) is True

    def test_missing_type_raises(self, sanitizer):
        schema = {
            "function": {
                "name": "test",
                "description": "test",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        with pytest.raises(ValueError):
            sanitizer.validate_openai_schema(schema)

    def test_missing_name_raises(self, sanitizer):
        schema = {
            "type": "function",
            "function": {
                "description": "test",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        with pytest.raises(ValueError):
            sanitizer.validate_openai_schema(schema)

    def test_missing_parameters_raises(self, sanitizer):
        schema = {
            "type": "function",
            "function": {
                "name": "test",
                "description": "test"
            }
        }
        with pytest.raises(ValueError):
            sanitizer.validate_openai_schema(schema)

    def test_invalid_property_type_raises(self, sanitizer):
        schema = {
            "type": "function",
            "function": {
                "name": "test",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bad": {"type": "INVALID_TYPE", "description": "bad"}
                    }
                }
            }
        }
        with pytest.raises(ValueError):
            sanitizer.validate_openai_schema(schema)

    def test_array_without_items_raises(self, sanitizer):
        schema = {
            "type": "function",
            "function": {
                "name": "test",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "description": "tags"}
                    }
                }
            }
        }
        with pytest.raises(ValueError):
            sanitizer.validate_openai_schema(schema)
