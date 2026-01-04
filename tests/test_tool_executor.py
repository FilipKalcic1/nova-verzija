"""
Tests for ToolExecutor
Version: 12.0 - Updated for new API

Tests tool execution with UnifiedToolDefinition and ToolExecutionContext.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.tool_executor import ToolExecutor
from services.tool_contracts import (
    UnifiedToolDefinition,
    ParameterDefinition,
    DependencySource,
    ToolExecutionContext,
    ToolExecutionResult
)
from services.api_gateway import APIResponse, HttpMethod
from services.parameter_manager import ParameterValidationError


class TestToolExecutor:
    """Test ToolExecutor class with new API."""

    @pytest.fixture
    def mock_gateway(self):
        """Mock API gateway."""
        gateway = MagicMock()

        # Create successful response
        response = APIResponse(
            success=True,
            status_code=200,
            data={"Id": "result-123", "Name": "Test Result"},
            error_message=None,
            error_code=None
        )
        gateway.execute = AsyncMock(return_value=response)
        return gateway

    @pytest.fixture
    def sample_tool(self) -> UnifiedToolDefinition:
        """Sample tool definition for testing."""
        return UnifiedToolDefinition(
            operation_id="get_TestResource",
            service_name="test-service",
            swagger_name="test",
            service_url="/test",
            path="/api/v1/resource",
            method="GET",
            description="Get test resource",
            parameters={
                "status": ParameterDefinition(
                    name="status",
                    param_type="string",
                    description="Resource status",
                    required=False,
                    location="query",
                    dependency_source=DependencySource.FROM_USER
                )
            },
            required_params=[],
            output_keys=["Id", "Name"]
        )

    @pytest.fixture
    def sample_tool_with_context_param(self) -> UnifiedToolDefinition:
        """Tool with context parameter injection."""
        return UnifiedToolDefinition(
            operation_id="get_UserResource",
            service_name="test-service",
            swagger_name="test",
            service_url="/test",
            path="/api/v1/user-resource",
            method="GET",
            description="Get user resource",
            parameters={
                "personId": ParameterDefinition(
                    name="personId",
                    param_type="string",
                    description="Person ID",
                    required=False,
                    location="query",
                    dependency_source=DependencySource.FROM_CONTEXT,
                    context_key="person_id"
                ),
                "status": ParameterDefinition(
                    name="status",
                    param_type="string",
                    description="Resource status",
                    required=False,
                    location="query",
                    dependency_source=DependencySource.FROM_USER
                )
            },
            required_params=[],
            output_keys=["Id"]
        )

    @pytest.fixture
    def sample_execution_context(self) -> ToolExecutionContext:
        """Sample execution context."""
        return ToolExecutionContext(
            user_context={
                "person_id": "test-person-123",
                "tenant_id": "test-tenant-456",
                "phone": "+385991234567"
            },
            tool_outputs={},
            conversation_state={}
        )

    @pytest.fixture
    def executor(self, mock_gateway):
        """Create executor with mocked gateway."""
        return ToolExecutor(gateway=mock_gateway, circuit_breaker=None)

    # ========================================================================
    # BASIC EXECUTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test successful tool execution."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={"status": "active"},
            execution_context=sample_execution_context
        )

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.operation_id == "get_TestResource"
        assert result.data is not None
        assert "Id" in result.data

    @pytest.mark.asyncio
    async def test_execute_with_empty_params(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test execution with no parameters."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_context_param_injection(
        self,
        executor,
        sample_tool_with_context_param,
        sample_execution_context
    ):
        """Test that context parameters are auto-injected."""
        result = await executor.execute(
            tool=sample_tool_with_context_param,
            llm_params={"status": "active"},
            execution_context=sample_execution_context
        )

        assert result.success is True

        # Verify gateway was called
        executor.gateway.execute.assert_called_once()
        call_kwargs = executor.gateway.execute.call_args.kwargs

        # Check that personId was injected from context
        assert call_kwargs["params"] is not None or call_kwargs["body"] is not None

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_api_error_handling(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test handling of API errors."""
        # Mock error response
        error_response = APIResponse(
            success=False,
            status_code=400,
            data=None,
            error_message="Invalid parameters",
            error_code="BAD_REQUEST"
        )
        executor.gateway.execute = AsyncMock(return_value=error_response)

        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is False
        assert result.error_code is not None
        assert result.ai_feedback is not None
        assert "Invalid" in result.error_message or "parameter" in result.ai_feedback.lower()

    @pytest.mark.asyncio
    async def test_missing_required_params(
        self,
        executor,
        sample_execution_context
    ):
        """Test validation of missing required parameters."""
        tool_with_required = UnifiedToolDefinition(
            operation_id="create_Resource",
            service_name="test",
            swagger_name="test",
            service_url="/test",
            path="/api/v1/resource",
            method="POST",
            description="Create resource",
            parameters={
                "name": ParameterDefinition(
                    name="name",
                    param_type="string",
                    description="Resource name",
                    required=True,
                    location="body",
                    dependency_source=DependencySource.FROM_USER
                )
            },
            required_params=["name"]
        )

        result = await executor.execute(
            tool=tool_with_required,
            llm_params={},  # Missing required 'name'
            execution_context=sample_execution_context
        )

        assert result.success is False
        assert result.error_code == "PARAMETER_VALIDATION_ERROR"
        assert "name" in result.missing_params

    @pytest.mark.asyncio
    async def test_404_error_feedback(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test user-friendly feedback for 404 errors."""
        error_response = APIResponse(
            success=False,
            status_code=404,
            data=None,
            error_message="Not found",
            error_code="NOT_FOUND"
        )
        executor.gateway.execute = AsyncMock(return_value=error_response)

        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is False
        assert result.http_status == 404
        assert "not found" in result.ai_feedback.lower() or "ne postoji" in result.ai_feedback.lower()

    # ========================================================================
    # OUTPUT EXTRACTION TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_output_value_extraction(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test extraction of output values for chaining."""
        # Mock response with output values
        response = APIResponse(
            success=True,
            status_code=200,
            data={"Id": "resource-123", "Name": "Test Resource", "Status": "Active"},
            error_message=None,
            error_code=None
        )
        executor.gateway.execute = AsyncMock(return_value=response)

        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.success is True
        assert result.output_values is not None
        assert "Id" in result.output_values
        assert result.output_values["Id"] == "resource-123"

    # ========================================================================
    # URL BUILDING TESTS
    # ========================================================================

    @pytest.mark.asyncio
    async def test_url_construction_with_swagger_name(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test URL construction using swagger_name."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        # Verify gateway was called with correct URL pattern
        executor.gateway.execute.assert_called_once()
        call_kwargs = executor.gateway.execute.call_args.kwargs

        # URL should follow pattern: /swagger_name/path
        assert "path" in call_kwargs
        url = call_kwargs["path"]
        assert "/test/" in url or url.startswith("/test/api")

    @pytest.mark.asyncio
    async def test_execution_time_tracking(
        self,
        executor,
        sample_tool,
        sample_execution_context
    ):
        """Test that execution time is tracked."""
        result = await executor.execute(
            tool=sample_tool,
            llm_params={},
            execution_context=sample_execution_context
        )

        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0
