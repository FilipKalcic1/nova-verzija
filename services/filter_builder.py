from typing import Optional

from services.tool_contracts import UnifiedToolDefinition


class FilterBuilder:
    @staticmethod
    def build_filter_string(tool: UnifiedToolDefinition, resolved_params: dict) -> Optional[str]:
        """
        Builds a filter string for parameters that are marked as filterable.
        e.g., Phone(contains)123456 and Name(=)John

        Args:
            tool: The tool definition containing parameter metadata.
            resolved_params: The dictionary of resolved parameters and their values.

        Returns:
            A filter string if any filterable parameters are found, otherwise None.
        """
        filters = []
        for name, value in resolved_params.items():
            param_def = tool.parameters.get(name)
            if param_def and param_def.is_filterable:
                # Build string: e.g., Phone(contains)123456
                filters.append(f"{name}{param_def.preferred_operator}{value}")
        
        return " and ".join(filters) if filters else None
