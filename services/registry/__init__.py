"""
Tool Registry - Public API facade.
Version: 2.1 (With Documentation Merge - Pillar 5)

This module provides backward-compatible interface to the refactored registry.
All existing imports from services.tool_registry should work unchanged.

NEW in 2.1: Merges tool_documentation.json into tool objects at runtime
so that origin_guide is available throughout the system.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Set

from services.tool_contracts import UnifiedToolDefinition, DependencyGraph

from .tool_store import ToolStore
from .cache_manager import CacheManager
from .swagger_parser import SwaggerParser
from .embedding_engine import EmbeddingEngine
from .search_engine import SearchEngine

logger = logging.getLogger(__name__)

# Documentation cache - loaded once at startup
_documentation_cache: Optional[Dict[str, Any]] = None

# Re-export for backward compatibility
__all__ = ['ToolRegistry', 'ToolStore', 'CacheManager', 'SwaggerParser', 'EmbeddingEngine', 'SearchEngine']


class ToolRegistry:
    """
    Dynamic tool registry with 2-step discovery.

    Architecture:
    1. Load Swagger specs -> Parse -> Build UnifiedToolDefinition
    2. Generate embeddings for semantic search
    3. Build dependency graph for auto-chaining
    4. Persist to .cache/ for fast startup

    This is a facade that coordinates the refactored components.
    """

    MAX_TOOLS_PER_RESPONSE = 12

    def __init__(self, redis_client=None):
        """Initialize tool registry with all components."""
        self.redis = redis_client

        # Initialize components
        self._store = ToolStore()
        self._cache = CacheManager()
        self._parser = SwaggerParser()
        self._embedding = EmbeddingEngine()
        self._search = SearchEngine()

        # PILLAR 5: Documentation merge storage
        self._tool_documentation: Dict[str, Any] = {}

        # State
        self.is_ready = False
        self._load_lock = asyncio.Lock()

        # Load documentation at startup
        self._load_documentation()

        logger.info("ToolRegistry initialized (v3.1 - with documentation merge)")

    def _load_documentation(self) -> None:
        """
        PILLAR 5: Load tool_documentation.json for runtime access.

        This enables origin_guide and other documentation to be available
        throughout the system without repeated file reads.
        """
        global _documentation_cache

        # Use cached version if available
        if _documentation_cache is not None:
            self._tool_documentation = _documentation_cache
            return

        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        doc_path = os.path.join(base_path, "config", "tool_documentation.json")

        if os.path.exists(doc_path):
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    self._tool_documentation = json.load(f)
                    _documentation_cache = self._tool_documentation
                    logger.info(f"PILLAR 5: Loaded documentation for {len(self._tool_documentation)} tools")
            except Exception as e:
                logger.warning(f"Could not load tool documentation: {e}")
                self._tool_documentation = {}
        else:
            logger.info("No tool_documentation.json found - run generate_documentation.py first")

    # ═══════════════════════════════════════════════
    # BACKWARD COMPATIBILITY PROPERTIES
    # ═══════════════════════════════════════════════

    @property
    def tools(self) -> Dict[str, UnifiedToolDefinition]:
        """Access tools dict for backward compatibility."""
        return self._store.tools

    @property
    def embeddings(self) -> Dict[str, List[float]]:
        """Access embeddings dict for backward compatibility."""
        return self._store.embeddings

    @property
    def dependency_graph(self) -> Dict[str, DependencyGraph]:
        """Access dependency graph for backward compatibility."""
        return self._store.dependency_graph

    @property
    def retrieval_tools(self) -> Set[str]:
        """Access retrieval tools set for backward compatibility."""
        return self._store.retrieval_tools

    @property
    def mutation_tools(self) -> Set[str]:
        """Access mutation tools set for backward compatibility."""
        return self._store.mutation_tools

    # Context param patterns (exposed for message_engine)
    @property
    def CONTEXT_PARAM_FALLBACK(self) -> Dict[str, str]:
        """Access context param fallback for backward compatibility."""
        return self._parser.context_param_fallback

    # ═══════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════

    async def load_swagger(self, source: str) -> bool:
        """Backward compatible method for single source."""
        logger.warning("⚠️ load_swagger() is DEPRECATED. Use initialize([sources]).")
        return await self.initialize([source])

    async def initialize(self, swagger_sources: List[str]) -> bool:
        """
        Initialize registry from pre-processed file or Swagger sources.
        """
        async with self._load_lock:
            try:
                # Preferred method: Load from pre-processed file
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                registry_path = os.path.join(base_path, "config", "processed_tool_registry.json")

                if os.path.exists(registry_path):
                    logger.info(f"✅ Pre-processed registry found - loading from {registry_path}")
                    with open(registry_path, 'r', encoding='utf-8') as f:
                        registry_data = json.load(f)
                    
                    for tool_data in registry_data.get("tools", []):
                        tool = UnifiedToolDefinition(**tool_data)
                        self._store.add_tool(tool)
                    
                    for dep_data in registry_data.get("dependency_graph", []):
                        dep = DependencyGraph(**dep_data)
                        self._store.add_dependency(dep)
                    
                    logger.info(f"Loaded {self._store.count()} tools and {len(self._store.dependency_graph)} dependencies from file.")
                    
                    # v20.1: Try to load embeddings from cache to avoid re-generating
                    embeddings_cache_path = os.path.join(base_path, ".cache", "tool_embeddings.json")
                    if os.path.exists(embeddings_cache_path):
                        try:
                            with open(embeddings_cache_path, 'r', encoding='utf-8') as ef:
                                cache_data = json.load(ef)
                            # Cache has wrapper structure: {version, timestamp, embeddings: {op_id: [...]}}
                            cached_embeddings = cache_data.get("embeddings", cache_data)
                            if isinstance(cached_embeddings, dict):
                                for op_id, embedding in cached_embeddings.items():
                                    if op_id in self._store.tools and isinstance(embedding, list):
                                        self._store.add_embedding(op_id, embedding)
                            logger.info(f"📦 Loaded {len(self._store.embeddings)} embeddings from cache")
                        except Exception as e:
                            logger.warning(f"Failed to load embeddings cache: {e}")

                else:
                    logger.warning(f"⚠️ Pre-processed registry not found at {registry_path}. Falling back to dynamic loading from Swagger URLs.")
                    
                    # Fallback to old logic
                    if await self._cache.is_cache_valid(swagger_sources):
                        logger.info("✅ Cache valid - loading from disk")
                        cached_data = await self._cache.load_cache()

                        for tool in cached_data["tools"]:
                            self._store.add_tool(tool)
                        for dep in cached_data["dependency_graph"]:
                            self._store.add_dependency(dep)
                        for op_id, embedding in cached_data["embeddings"].items():
                            self._store.add_embedding(op_id, embedding)
                        
                        logger.info(f"✅ Loaded {self._store.count()} tools from cache.")

                    else:
                        logger.info("🔄 Cache invalid - fetching Swagger specs...")
                        for source in swagger_sources:
                            tools = await self._parser.parse_spec(source, self._embedding.build_embedding_text)
                            for tool in tools:
                                self._store.add_tool(tool)

                        if self._store.count() == 0:
                            logger.error("❌ No tools loaded from Swagger sources")
                            return False
                        
                        logger.info(f"📦 Loaded {self._store.count()} tools from Swagger")

                        # Build dependency graph
                        dep_graph = self._embedding.build_dependency_graph(self._store.tools)
                        for dep in dep_graph.values():
                            self._store.add_dependency(dep)
                
                # v20.1 FIX: Only generate embeddings if not all cached
                cached_count = len(self._store.embeddings)
                total_tools = self._store.count()
                
                if cached_count >= total_tools:
                    logger.info(f"✅ All embeddings cached ({cached_count}/{total_tools}) - skipping generation")
                else:
                    logger.info(f"🔄 Generating missing embeddings ({cached_count}/{total_tools} cached)...")
                    new_embeddings = await self._embedding.generate_embeddings(
                        self._store.tools,
                        self._store.embeddings
                    )
                    for op_id, embedding in new_embeddings.items():
                        self._store.add_embedding(op_id, embedding)
                    logger.info(f"✅ Generated {len(new_embeddings)} new embeddings")

                # Save cache if we didn't load from pre-processed file
                if not os.path.exists(registry_path):
                    await self._cache.save_cache(
                        swagger_sources,
                        list(self._store.tools.values()),
                        self._store.embeddings,
                        list(self._store.dependency_graph.values())
                    )

                self.is_ready = True
                logger.info(
                    f"✅ Initialized {self._store.count()} tools "
                    f"({len(self._store.retrieval_tools)} retrieval, "
                    f"{len(self._store.mutation_tools)} mutation)"
                )
                return True

            except Exception as e:
                logger.error(f"❌ Initialization failed: {e}", exc_info=True)
                return False

    # ═══════════════════════════════════════════════
    # SEARCH & DISCOVERY
    # ═══════════════════════════════════════════════

    async def find_relevant_tools(
        self,
        query: str,
        top_k: int = 5,
        prefer_retrieval: bool = False,
        prefer_mutation: bool = False,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Find relevant tools (backward compatible).

        Returns list of OpenAI function schemas.
        """
        if not self.is_ready:
            logger.warning("Registry not ready")
            return []

        results = await self.find_relevant_tools_with_scores(
            query=query,
            top_k=top_k,
            threshold=threshold or 0.55,
            prefer_retrieval=prefer_retrieval,
            prefer_mutation=prefer_mutation
        )

        return [r["schema"] for r in results]

    async def find_relevant_tools_with_scores(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.55,
        prefer_retrieval: bool = False,
        prefer_mutation: bool = False,
        use_filtered_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find relevant tools WITH SIMILARITY SCORES.

        Args:
            query: User query
            top_k: Number of tools to return
            threshold: Minimum similarity threshold
            prefer_retrieval: Only search GET methods
            prefer_mutation: Only search POST/PUT/DELETE methods
            use_filtered_search: Use FILTER-THEN-SEARCH approach (default True)

        Returns list of dicts with name, score, and schema.
        """
        if not self.is_ready:
            logger.warning("Registry not ready")
            return []

        # v3.0: FILTER-THEN-SEARCH - reduces search space for better accuracy
        if use_filtered_search:
            return await self._search.find_relevant_tools_filtered(
                query=query,
                tools=self._store.tools,
                embeddings=self._store.embeddings,
                dependency_graph=self._store.dependency_graph,
                retrieval_tools=self._store.retrieval_tools,
                mutation_tools=self._store.mutation_tools,
                top_k=top_k,
                threshold=threshold
            )

        # Fallback: Original search method
        return await self._search.find_relevant_tools_with_scores(
            query=query,
            tools=self._store.tools,
            embeddings=self._store.embeddings,
            dependency_graph=self._store.dependency_graph,
            retrieval_tools=self._store.retrieval_tools,
            mutation_tools=self._store.mutation_tools,
            top_k=top_k,
            threshold=threshold,
            prefer_retrieval=prefer_retrieval,
            prefer_mutation=prefer_mutation
        )

    # ═══════════════════════════════════════════════
    # TOOL ACCESS
    # ═══════════════════════════════════════════════

    def get_tool(self, operation_id: str) -> Optional[UnifiedToolDefinition]:
        """Get tool by operation ID."""
        return self._store.get_tool(operation_id)

    def list_tools(self) -> List[str]:
        """List all tool operation IDs."""
        return self._store.list_tools()

    # ═══════════════════════════════════════════════
    # PILLAR 5: DOCUMENTATION ACCESS
    # ═══════════════════════════════════════════════

    def get_tool_documentation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get documentation for a specific tool.

        Returns documentation dict including:
        - purpose
        - when_to_use
        - parameter_origin_guide
        - example_queries_hr
        - etc.
        """
        return self._tool_documentation.get(operation_id)

    def get_parameter_origin_guide(self, operation_id: str) -> Dict[str, str]:
        """
        Get parameter origin guide for a tool.

        Returns dict like:
        {
            "personId": "CONTEXT: Sustav ubacuje iz sesije",
            "vehicleId": "USER: Pitaj korisnika za ovo",
            ...
        }
        """
        doc = self._tool_documentation.get(operation_id, {})
        return doc.get("parameter_origin_guide", {})

    def get_tool_with_documentation(
        self,
        operation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get tool definition merged with its documentation.

        Returns combined dict with tool schema AND documentation.
        Useful for LLM prompts that need full context.
        """
        tool = self._store.get_tool(operation_id)
        if not tool:
            return None

        result = {
            "tool": tool.to_openai_function(),
            "documentation": self._tool_documentation.get(operation_id, {})
        }

        return result

    def is_context_param(self, operation_id: str, param_name: str) -> bool:
        """
        Check if a parameter should be auto-injected from context.

        Returns True if the parameter origin is CONTEXT.
        """
        origin_guide = self.get_parameter_origin_guide(operation_id)
        origin = origin_guide.get(param_name, "")
        return "CONTEXT" in origin.upper()

    def is_user_param(self, operation_id: str, param_name: str) -> bool:
        """
        Check if a parameter should be provided by the user.

        Returns True if the parameter origin is USER.
        """
        origin_guide = self.get_parameter_origin_guide(operation_id)
        origin = origin_guide.get(param_name, "USER")  # Default to USER if not specified
        return "USER" in origin.upper()
