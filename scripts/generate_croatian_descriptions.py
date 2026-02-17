"""
Croatian Description Generator - LLM-based Croatian tool descriptions.
Version: 1.0

Generates Croatian descriptions for all tools using Azure GPT-4.
This replaces manual dictionary-based translation at index time.

Usage:
    python -m scripts.generate_croatian_descriptions

Output:
    Updates config/tool_documentation.json with 'croatian_description' field
    for each tool.

Estimated time: ~20 minutes for 900+ tools
Estimated cost: ~$3-5 (GPT-4)
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncAzureOpenAI, RateLimitError
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
settings = get_settings()

CONFIG_DIR = Path(__file__).parent.parent / "config"
BATCH_SIZE = 5  # Tools per LLM request
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class CroatianDescriptionGenerator:
    """
    Generates Croatian tool descriptions using Azure OpenAI GPT-4.

    For each tool, produces:
    - croatian_description: Full Croatian description of what the tool does
    - croatian_synonyms: Alternative Croatian terms users might use
    - croatian_embedding_text: Optimized text for embedding generation

    This runs ONCE at build time, not at query time.
    """

    def __init__(self):
        self.client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            max_retries=MAX_RETRIES,
            timeout=60.0
        )
        self.model = settings.AZURE_OPENAI_DEPLOYMENT_NAME

    async def generate_all(self) -> Dict[str, Any]:
        """
        Generate Croatian descriptions for all tools.

        Returns:
            Dict mapping operation_id -> croatian description data
        """
        # Load tool registry
        registry_path = CONFIG_DIR / "processed_tool_registry.json"
        if not registry_path.exists():
            logger.error(f"Tool registry not found: {registry_path}")
            return {}

        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)

        tools = registry.get("tools", [])
        logger.info(f"Loaded {len(tools)} tools from registry")

        # Load existing documentation
        doc_path = CONFIG_DIR / "tool_documentation.json"
        existing_docs = {}
        if doc_path.exists():
            with open(doc_path, 'r', encoding='utf-8') as f:
                existing_docs = json.load(f)
            logger.info(f"Loaded existing documentation for {len(existing_docs)} tools")

        # Find tools that need Croatian descriptions
        tools_needing_desc = []
        for tool in tools:
            op_id = tool.get("operation_id", "")
            doc = existing_docs.get(op_id, {})
            if not doc.get("croatian_description"):
                tools_needing_desc.append(tool)

        if not tools_needing_desc:
            logger.info("All tools already have Croatian descriptions")
            return existing_docs

        logger.info(f"Generating Croatian descriptions for {len(tools_needing_desc)} tools...")

        # Process in batches
        results = {}
        for i in range(0, len(tools_needing_desc), BATCH_SIZE):
            batch = tools_needing_desc[i:i + BATCH_SIZE]
            batch_results = await self._process_batch(batch)
            results.update(batch_results)

            progress = min(i + BATCH_SIZE, len(tools_needing_desc))
            logger.info(f"Progress: {progress}/{len(tools_needing_desc)}")

            # Rate limiting
            await asyncio.sleep(1)

        # Merge with existing documentation
        for op_id, desc_data in results.items():
            if op_id in existing_docs:
                existing_docs[op_id].update(desc_data)
            else:
                existing_docs[op_id] = desc_data

        # Save updated documentation
        with open(doc_path, 'w', encoding='utf-8') as f:
            json.dump(existing_docs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved Croatian descriptions to {doc_path}")
        return existing_docs

    async def _process_batch(self, tools: List[Dict]) -> Dict[str, Any]:
        """Process a batch of tools to generate Croatian descriptions."""
        # Build tool summaries for the prompt
        tool_summaries = []
        for tool in tools:
            op_id = tool.get("operation_id", "")
            method = tool.get("method", "GET")
            path = tool.get("path", "")
            description = tool.get("description", "")
            params = tool.get("parameters", {})
            output_keys = tool.get("output_keys", [])

            param_names = list(params.keys())[:10] if isinstance(params, dict) else []
            output_sample = output_keys[:10] if output_keys else []

            tool_summaries.append({
                "operation_id": op_id,
                "method": method,
                "path": path,
                "description": description[:200],
                "parameters": param_names,
                "output_keys": output_sample
            })

        prompt = f"""Za svaki od sljedećih API alata, generiraj:
1. **croatian_description**: Opis na hrvatskom što alat radi (1-2 rečenice). Koristi termine koje bi korisnik fleet management sustava koristio.
2. **croatian_synonyms**: Lista od 3-5 alternativnih hrvatskih pojmova/fraza koje bi korisnik mogao koristiti za ovu funkcionalnost.
3. **croatian_embedding_text**: Optimizirani tekst za vektorsko pretraživanje (uključi sve relevantne pojmove, sinonime i fraze).

KONTEKST: Ovo je MobilityOne fleet management sustav za upravljanje vozilima, rezervacijama, troškovima itd. Korisnici su hrvatski i koriste razgovorne izraze.

ALATI:
{json.dumps(tool_summaries, indent=2, ensure_ascii=False)}

ODGOVORI U JSON FORMATU (dict mapping operation_id -> opis):
{{
  "operation_id_1": {{
    "croatian_description": "...",
    "croatian_synonyms": ["...", "..."],
    "croatian_embedding_text": "..."
  }},
  ...
}}"""

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Ti si stručnjak za fleet management terminologiju na hrvatskom jeziku. Generiraj opise koji koriste svakodnevne hrvatske izraze."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )

                result_text = response.choices[0].message.content
                return json.loads(result_text)

            except RateLimitError:
                wait_time = RETRY_DELAY * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                if attempt < MAX_RETRIES - 1:
                    continue
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue

        # Return empty results for failed batch
        return {tool.get("operation_id", ""): {} for tool in tools}


async def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Croatian Description Generator v1.0")
    logger.info("=" * 60)

    generator = CroatianDescriptionGenerator()
    results = await generator.generate_all()

    # Summary
    total = len(results)
    with_desc = sum(1 for v in results.values() if v.get("croatian_description"))
    logger.info(f"\nSummary:")
    logger.info(f"  Total tools: {total}")
    logger.info(f"  With Croatian descriptions: {with_desc}")
    logger.info(f"  Coverage: {with_desc/total*100:.1f}%" if total > 0 else "  Coverage: N/A")


if __name__ == "__main__":
    asyncio.run(main())
