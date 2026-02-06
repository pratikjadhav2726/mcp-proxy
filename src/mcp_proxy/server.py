"""
MCP Proxy Server implementation.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Content, TextContent, Tool, ServerCapabilities

from mcp_proxy.logging_config import get_logger
from mcp_proxy.processors import GrepProcessor, ProjectionProcessor

logger = get_logger(__name__)


class ConnectionPoolMetrics:
    """Tracks metrics for connection pool and token savings."""
    
    def __init__(self):
        self.total_calls = 0
        self.total_original_tokens = 0
        self.total_filtered_tokens = 0
        self.projection_calls = 0
        self.grep_calls = 0
        self.connection_count = 0
        self.failed_connections = 0
        
    def record_call(self, original_tokens: int, filtered_tokens: int, 
                   used_projection: bool = False, used_grep: bool = False):
        """Record a tool call with token metrics."""
        self.total_calls += 1
        self.total_original_tokens += original_tokens
        self.total_filtered_tokens += filtered_tokens
        if used_projection:
            self.projection_calls += 1
        if used_grep:
            self.grep_calls += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self.total_original_tokens == 0:
            savings_percent = 0
        else:
            savings_percent = ((self.total_original_tokens - self.total_filtered_tokens) 
                             / self.total_original_tokens * 100)
        
        return {
            "total_calls": self.total_calls,
            "projection_calls": self.projection_calls,
            "grep_calls": self.grep_calls,
            "total_original_tokens": self.total_original_tokens,
            "total_filtered_tokens": self.total_filtered_tokens,
            "tokens_saved": self.total_original_tokens - self.total_filtered_tokens,
            "savings_percent": round(savings_percent, 2),
            "active_connections": self.connection_count,
            "failed_connections": self.failed_connections
        }
    
    def log_summary(self):
        """Log summary statistics."""
        summary = self.get_summary()
        if summary["total_calls"] > 0:
            logger.info(f"=== Proxy Performance Summary ===")
            logger.info(f"  Total calls: {summary['total_calls']}")
            logger.info(f"  Projection calls: {summary['projection_calls']}")
            logger.info(f"  Grep calls: {summary['grep_calls']}")
            logger.info(f"  Original tokens: {summary['total_original_tokens']}")
            logger.info(f"  Filtered tokens: {summary['total_filtered_tokens']}")
            logger.info(f"  Tokens saved: {summary['tokens_saved']}")
            logger.info(f"  Savings: {summary['savings_percent']:.1f}%")
            logger.info(f"  Active connections: {summary['active_connections']}")
            logger.info(f"  Failed connections: {summary['failed_connections']}")


class MCPProxyServer:
    """MCP Proxy Server that intermediates between clients and underlying servers."""

    def __init__(self, underlying_servers: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the proxy server.

        Args:
            underlying_servers: List of server configurations with 'name', 'command', 'args'
        """
        self.server = Server("mcp-proxy-server")
        self.underlying_servers: Dict[str, ClientSession] = {}
        self.server_configs = underlying_servers or []
        self.tools_cache: Dict[str, List[Tool]] = {}
        self.projection_processor = ProjectionProcessor()
        self.grep_processor = GrepProcessor()
        # Store context managers and tasks to keep connections alive
        self._server_contexts: Dict[str, Any] = {}
        self._connection_tasks: Dict[str, asyncio.Task] = {}
        # Metrics tracking
        self.metrics = ConnectionPoolMetrics()

        # Register handlers
        self._register_handlers()

    def _enhance_tool_schema(self, input_schema: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Enhance tool input schema to include _meta parameter for projection and grep.

        Args:
            input_schema: Original tool input schema (dict or Pydantic model)

        Returns:
            Enhanced schema with _meta parameter
        """
        # Convert to dict if it's a Pydantic model
        if hasattr(input_schema, 'model_dump'):
            schema_dict = input_schema.model_dump()
        elif hasattr(input_schema, 'dict'):
            schema_dict = input_schema.dict()
        elif isinstance(input_schema, dict):
            schema_dict = input_schema
        else:
            # Fallback: try to convert to dict
            schema_dict = dict(input_schema) if input_schema else {}

        # Create a deep copy to avoid modifying the original
        enhanced_schema = json.loads(json.dumps(schema_dict))

        # Ensure it's a valid JSON Schema object
        if "type" not in enhanced_schema:
            enhanced_schema["type"] = "object"

        # Ensure properties exist
        if "properties" not in enhanced_schema:
            enhanced_schema["properties"] = {}

        # CRITICAL: Override additionalProperties to True to allow _meta parameter
        # Even if the original schema has additionalProperties: false, we need to allow _meta
        enhanced_schema["additionalProperties"] = True

        # Add _meta parameter
        enhanced_schema["properties"]["_meta"] = {
            "type": "object",
            "description": "Optional metadata for field projection and grep filtering. Use this to optimize token usage and filter results.",
            "properties": {
                "projection": {
                    "type": "object",
                    "description": "Field projection to include/exclude specific fields from the response. Reduces token usage by 85-95% in many cases.",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["include", "exclude", "view"],
                            "description": "Projection mode: 'include' returns only specified fields, 'exclude' returns all except specified fields, 'view' uses named preset views"
                        },
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of field paths to include/exclude. Supports nested paths like 'user.name' or 'users.email' for arrays."
                        },
                        "view": {
                            "type": "string",
                            "description": "Optional named view preset (used with mode='view')"
                        }
                    },
                    "required": ["mode", "fields"]
                },
                "grep": {
                    "type": "object",
                    "description": "Advanced search to filter tool outputs. Supports multiple modes for different search strategies.",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["regex", "bm25", "fuzzy", "context", "structure"],
                            "default": "regex",
                            "description": (
                                "Search mode:\n"
                                "- 'regex': Traditional regex search (default)\n"
                                "- 'bm25': Relevance-ranked search (returns top-K most relevant chunks)\n"
                                "- 'fuzzy': Approximate/fuzzy matching (handles typos)\n"
                                "- 'context': Extract with intelligent context (paragraphs/sections)\n"
                                "- 'structure': Navigate data structure without loading full content"
                            )
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex for 'regex' mode, query text for others)"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query (used for 'bm25' mode, alternative to 'pattern')"
                        },
                        "topK": {
                            "type": "number",
                            "default": 5,
                            "description": "Number of top results to return (for 'bm25' mode)"
                        },
                        "threshold": {
                            "type": "number",
                            "default": 0.7,
                            "description": "Similarity threshold 0-1 (for 'fuzzy' mode)"
                        },
                        "contextType": {
                            "type": "string",
                            "enum": ["paragraph", "section", "sentence", "lines"],
                            "default": "paragraph",
                            "description": "Context extraction type (for 'context' mode)"
                        },
                        "chunkSize": {
                            "type": "number",
                            "default": 500,
                            "description": "Size of chunks for analysis (for 'bm25' mode)"
                        },
                        "maxDepth": {
                            "type": "number",
                            "default": 3,
                            "description": "Maximum depth for structure navigation (for 'structure' mode)"
                        },
                        "caseInsensitive": {
                            "type": "boolean",
                            "default": False,
                            "description": "Case-insensitive matching (for 'regex' mode)"
                        },
                        "multiline": {
                            "type": "boolean",
                            "default": False,
                            "description": "Enable multiline patterns (for 'regex' mode)"
                        },
                        "maxMatches": {
                            "type": "number",
                            "description": "Maximum number of matches to return"
                        },
                        "contextLines": {
                            "type": "object",
                            "description": "Include context lines around matches (for 'regex' mode)",
                            "properties": {
                                "before": {
                                    "type": "number",
                                    "default": 0
                                },
                                "after": {
                                    "type": "number",
                                    "default": 0
                                },
                                "both": {
                                    "type": "number",
                                    "default": 0
                                }
                            }
                        },
                        "target": {
                            "type": "string",
                            "enum": ["content", "structuredContent"],
                            "default": "content",
                            "description": "Where to search: 'content' for plain text, 'structuredContent' for JSON"
                        }
                    },
                    "required": []
                }
            },
            "additionalProperties": False
        }

        # Note: We don't add "_meta" to required fields since it's optional
        # The original required fields are preserved

        return enhanced_schema

    def _register_handlers(self):
        """Register MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Aggregate tools from all underlying servers."""
            all_tools = []

            logger.debug("list_tools called")
            logger.debug(f"underlying_servers keys: {list(self.underlying_servers.keys())}")
            logger.debug(f"tools_cache keys: {list(self.tools_cache.keys())}")
            
            # Add a special proxy capabilities tool first
            proxy_capabilities_tool = Tool(
                name="proxy_get_capabilities",
                description=(
                    "Get MCP-RLM-Proxy advanced search capabilities and usage guide.\n\n"
                    "This proxy enhances ALL tools with:\n"
                    "- Field projection (filter fields, 85-95% token savings)\n"
                    "- Advanced search modes:\n"
                    "  • BM25: Relevance-ranked search (top-K most relevant chunks)\n"
                    "  • Fuzzy: Approximate matching (handles typos/variations)\n"
                    "  • Context: Extract paragraphs/sections (not just lines)\n"
                    "  • Structure: Navigate metadata without loading data (99.9% savings)\n"
                    "  • Regex: Traditional pattern matching (default)\n\n"
                    "Call this tool to see examples and learn how to use these features.\n\n"
                    "**Key principle**: Filter at proxy BEFORE data enters your context!"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "show_examples": {
                            "type": "boolean",
                            "description": "Show usage examples",
                            "default": True
                        }
                    }
                }
            )
            all_tools.append(proxy_capabilities_tool)

            # First, process cached tools
            for server_name, cached_tools in self.tools_cache.items():
                logger.debug(f"Using {len(cached_tools)} cached tools from {server_name}")
                for tool in cached_tools:
                    # Enhance schema with _meta parameter
                    enhanced_schema = self._enhance_tool_schema(tool.inputSchema)
                    
                    # Create enhanced description
                    enhanced_description = tool.description or ""
                    if not enhanced_description.endswith('\n'):
                        enhanced_description += '\n'
                    enhanced_description += (
                        f"\n**Server**: {server_name} | **Call as**: `{server_name}_{tool.name}`\n\n"
                        f"**💡 Proxy Enhancement**: Add `_meta` parameter for:\n"
                        f"- **Projection**: Filter fields (85-95% token savings)\n"
                        f"- **BM25 Search**: Relevance ranking (99%+ savings)\n"
                        f"- **Fuzzy Match**: Handle typos (98%+ savings)\n"
                        f"- **Structure Nav**: Explore without loading (99.9%+ savings)\n\n"
                        f"💡 Tip: Call `proxy_get_capabilities` to see examples!\n"
                    )

                    prefixed_tool = Tool(
                        name=f"{server_name}_{tool.name}",
                        description=enhanced_description,
                        inputSchema=enhanced_schema,
                    )
                    all_tools.append(prefixed_tool)

            # Parallelize tool listing from servers with cache misses or empty cache
            servers_to_fetch = [
                (server_name, session)
                for server_name, session in self.underlying_servers.items()
                if server_name not in self.tools_cache or len(self.tools_cache.get(server_name, [])) == 0
            ]

            if servers_to_fetch:
                logger.debug(f"Fetching tools from {len(servers_to_fetch)} server(s) in parallel")

                async def fetch_tools_from_server(server_name: str, session: ClientSession) -> tuple[str, List[Tool]]:
                    """Fetch tools from a single server."""
                    try:
                        logger.debug(f"Fetching tools from {server_name} session (cache miss or empty)")
                        tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                        logger.debug(f"Got {len(tools_result.tools)} tools from {server_name} session")
                        return server_name, tools_result.tools
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout fetching tools from {server_name} (10s)")
                        return server_name, []
                    except Exception as e:
                        logger.error(f"Error listing tools from {server_name}: {e}", exc_info=True)
                        return server_name, []

                # Fetch tools from all servers in parallel
                fetch_tasks = [
                    fetch_tools_from_server(server_name, session)
                    for server_name, session in servers_to_fetch
                ]
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                # Process results and add tools
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Exception during parallel tool fetch: {result}", exc_info=True)
                        continue

                    server_name, tools = result
                    if tools:
                        self.tools_cache[server_name] = tools
                        logger.info(f"Loaded {len(tools)} tools from {server_name}")
                        if tools:
                            tool_names = [t.name for t in tools[:5]]
                            logger.debug(f"Sample tools from {server_name}: {tool_names}{'...' if len(tools) > 5 else ''}")

                        # Add tools with enhanced schemas
                        for tool in tools:
                            enhanced_schema = self._enhance_tool_schema(tool.inputSchema)
                            
                            # Create enhanced description
                            enhanced_description = tool.description or ""
                            if not enhanced_description.endswith('\n'):
                                enhanced_description += '\n'
                            enhanced_description += (
                                f"\n**Note**: This tool is from the '{server_name}' server. "
                                f"Call it as '{server_name}_{tool.name}'. "
                                f"You can add '_meta' parameter for field projection or grep filtering."
                            )
                            
                            prefixed_tool = Tool(
                                name=f"{server_name}_{tool.name}",
                                description=enhanced_description,
                                inputSchema=enhanced_schema,
                            )
                            all_tools.append(prefixed_tool)
                    else:
                        logger.warning(f"{server_name} returned 0 tools")

            logger.debug(f"Returning {len(all_tools)} total tools")
            return all_tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[Content]:
            """Intercept tool calls, forward to underlying servers, and apply transformations."""
            logger.debug(f"call_tool called: {name}")
            
            # Handle special proxy capabilities tool
            if name == "proxy_get_capabilities":
                return self._handle_capabilities_request(arguments)

            # Validate arguments
            if not isinstance(arguments, dict):
                raise ValueError(f"Arguments must be a dictionary, got: {type(arguments)}")

            # Extract meta from arguments (following the _meta convention from the discussion)
            meta = arguments.pop("_meta", None) if isinstance(arguments, dict) else None
            if meta:
                logger.debug(f"Meta found: {meta}")
                # Validate meta structure
                if not isinstance(meta, dict):
                    raise ValueError("_meta must be a dictionary")

            # Parse server_tool name (using underscore separator)
            if "_" not in name:
                raise ValueError(f"Tool name must be in format 'server_tool', got: {name}")

            # Try to match known server names from the start
            # This handles tool names with underscores (e.g., playwright_browser_run)
            server_name = None
            tool_name = None
            
            for known_server in self.underlying_servers.keys():
                if name.startswith(known_server + "_"):
                    server_name = known_server
                    tool_name = name[len(known_server) + 1:]  # +1 to skip the underscore
                    break
            
            # Fallback to old method if no known server matched
            if server_name is None:
                parts = name.rsplit("_", 1)
                if len(parts) != 2:
                    raise ValueError(f"Tool name must be in format 'server_tool', got: {name}")
                server_name, tool_name = parts
            
            logger.debug(f"Parsed: server={server_name}, tool={tool_name}")

            if server_name not in self.underlying_servers:
                available = list(self.underlying_servers.keys())
                logger.error(f"Unknown server: {server_name}. Available: {available}")
                
                # Provide helpful error message
                available_str = ', '.join(available) if available else 'none'
                error_msg = (
                    f"Unknown server: '{server_name}'. Available servers: {available_str}.\n"
                    f"\n"
                    f"Tool name format: {{server_name}}_{{tool_name}}\n"
                    f"Note: Tool names can contain underscores (e.g., 'server_tool_name')\n"
                    f"\n"
                    f"Attempted to call: {name}\n"
                    f"Parsed as: server='{server_name}', tool='{tool_name}'\n"
                    f"\n"
                    f"If this parsing is wrong, the issue is likely:\n"
                    f"1. Server '{server_name}' doesn't exist in configuration\n"
                    f"2. Tool name should start with one of: {available_str}\n"
                    f"\n"
                    f"Common mistakes:\n"
                    f"- Calling 'browser_run' instead of 'playwright_browser_run'\n"
                    f"- Calling 'read_file' instead of 'filesystem_read_file'\n"
                    f"\n"
                    f"Suggestions based on your call '{name}':\n"
                )
                
                # Suggest corrections for each available server
                for avail_server in available:
                    # If the tool name contains the available server name, suggest it
                    if avail_server in name:
                        suggested_tool = name
                        if not name.startswith(avail_server + "_"):
                            suggested_tool = f"{avail_server}_{name}"
                        error_msg += f"  - {suggested_tool}\n"
                
                error_msg += (
                    f"\n"
                    f"💡 Best practice: Call list_tools() first to see all available tool names.\n"
                    f"   Available servers: {available_str}"
                )
                
                raise ValueError(error_msg)

            session = self.underlying_servers[server_name]
            logger.debug(f"Got session for {server_name}, calling tool {tool_name}")

            # Extract original tool from cache
            original_tool = None
            if server_name in self.tools_cache:
                for tool in self.tools_cache[server_name]:
                    if tool.name == tool_name:
                        original_tool = tool
                        break

            # Track original content size for token savings calculation
            original_size = 0

            # Call underlying tool (arguments now have _meta removed) with timeout
            try:
                logger.debug(f"Calling tool {tool_name} on {server_name} with timeout 60s")
                result = await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=60.0)
                logger.debug("Tool call completed successfully")
            except asyncio.TimeoutError:
                error_msg = f"Timeout calling tool {tool_name} on {server_name} (60s)"
                logger.error(error_msg)
                return [TextContent(type="text", text=f"Error: {error_msg}")]
            except Exception as e:
                error_msg = f"Error calling tool {tool_name} on {server_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [TextContent(type="text", text=f"Error: {error_msg}")]

            # Extract content from result
            content = result.content if hasattr(result, "content") else []

            # Calculate original size
            for item in content:
                if isinstance(item, TextContent):
                    original_size += len(item.text)

            # Apply transformations with error handling
            transformation_meta = {}
            if meta:
                # Apply projection
                if "projection" in meta:
                    try:
                        projection_spec = meta["projection"]
                        if not isinstance(projection_spec, dict):
                            raise ValueError("projection must be a dictionary")
                        mode = projection_spec.get("mode", "include")
                        if mode not in ["include", "exclude", "view"]:
                            raise ValueError(f"Invalid projection mode: {mode}. Must be 'include', 'exclude', or 'view'")

                        content = self.projection_processor.project_content(
                            content, projection_spec
                        )
                        transformation_meta["projection"] = {
                            "applied": True,
                            "mode": mode,
                        }
                        logger.debug(f"Applied projection with mode: {mode}")
                    except Exception as e:
                        error_msg = f"Error applying projection: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        return [TextContent(type="text", text=f"Error: {error_msg}")]

                # Apply grep
                if "grep" in meta:
                    try:
                        grep_spec = meta["grep"]
                        if not isinstance(grep_spec, dict):
                            raise ValueError("grep must be a dictionary")
                        if "pattern" not in grep_spec:
                            raise ValueError("grep must include a 'pattern' field")

                        content = self.grep_processor.apply_grep(content, grep_spec)
                        transformation_meta["grep"] = {
                            "applied": True,
                            "pattern": grep_spec.get("pattern"),
                        }
                        logger.debug(f"Applied grep with pattern: {grep_spec.get('pattern')}")
                    except Exception as e:
                        error_msg = f"Error applying grep: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        return [TextContent(type="text", text=f"Error: {error_msg}")]

            # Calculate token savings
            new_size = sum(len(item.text) for item in content if isinstance(item, TextContent))
            if original_size > 0:
                savings_percent = ((original_size - new_size) / original_size) * 100
                transformation_meta["token_savings"] = {
                    "original_size": original_size,
                    "new_size": new_size,
                    "savings_percent": round(savings_percent, 2),
                }
                logger.info(f"Token savings: {original_size} → {new_size} tokens ({savings_percent:.1f}% reduction)")
            
            # Record metrics
            used_projection = "projection" in meta if meta else False
            used_grep = "grep" in meta if meta else False
            self.metrics.record_call(original_size, new_size, used_projection, used_grep)

            return content
    
    def _handle_capabilities_request(self, arguments: dict) -> List[Content]:
        """Handle the proxy_get_capabilities tool call."""
        show_examples = arguments.get("show_examples", True)
        
        response = """# MCP-RLM-Proxy Advanced Capabilities

## Overview
This proxy enhances ALL tools with advanced search and filtering capabilities.
Use the `_meta` parameter to access these features.

## Key Features

### 1. Field Projection (85-95% token savings)
Extract only specific fields from responses.

Example:
{
  "path": "/data/users.json",
  "_meta": {
    "projection": {
      "mode": "include",
      "fields": ["users.name", "users.email"]
    }
  }
}

### 2. Advanced Search Modes

#### BM25 Relevance Ranking (99%+ token savings)
Returns top-K most relevant chunks instead of loading everything.

{
  "_meta": {
    "grep": {
      "mode": "bm25",
      "query": "database connection error timeout",
      "topK": 5
    }
  }
}

#### Fuzzy Matching (98%+ token savings)
Handles typos and variations.

{
  "_meta": {
    "grep": {
      "mode": "fuzzy",
      "pattern": "MacBook",
      "threshold": 0.7
    }
  }
}

#### Context Extraction (95%+ token savings)
Get paragraphs/sections, not just lines.

{
  "_meta": {
    "grep": {
      "mode": "context",
      "pattern": "error",
      "contextType": "paragraph"
    }
  }
}

#### Structure Navigation (99.9%+ token savings)
Explore data structure WITHOUT loading content!

{
  "_meta": {
    "grep": {
      "mode": "structure",
      "maxDepth": 3
    }
  }
}

Returns metadata: types, sizes, field names, samples - NOT full data.

#### Regex Search (default, 95%+ token savings)
Traditional pattern matching.

{
  "_meta": {
    "grep": {
      "mode": "regex",  // or omit for default
      "pattern": "ERROR|FATAL",
      "caseInsensitive": true
    }
  }
}

## Recommended Workflow for AI Agents

### Step 1: Discover Structure (50 tokens)
{
  "_meta": {
    "grep": {
      "mode": "structure"
    }
  }
}

### Step 2: Search for Relevance (1,500 tokens)
{
  "_meta": {
    "grep": {
      "mode": "bm25",
      "query": "what you're looking for",
      "topK": 5
    }
  }
}

### Step 3: Extract Specific Fields (500 tokens)
{
  "_meta": {
    "projection": {
      "mode": "include",
      "fields": ["specific", "fields"]
    }
  }
}

Total: ~2,000 tokens vs 500,000+ (99.6% savings!)

## Why This Matters

❌ **Without proxy**: Load full output → 50,000 tokens → Agent context polluted
✅ **With proxy**: Filter at source → 500 tokens → Clean agent context

## Key Principle

**Filter BEFORE data enters your context, not after!**

The proxy does the heavy lifting so you don't waste tokens and context.

## More Information

See detailed documentation:
- Advanced Search Guide: docs/ADVANCED_SEARCH.md
- AI Agent Guide: docs/AI_AGENT_GUIDE.md

## Available Modes Summary

| Mode | Use When | Token Savings |
|------|----------|---------------|
| structure | Don't know data format | 99.9%+ |
| bm25 | Know what, not where | 99%+ |
| fuzzy | Handle typos/variations | 98%+ |
| context | Need full paragraphs | 95%+ |
| regex | Know exact pattern | 95%+ |
"""
        
        if show_examples:
            response += "\n\n## Quick Example\n\n"
            response += "Instead of:\n"
            response += '  call_tool("filesystem_read_file", {"path": "log.txt"})\n'
            response += "  → Returns 280,000 tokens\n\n"
            response += "Do this:\n"
            response += '  call_tool("filesystem_read_file", {\n'
            response += '    "path": "log.txt",\n'
            response += '    "_meta": {\n'
            response += '      "grep": {\n'
            response += '        "mode": "bm25",\n'
            response += '        "query": "error database",\n'
            response += '        "topK": 3\n'
            response += '      }\n'
            response += '    }\n'
            response += '  })\n'
            response += "  → Returns 1,500 tokens (99.5% savings!)\n"
        
        return [TextContent(type="text", text=response)]

    async def _connect_to_server_sync(self, server_name: str, server_params: StdioServerParameters):
        """Connect to a server synchronously and keep connection alive using background task."""
        logger.debug(f"_connect_to_server_sync called for {server_name}")

        # Use a background task to keep the connection alive
        # This allows the context manager to stay open
        connection_event = asyncio.Event()
        connection_error = [None]

        async def _keep_connection():
            """Background task to maintain connection."""
            try:
                logger.debug(f"Background task: Creating stdio_client for {server_name}...")
                logger.debug(f"Background task: server_params command={server_params.command}, args={server_params.args}")

                # Use stdio_client context manager - this properly isolates subprocess streams
                async with stdio_client(server_params) as (read_stream, write_stream):
                    logger.debug(f"Background task: Got streams for {server_name}")

                    # Use ClientSession as context manager (like direct test) but keep it alive
                    # by not exiting the context
                    logger.debug(f"Background task: Creating ClientSession for {server_name}...")
                    session_obj = ClientSession(read_stream, write_stream)
                    session = await session_obj.__aenter__()
                    # Store the session object for cleanup
                    self._server_contexts[server_name] = session_obj

                    try:
                        # Initialize with timeout
                        logger.debug(f"Background task: Initializing session for {server_name}...")
                        try:
                            init_result = await asyncio.wait_for(session.initialize(), timeout=30.0)
                            logger.info(f"Connected to underlying server: {server_name}")
                            if init_result.serverInfo:
                                logger.info(f"     Server: {init_result.serverInfo.name}, Version: {init_result.serverInfo.version}")
                        except asyncio.TimeoutError:
                            logger.error(f"Timeout initializing session for {server_name} (30s)")
                            connection_error[0] = Exception(f"Timeout connecting to {server_name}")
                            connection_event.set()
                            await session_obj.__aexit__(None, None, None)
                            return
                        except Exception as e:
                            logger.error(f"Exception during initialization: {e}", exc_info=True)
                            connection_error[0] = e
                            connection_event.set()
                            await session_obj.__aexit__(None, None, None)
                            return

                        # Store session immediately
                        self.underlying_servers[server_name] = session
                        logger.debug(f"Background task: Session stored for {server_name}")
                        
                        # Update metrics
                        self.metrics.connection_count += 1

                        # Pre-load tools
                        try:
                            logger.debug(f"Background task: Listing tools for {server_name}...")
                            tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                            logger.debug(f"Background task: Got {len(tools_result.tools)} tools from {server_name}")
                            logger.info(f"     Loaded {len(tools_result.tools)} tools from {server_name}")
                            self.tools_cache[server_name] = tools_result.tools
                            if tools_result.tools:
                                tool_names = [t.name for t in tools_result.tools[:5]]
                                logger.info(f"     Sample tools: {tool_names}{'...' if len(tools_result.tools) > 5 else ''}")
                            else:
                                logger.warning(f"{server_name} returned 0 tools")
                        except Exception as e:
                            logger.error(f"Could not list tools from {server_name}: {e}", exc_info=True)

                        # Signal that connection is ready
                        connection_event.set()
                        logger.debug(f"Background task: Connection ready for {server_name}")

                        # Keep connection alive by waiting (both contexts stay open)
                        try:
                            await asyncio.Event().wait()  # Wait forever
                        except asyncio.CancelledError:
                            logger.info(f"Connection to {server_name} cancelled")
                            if server_name in self.underlying_servers:
                                del self.underlying_servers[server_name]
                            if server_name in self._server_contexts:
                                try:
                                    await self._server_contexts[server_name].__aexit__(None, None, None)
                                except Exception:
                                    pass
                                del self._server_contexts[server_name]
                            raise
                    finally:
                        # Clean up session context if we exit
                        if server_name in self._server_contexts:
                            try:
                                await self._server_contexts[server_name].__aexit__(None, None, None)
                            except Exception:
                                pass
                            del self._server_contexts[server_name]

            except Exception as e:
                logger.error(f"Connection task failed for {server_name}: {e}", exc_info=True)
                connection_error[0] = e
                connection_event.set()
                if server_name in self.underlying_servers:
                    del self.underlying_servers[server_name]

        # Start background task
        task = asyncio.create_task(_keep_connection())
        self._connection_tasks[server_name] = task

        # Wait for connection to be established (with timeout)
        logger.debug(f"Waiting for connection to {server_name} to be established...")
        try:
            await asyncio.wait_for(connection_event.wait(), timeout=35.0)
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for connection to {server_name}")
            task.cancel()
            raise Exception(f"Timeout waiting for connection to {server_name}")

        # Check for errors
        if connection_error[0]:
            raise connection_error[0]

        logger.debug(f"Connection setup complete for {server_name}")

    async def initialize_underlying_servers(self):
        """Initialize connections to underlying MCP servers."""
        if not self.server_configs:
            logger.info("No underlying servers configured.")
            return

        logger.info(f"Initializing {len(self.server_configs)} underlying server(s)...")

        for config in self.server_configs:
            server_name = config["name"]
            command = config["command"]
            args = config.get("args", [])

            try:
                logger.info(f"Connecting to {server_name}... (command: {command}, args: {args})")
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=None,
                )

                # Connect synchronously - wait for it to complete
                logger.debug(f"Calling _connect_to_server_sync for {server_name}...")
                await self._connect_to_server_sync(server_name, server_params)
                logger.debug(f"Connection to {server_name} established")

            except Exception as e:
                logger.error(f"Failed to start connection to {server_name}: {e}", exc_info=True)
                self.metrics.failed_connections += 1

    async def cleanup(self):
        """Clean up connections to underlying servers."""
        try:
            logger.info("Cleaning up connections...")
            
            # Log metrics summary before cleanup
            self.metrics.log_summary()
            
            # Close all context managers
            for server_name, session_obj in list(self._server_contexts.items()):
                try:
                    await session_obj.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing {server_name}: {e}")
            self._server_contexts.clear()
            # Cancel connection tasks
            for server_name, task in list(self._connection_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self._connection_tasks.clear()
            self.underlying_servers.clear()
            self.tools_cache.clear()
        except Exception:
            pass  # Ignore errors during cleanup

    async def run(self):
        """Run the proxy server."""
        try:
            # Initialize underlying servers
            await self.initialize_underlying_servers()

            # Build capabilities with experimental features
            capabilities = ServerCapabilities.model_validate({
                "tools": {},
                "experimental": {
                    "projection": {
                        "supported": True,
                        "modes": ["include", "exclude", "view"],
                        "description": "Filter response fields to reduce token usage by 85-95%"
                    },
                    "grep": {
                        "supported": True,
                        "modes": ["regex", "bm25", "fuzzy", "context", "structure"],
                        "maxPatternLength": 1000,
                        "description": "Advanced search with multiple modes: regex (default), bm25 (relevance ranking), fuzzy (approximate), context (paragraph/section extraction), structure (metadata only)",
                        "features": {
                            "bm25_ranking": "Relevance-ranked search, returns top-K most relevant chunks",
                            "fuzzy_matching": "Approximate matching with configurable similarity threshold",
                            "context_extraction": "Extract paragraphs/sections/sentences containing matches",
                            "structure_navigation": "Explore data structure without loading content (99.9% token savings)",
                            "progressive_refinement": "Multi-step workflow: discover → search → refine"
                        }
                    },
                    "rlm_support": {
                        "supported": True,
                        "description": "Recursive Language Model principles implemented - treat outputs as external environment to explore programmatically",
                        "paper": "arXiv:2512.24601"
                    }
                },
            })

            # Run the server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions.model_validate({
                        "server_name": "mcp-proxy-server",
                        "server_version": "0.1.0",
                        "capabilities": capabilities.model_dump(),
                    }),
                )
        finally:
            # Clean up connections
            await self.cleanup()

