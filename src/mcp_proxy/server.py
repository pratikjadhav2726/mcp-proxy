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
                    "description": "Grep-like search to filter tool outputs using regex patterns. Extracts only matching content.",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for in the tool output"
                        },
                        "caseInsensitive": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to perform case-insensitive matching"
                        },
                        "multiline": {
                            "type": "boolean",
                            "default": False,
                            "description": "Enable multiline pattern matching. When true, '.' matches newlines and patterns can span multiple lines (similar to grep -P or re.DOTALL)"
                        },
                        "maxMatches": {
                            "type": "number",
                            "description": "Maximum number of matches to return"
                        },
                        "contextLines": {
                            "type": "object",
                            "description": "Include context lines around matches (similar to grep -A, -B, -C options)",
                            "properties": {
                                "before": {
                                    "type": "number",
                                    "default": 0,
                                    "description": "Number of lines to include before each match (grep -B)"
                                },
                                "after": {
                                    "type": "number",
                                    "default": 0,
                                    "description": "Number of lines to include after each match (grep -A)"
                                },
                                "both": {
                                    "type": "number",
                                    "default": 0,
                                    "description": "Number of lines to include both before and after each match (grep -C). If specified, overrides 'before' and 'after'"
                                }
                            }
                        },
                        "target": {
                            "type": "string",
                            "enum": ["content", "structuredContent"],
                            "default": "content",
                            "description": "Where to search: 'content' for plain text, 'structuredContent' for JSON data"
                        }
                    },
                    "required": ["pattern"]
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

            # First, process cached tools
            for server_name, cached_tools in self.tools_cache.items():
                logger.debug(f"Using {len(cached_tools)} cached tools from {server_name}")
                for tool in cached_tools:
                    # Enhance schema with _meta parameter
                    enhanced_schema = self._enhance_tool_schema(tool.inputSchema)

                    prefixed_tool = Tool(
                        name=f"{server_name}_{tool.name}",
                        description=tool.description or "",
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
                            prefixed_tool = Tool(
                                name=f"{server_name}_{tool.name}",
                                description=tool.description or "",
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

            # Split on last underscore to handle tool names that might contain underscores
            parts = name.rsplit("_", 1)
            if len(parts) != 2:
                raise ValueError(f"Tool name must be in format 'server_tool', got: {name}")
            server_name, tool_name = parts
            logger.debug(f"Parsed: server={server_name}, tool={tool_name}")

            if server_name not in self.underlying_servers:
                available = list(self.underlying_servers.keys())
                logger.error(f"Unknown server: {server_name}. Available: {available}")
                raise ValueError(f"Unknown server: {server_name}. Available servers: {', '.join(available) if available else 'none'}")

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

            return content

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

    async def cleanup(self):
        """Clean up connections to underlying servers."""
        try:
            logger.info("Cleaning up connections...")
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
                    },
                    "grep": {
                        "supported": True,
                        "maxPatternLength": 1000,
                    },
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

