"""
MCP Proxy Server implementation.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Content, TextContent, Tool, ServerCapabilities

from mcp_proxy.processors import GrepProcessor, ProjectionProcessor


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

            print(f"[DEBUG] list_tools called", file=sys.stderr)
            print(f"[DEBUG] underlying_servers keys: {list(self.underlying_servers.keys())}", file=sys.stderr)
            print(f"[DEBUG] tools_cache keys: {list(self.tools_cache.keys())}", file=sys.stderr)

            # First, try to use cached tools
            for server_name, cached_tools in self.tools_cache.items():
                print(f"[DEBUG] Using {len(cached_tools)} cached tools from {server_name}", file=sys.stderr)
                for tool in cached_tools:
                    # Enhance schema with _meta parameter
                    enhanced_schema = self._enhance_tool_schema(tool.inputSchema)

                    # Debug: Verify _meta is in the schema
                    if isinstance(enhanced_schema, dict) and "properties" in enhanced_schema:
                        has_meta = "_meta" in enhanced_schema.get("properties", {})
                        print(f"[DEBUG] Tool {server_name}_{tool.name}: _meta in schema = {has_meta}", file=sys.stderr)
                        if has_meta:
                            print(f"[DEBUG]   _meta properties: {list(enhanced_schema['properties']['_meta'].get('properties', {}).keys())}", file=sys.stderr)

                    prefixed_tool = Tool(
                        name=f"{server_name}_{tool.name}",
                        description=tool.description or "",
                        inputSchema=enhanced_schema,
                    )

                    # Debug: Verify _meta is in the created Tool object
                    tool_schema = prefixed_tool.inputSchema
                    if hasattr(tool_schema, 'model_dump'):
                        tool_schema_dict = tool_schema.model_dump()
                    elif isinstance(tool_schema, dict):
                        tool_schema_dict = tool_schema
                    else:
                        tool_schema_dict = {}

                    if isinstance(tool_schema_dict, dict) and "properties" in tool_schema_dict:
                        has_meta_after = "_meta" in tool_schema_dict.get("properties", {})
                        print(f"[DEBUG] Tool {server_name}_{tool.name} after Tool() creation: _meta in schema = {has_meta_after}", file=sys.stderr)

                    all_tools.append(prefixed_tool)

            # Also try to get tools from active sessions (in case cache is stale or empty)
            for server_name, session in self.underlying_servers.items():
                if server_name not in self.tools_cache or len(self.tools_cache.get(server_name, [])) == 0:
                    try:
                        print(f"[DEBUG] Fetching tools from {server_name} session (cache miss or empty)", file=sys.stderr)
                        tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                        print(f"[DEBUG] Got {len(tools_result.tools)} tools from {server_name} session", file=sys.stderr)
                        # Prefix tool names with server name to avoid conflicts
                        # Use underscore separator instead of :: to avoid validation warnings
                        for tool in tools_result.tools:
                            # Enhance schema with _meta parameter
                            enhanced_schema = self._enhance_tool_schema(tool.inputSchema)

                            # Debug: Verify _meta is in the schema
                            if isinstance(enhanced_schema, dict) and "properties" in enhanced_schema:
                                has_meta = "_meta" in enhanced_schema.get("properties", {})
                                print(f"[DEBUG] Tool {server_name}_{tool.name}: _meta in schema = {has_meta}", file=sys.stderr)

                            prefixed_tool = Tool(
                                name=f"{server_name}_{tool.name}",
                                description=tool.description or "",
                                inputSchema=enhanced_schema,
                            )

                            # Debug: Verify _meta is in the created Tool object
                            tool_schema = prefixed_tool.inputSchema
                            if hasattr(tool_schema, 'model_dump'):
                                tool_schema_dict = tool_schema.model_dump()
                            elif isinstance(tool_schema, dict):
                                tool_schema_dict = tool_schema
                            else:
                                tool_schema_dict = {}

                            if isinstance(tool_schema_dict, dict) and "properties" in tool_schema_dict:
                                has_meta_after = "_meta" in tool_schema_dict.get("properties", {})
                                print(f"[DEBUG] Tool {server_name}_{tool.name} after Tool() creation: _meta in schema = {has_meta_after}", file=sys.stderr)

                            all_tools.append(prefixed_tool)
                        self.tools_cache[server_name] = tools_result.tools
                    except Exception as e:
                        print(f"[ERROR] Error listing tools from {server_name}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)

            print(f"[DEBUG] Returning {len(all_tools)} total tools", file=sys.stderr)
            return all_tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[Content]:
            """Intercept tool calls, forward to underlying servers, and apply transformations."""
            print(f"[DEBUG] call_tool called: {name}", file=sys.stderr)

            # Validate arguments
            if not isinstance(arguments, dict):
                raise ValueError(f"Arguments must be a dictionary, got: {type(arguments)}")

            # Extract meta from arguments (following the _meta convention from the discussion)
            meta = arguments.pop("_meta", None) if isinstance(arguments, dict) else None
            if meta:
                print(f"[DEBUG] Meta found: {meta}", file=sys.stderr)
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
            print(f"[DEBUG] Parsed: server={server_name}, tool={tool_name}", file=sys.stderr)

            if server_name not in self.underlying_servers:
                available = list(self.underlying_servers.keys())
                print(f"[ERROR] Unknown server: {server_name}. Available: {available}", file=sys.stderr)
                raise ValueError(f"Unknown server: {server_name}. Available servers: {', '.join(available) if available else 'none'}")

            session = self.underlying_servers[server_name]
            print(f"[DEBUG] Got session for {server_name}, calling tool {tool_name}", file=sys.stderr)

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
                print(f"[DEBUG] Calling tool {tool_name} on {server_name} with timeout 60s", file=sys.stderr)
                result = await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=60.0)
                print(f"[DEBUG] Tool call completed successfully", file=sys.stderr)
            except asyncio.TimeoutError:
                error_msg = f"Timeout calling tool {tool_name} on {server_name} (60s)"
                print(f"[ERROR] {error_msg}", file=sys.stderr)
                return [TextContent(type="text", text=f"Error: {error_msg}")]
            except Exception as e:
                error_msg = f"Error calling tool {tool_name} on {server_name}: {str(e)}"
                print(f"[ERROR] {error_msg}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
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
                    except Exception as e:
                        error_msg = f"Error applying projection: {str(e)}"
                        print(f"[ERROR] {error_msg}", file=sys.stderr)
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
                    except Exception as e:
                        error_msg = f"Error applying grep: {str(e)}"
                        print(f"[ERROR] {error_msg}", file=sys.stderr)
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
        print(f"[DEBUG] _connect_to_server_sync called for {server_name}", file=sys.stderr)

        # Use a background task to keep the connection alive
        # This allows the context manager to stay open
        connection_event = asyncio.Event()
        connection_error = [None]

        async def _keep_connection():
            """Background task to maintain connection."""
            try:
                print(f"[DEBUG] Background task: Creating stdio_client for {server_name}...", file=sys.stderr)
                print(f"[DEBUG] Background task: server_params command={server_params.command}, args={server_params.args}", file=sys.stderr)

                # Use stdio_client context manager - this properly isolates subprocess streams
                async with stdio_client(server_params) as (read_stream, write_stream):
                    print(f"[DEBUG] Background task: Got streams for {server_name}", file=sys.stderr)

                    # Use ClientSession as context manager (like direct test) but keep it alive
                    # by not exiting the context
                    print(f"[DEBUG] Background task: Creating ClientSession for {server_name}...", file=sys.stderr)
                    session_obj = ClientSession(read_stream, write_stream)
                    session = await session_obj.__aenter__()
                    # Store the session object for cleanup
                    self._server_contexts[server_name] = session_obj

                    try:
                        # Initialize with timeout
                        print(f"[DEBUG] Background task: Initializing session for {server_name}...", file=sys.stderr)
                        try:
                            init_result = await asyncio.wait_for(session.initialize(), timeout=30.0)
                            print(f"[OK] Connected to underlying server: {server_name}", file=sys.stderr)
                            if init_result.serverInfo:
                                print(f"     Server: {init_result.serverInfo.name}, Version: {init_result.serverInfo.version}", file=sys.stderr)
                        except asyncio.TimeoutError:
                            print(f"[ERROR] Timeout initializing session for {server_name} (30s)", file=sys.stderr)
                            connection_error[0] = Exception(f"Timeout connecting to {server_name}")
                            connection_event.set()
                            await session_obj.__aexit__(None, None, None)
                            return
                        except Exception as e:
                            print(f"[ERROR] Exception during initialization: {e}", file=sys.stderr)
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            connection_error[0] = e
                            connection_event.set()
                            await session_obj.__aexit__(None, None, None)
                            return

                        # Store session immediately
                        self.underlying_servers[server_name] = session
                        print(f"[DEBUG] Background task: Session stored for {server_name}", file=sys.stderr)

                        # Pre-load tools
                        try:
                            print(f"[DEBUG] Background task: Listing tools for {server_name}...", file=sys.stderr)
                            tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                            print(f"[DEBUG] Background task: Got {len(tools_result.tools)} tools from {server_name}", file=sys.stderr)
                            print(f"     Loaded {len(tools_result.tools)} tools from {server_name}", file=sys.stderr)
                            self.tools_cache[server_name] = tools_result.tools
                            if tools_result.tools:
                                tool_names = [t.name for t in tools_result.tools[:5]]
                                print(f"     Sample tools: {tool_names}{'...' if len(tools_result.tools) > 5 else ''}", file=sys.stderr)
                            else:
                                print(f"[WARNING] {server_name} returned 0 tools", file=sys.stderr)
                        except Exception as e:
                            print(f"[ERROR] Could not list tools from {server_name}: {e}", file=sys.stderr)
                            import traceback
                            traceback.print_exc(file=sys.stderr)

                        # Signal that connection is ready
                        connection_event.set()
                        print(f"[DEBUG] Background task: Connection ready for {server_name}", file=sys.stderr)

                        # Keep connection alive by waiting (both contexts stay open)
                        try:
                            await asyncio.Event().wait()  # Wait forever
                        except asyncio.CancelledError:
                            print(f"[INFO] Connection to {server_name} cancelled", file=sys.stderr)
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
                print(f"[ERROR] Connection task failed for {server_name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                connection_error[0] = e
                connection_event.set()
                if server_name in self.underlying_servers:
                    del self.underlying_servers[server_name]

        # Start background task
        task = asyncio.create_task(_keep_connection())
        self._connection_tasks[server_name] = task

        # Wait for connection to be established (with timeout)
        print(f"[DEBUG] Waiting for connection to {server_name} to be established...", file=sys.stderr)
        try:
            await asyncio.wait_for(connection_event.wait(), timeout=35.0)
        except asyncio.TimeoutError:
            print(f"[ERROR] Timeout waiting for connection to {server_name}", file=sys.stderr)
            task.cancel()
            raise Exception(f"Timeout waiting for connection to {server_name}")

        # Check for errors
        if connection_error[0]:
            raise connection_error[0]

        print(f"[DEBUG] Connection setup complete for {server_name}", file=sys.stderr)

    async def initialize_underlying_servers(self):
        """Initialize connections to underlying MCP servers."""
        if not self.server_configs:
            print("No underlying servers configured.")
            return

        print(f"Initializing {len(self.server_configs)} underlying server(s)...")

        for config in self.server_configs:
            server_name = config["name"]
            command = config["command"]
            args = config.get("args", [])

            try:
                print(f"Connecting to {server_name}... (command: {command}, args: {args})")
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=None,
                )

                # Connect synchronously - wait for it to complete
                print(f"[DEBUG] Calling _connect_to_server_sync for {server_name}...", file=sys.stderr)
                await self._connect_to_server_sync(server_name, server_params)
                print(f"[DEBUG] Connection to {server_name} established", file=sys.stderr)

            except Exception as e:
                print(f"[ERROR] Failed to start connection to {server_name}: {e}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """Clean up connections to underlying servers."""
        try:
            print("\n[INFO] Cleaning up connections...", file=sys.stderr)
            # Close all context managers
            for server_name, session_obj in list(self._server_contexts.items()):
                try:
                    await session_obj.__aexit__(None, None, None)
                except Exception as e:
                    print(f"[WARNING] Error closing {server_name}: {e}", file=sys.stderr)
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

