"""
MCP Proxy Server

A proxy intermediary that sits between MCP clients and underlying MCP tool servers.
Implements field projection and grep search capabilities to optimize token usage,
privacy, and performance.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Content,
    ImageContent,
    TextContent,
    Tool,
    ServerCapabilities,
)
import yaml


class ProjectionProcessor:
    """Handles field projection operations on tool responses."""

    @staticmethod
    def apply_projection(
        data: Any, projection: Dict[str, Any], path: str = ""
    ) -> Any:
        """
        Apply field projection to data based on projection spec.

        Args:
            data: The data to project (dict, list, or primitive)
            projection: Projection specification with 'mode' and 'fields'
            path: Current JSON path for nested structures

        Returns:
            Projected data
        """
        mode = projection.get("mode", "include")
        fields = projection.get("fields", [])

        # Handle lists by applying projection to each item
        if isinstance(data, list):
            return [
                ProjectionProcessor.apply_projection(item, projection, path)
                for item in data
            ]

        # Handle dictionaries
        if not isinstance(data, dict):
            return data

        if mode == "include":
            # Group fields by parent path for arrays (e.g., "users.name", "users.email" -> "users": ["name", "email"])
            array_projections = {}  # parent_key -> list of nested fields
            regular_fields = []
            
            for field in fields:
                if "." in field:
                    parts = field.split(".")
                    parent_key = parts[0]
                    nested_field = ".".join(parts[1:])
                    
                    # Check if parent is an array
                    if parent_key in data and isinstance(data[parent_key], list):
                        if parent_key not in array_projections:
                            array_projections[parent_key] = []
                        array_projections[parent_key].append(nested_field)
                    else:
                        regular_fields.append(field)
                else:
                    regular_fields.append(field)
            
            result = {}
            
            # Handle array projections (e.g., "users": ["name", "email"])
            for parent_key, nested_fields in array_projections.items():
                if parent_key in data:
                    nested_projection = {
                        "mode": "include",
                        "fields": nested_fields
                    }
                    result[parent_key] = ProjectionProcessor.apply_projection(
                        data[parent_key], nested_projection, parent_key
                    )
            
            # Handle regular fields (direct keys or nested dict paths)
            for field in regular_fields:
                if field in data:
                    value = data[field]
                    # Recursively apply projection to nested objects
                    if isinstance(value, (dict, list)):
                        result[field] = ProjectionProcessor.apply_projection(
                            value, projection, f"{path}.{field}" if path else field
                        )
                    else:
                        result[field] = value
                # Handle nested paths like "user.name" (for dicts, not arrays)
                elif "." in field:
                    parts = field.split(".")
                    current = data
                    # Navigate to the nested field
                    for part in parts[:-1]:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            break
                    else:
                        if isinstance(current, dict) and parts[-1] in current:
                            # Create nested structure preserving hierarchy
                            nested = result
                            for part in parts[:-1]:
                                if part not in nested:
                                    nested[part] = {}
                                nested = nested[part]
                            value = current[parts[-1]]
                            # Recursively apply projection to nested values
                            if isinstance(value, (dict, list)):
                                nested[parts[-1]] = ProjectionProcessor.apply_projection(
                                    value, projection, field
                                )
                            else:
                                nested[parts[-1]] = value
            return result

        elif mode == "exclude":
            # Exclude specified fields
            result = {}
            for key, value in data.items():
                # Check if this key should be excluded directly
                if key in fields:
                    continue
                
                # Check if this key is part of a nested exclusion path
                # e.g., if excluding "user.password", we keep "user" but exclude "password" inside it
                should_exclude_key = False
                nested_exclusions = []
                for field in fields:
                    if field == key:
                        should_exclude_key = True
                        break
                    elif field.startswith(f"{key}."):
                        # This is a nested exclusion - keep the key but exclude nested field
                        nested_exclusions.append(field[len(key) + 1:])  # Remove "key." prefix
                
                if should_exclude_key:
                    continue
                
                # If there are nested exclusions, apply them recursively
                if nested_exclusions:
                    nested_projection = {
                        "mode": "exclude",
                        "fields": nested_exclusions
                    }
                    if isinstance(value, (dict, list)):
                        result[key] = ProjectionProcessor.apply_projection(
                            value, nested_projection, f"{path}.{key}" if path else key
                        )
                    else:
                        result[key] = value
                else:
                    # No exclusions for this key - include it
                    # Recursively apply exclusion to nested structures
                    if isinstance(value, (dict, list)):
                        result[key] = ProjectionProcessor.apply_projection(
                            value, projection, f"{path}.{key}" if path else key
                        )
                    else:
                        result[key] = value
            
            return result

        elif mode == "view":
            # Named preset view (simplified - could be extended)
            view_name = projection.get("view", fields[0] if fields else "default")
            # For now, treat as include
            return ProjectionProcessor.apply_projection(
                data, {"mode": "include", "fields": fields}
            )

        return data

    @staticmethod
    def project_content(
        content: List[Content], projection: Dict[str, Any]
    ) -> List[Content]:
        """
        Apply projection to MCP Content objects.

        Args:
            content: List of Content objects
            projection: Projection specification

        Returns:
            List of projected Content objects
        """
        projected = []
        for item in content:
            if isinstance(item, TextContent):
                # Try to parse as JSON for structured content
                try:
                    data = json.loads(item.text)
                    if isinstance(data, dict):
                        projected_data = ProjectionProcessor.apply_projection(
                            data, projection
                        )
                        projected.append(
                            TextContent(
                                type="text",
                                text=json.dumps(projected_data, indent=2),
                            )
                        )
                    else:
                        projected.append(item)
                except json.JSONDecodeError:
                    # Not JSON, return as-is (can't project plain text)
                    projected.append(item)
            elif isinstance(item, ImageContent):
                # Images can't be projected
                projected.append(item)
            else:
                projected.append(item)
        return projected


class GrepProcessor:
    """Handles grep-like search operations on tool outputs."""

    @staticmethod
    def apply_grep(
        content: List[Content], grep_spec: Dict[str, Any]
    ) -> List[Content]:
        """
        Apply grep search to content.

        Args:
            content: List of Content objects
            grep_spec: Grep specification with 'pattern', 'caseInsensitive', etc.

        Returns:
            Filtered Content objects
        """
        pattern = grep_spec.get("pattern", "")
        case_insensitive = grep_spec.get("caseInsensitive", False)
        max_matches = grep_spec.get("maxMatches")
        target = grep_spec.get("target", "content")  # 'content' or 'structuredContent'

        if not pattern:
            return content

        # Validate and compile regex pattern
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            # Invalid regex pattern - return error message as content
            return [
                TextContent(
                    type="text",
                    text=f"Error: Invalid regex pattern '{pattern}': {str(e)}",
                )
            ]

        filtered = []
        match_count = 0

        for item in content:
            if isinstance(item, TextContent):
                text = item.text

                # Try to parse as JSON for structured content search
                if target == "structuredContent":
                    try:
                        data = json.loads(text)
                        # Search in structured data
                        matches = GrepProcessor._search_in_structure(
                            data, regex, max_matches, match_count
                        )
                        if matches is not None and matches != {} and matches != []:
                            filtered.append(
                                TextContent(
                                    type="text",
                                    text=json.dumps(matches, indent=2),
                                )
                            )
                            # Count matches more accurately
                            if isinstance(matches, list):
                                match_count += len(matches)
                            elif isinstance(matches, dict):
                                match_count += GrepProcessor._count_dict_matches(matches)
                            else:
                                match_count += 1
                    except json.JSONDecodeError:
                        # Not JSON, fall back to text search
                        matches = GrepProcessor._search_in_text(
                            text, regex, max_matches, match_count
                        )
                        if matches:
                            filtered.append(TextContent(type="text", text=matches))
                            match_count += len(matches.split("\n"))
                else:
                    # Plain text search
                    matches = GrepProcessor._search_in_text(
                        text, regex, max_matches, match_count
                    )
                    if matches:
                        filtered.append(TextContent(type="text", text=matches))
                        match_count += len(matches.split("\n"))

            elif isinstance(item, ImageContent):
                # Can't grep images - optionally skip or include
                # For now, we skip images when grep is applied
                pass

            if max_matches and match_count >= max_matches:
                break

        return filtered if filtered else [TextContent(type="text", text="No matches found.")]

    @staticmethod
    def _search_in_text(text: str, regex: re.Pattern, max_matches: Optional[int], current_count: int) -> str:
        """Search for pattern in text, returning matching lines."""
        lines = text.split("\n")
        matches = []
        for line in lines:
            if regex.search(line):
                matches.append(line)
                if max_matches and len(matches) + current_count >= max_matches:
                    break
        return "\n".join(matches)

    @staticmethod
    def _search_in_structure(
        data: Any, regex: re.Pattern, max_matches: Optional[int], current_count: int
    ) -> Any:
        """Recursively search for pattern in structured data."""
        if max_matches and current_count >= max_matches:
            return None

        if isinstance(data, dict):
            matches = {}
            count = current_count
            for key, value in data.items():
                if max_matches and count >= max_matches:
                    break
                # Search in key
                key_matches = regex.search(str(key))
                # Search in value (if string)
                value_matches = isinstance(value, str) and regex.search(value)
                
                if key_matches or value_matches:
                    # Include the entire key-value pair if either matches
                    matches[key] = value
                    count += 1
                elif isinstance(value, (dict, list)):
                    # Recursively search nested structures
                    nested = GrepProcessor._search_in_structure(
                        value, regex, max_matches, count
                    )
                    if nested is not None and nested != {} and nested != []:
                        matches[key] = nested
                        count += 1
            return matches if matches else None
            
        elif isinstance(data, list):
            matches = []
            count = current_count
            for item in data:
                if max_matches and count >= max_matches:
                    break
                if isinstance(item, (dict, list)):
                    nested = GrepProcessor._search_in_structure(
                        item, regex, max_matches, count
                    )
                    if nested is not None and nested != {} and nested != []:
                        matches.append(nested)
                        count += 1
                elif isinstance(item, str) and regex.search(item):
                    matches.append(item)
                    count += 1
            return matches if matches else None
        else:
            # Primitive value - check if it matches
            if regex.search(str(data)):
                return data
            return None

    @staticmethod
    def _count_dict_matches(matches: Dict[str, Any]) -> int:
        """Count the number of matches in a dictionary structure."""
        count = 0
        for value in matches.values():
            if isinstance(value, dict):
                count += GrepProcessor._count_dict_matches(value)
            elif isinstance(value, list):
                count += len(value)
            else:
                count += 1
        return count


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

    def _register_handlers(self):
        """Register MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Aggregate tools from all underlying servers."""
            import sys
            all_tools = []
            
            print(f"[DEBUG] list_tools called", file=sys.stderr)
            print(f"[DEBUG] underlying_servers keys: {list(self.underlying_servers.keys())}", file=sys.stderr)
            print(f"[DEBUG] tools_cache keys: {list(self.tools_cache.keys())}", file=sys.stderr)
            
            # First, try to use cached tools
            for server_name, cached_tools in self.tools_cache.items():
                print(f"[DEBUG] Using {len(cached_tools)} cached tools from {server_name}", file=sys.stderr)
                for tool in cached_tools:
                    prefixed_tool = Tool(
                        name=f"{server_name}_{tool.name}",
                        description=tool.description or "",
                        inputSchema=tool.inputSchema,
                    )
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
                            prefixed_tool = Tool(
                                name=f"{server_name}_{tool.name}",
                                description=tool.description or "",
                                inputSchema=tool.inputSchema,
                            )
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
            import sys
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

            # Add transformation metadata as a text content item if transformations were applied
            if transformation_meta:
                meta_text = json.dumps(
                    {"_meta": {"transformations": transformation_meta}},
                    indent=2
                )
                # Prepend metadata (clients can parse this)
                content.insert(0, TextContent(type="text", text=f"<!-- MCP Proxy Metadata\n{meta_text}\n-->"))

            return content

    async def _connect_to_server_sync(self, server_name: str, server_params: StdioServerParameters):
        """Connect to a server synchronously and keep connection alive using background task."""
        import sys
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
                import sys
                print(f"[DEBUG] Calling _connect_to_server_sync for {server_name}...", file=sys.stderr)
                await self._connect_to_server_sync(server_name, server_params)
                print(f"[DEBUG] Connection to {server_name} established", file=sys.stderr)

            except Exception as e:
                print(f"[ERROR] Failed to start connection to {server_name}: {e}")
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """Clean up connections to underlying servers."""
        import sys
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


def load_config(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """Load server configurations from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                return []
            return config.get("underlying_servers", []) or []
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using empty configuration.")
        return []
    except Exception as e:
        print(f"Error loading config: {e}. Using empty configuration.")
        return []


async def main():
    """Main entry point."""
    # Load server configurations from config file
    underlying_servers = load_config()

    proxy = MCPProxyServer(underlying_servers)
    await proxy.run()


if __name__ == "__main__":
    asyncio.run(main())

