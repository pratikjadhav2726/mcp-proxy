# Architecture

## Overview

The MCP Proxy Server acts as an intermediary between MCP clients and underlying MCP tool servers. It provides value-added features like field projection and grep search while maintaining full compatibility with the MCP protocol.

## Components

### Core Modules

#### `mcp_proxy.server`
Contains the main `MCPProxyServer` class that:
- Manages connections to underlying MCP servers
- Aggregates tools from multiple servers
- Intercepts tool calls and applies transformations
- Handles server lifecycle and cleanup

#### `mcp_proxy.processors`
Contains transformation processors:
- **ProjectionProcessor**: Applies field projection (include/exclude/view modes)
- **GrepProcessor**: Applies regex-based filtering to tool outputs

#### `mcp_proxy.config`
Handles configuration loading from YAML files.

#### `mcp_proxy.logging_config`
Provides structured logging infrastructure with configurable log levels.

## Data Flow

```
MCP Client
    │
    │ Tool call with _meta
    ▼
MCP Proxy Server
    │
    │ 1. Parse tool name (server_tool)
    │ 2. Extract _meta from arguments
    │ 3. Forward call to underlying server
    ▼
Underlying MCP Server
    │
    │ Response
    ▼
MCP Proxy Server
    │
    │ 1. Apply projection (if specified)
    │ 2. Apply grep (if specified)
    │ 3. Calculate token savings
    ▼
MCP Client
```

## Connection Management

The proxy maintains persistent connections to underlying servers using:
- Background async tasks to keep connections alive
- Context managers for proper resource cleanup
- Connection pooling and caching

## Tool Aggregation

Tools from multiple servers are aggregated with naming convention:
- Format: `{server_name}_{tool_name}`
- Prevents naming conflicts
- Maintains server identity

**Parallel Discovery**: Tool listing from multiple servers is performed in parallel using `asyncio.gather()`, providing significant performance improvements (2-3x speedup) when multiple servers need tool discovery.

**Caching**: Tool definitions are cached after first discovery to avoid repeated queries and provide instant tool listing for cached tools.

## Transformation Pipeline

1. **Schema Enhancement**: Add `_meta` parameter to all tool schemas
2. **Request Processing**: Extract `_meta` from arguments before forwarding
3. **Response Processing**: Apply transformations in order:
   - Projection (if specified)
   - Grep (if specified)
4. **Metadata Addition**: Add transformation metadata to response

## Error Handling

- Graceful degradation if underlying server fails
- Validation of transformation parameters
- Clear error messages with context
- Timeout handling for long-running operations

## Performance Considerations

- **Async/await**: All operations are asynchronous for non-blocking I/O
- **Parallel Tool Discovery**: Tools are fetched from multiple servers concurrently using `asyncio.gather()`
- **Connection Pooling**: Persistent connections reduce connection overhead
- **Tool Caching**: Tool definitions are cached to avoid repeated queries
- **Minimal Transformation Overhead**: Projection and grep operations typically add <10ms latency
- **Structured Logging**: Configurable log levels allow performance tuning (DEBUG can impact performance)

See [PERFORMANCE.md](PERFORMANCE.md) for detailed performance documentation.

