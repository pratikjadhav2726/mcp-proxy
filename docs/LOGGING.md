# Logging Documentation

## Overview

The MCP Proxy Server uses Python's standard `logging` module to provide structured, configurable logging throughout the application. This replaces the previous `print()` statements with a proper logging infrastructure.

## Features

- **Structured Logging**: Consistent log format across all modules
- **Configurable Log Levels**: Control verbosity via environment variable
- **Module-Specific Loggers**: Each module has its own logger for better traceability
- **Error Stack Traces**: Automatic exception logging with full stack traces
- **Standard Output**: Logs to stderr by default (following MCP stdio conventions)

## Configuration

### Environment Variable

Set the `MCP_PROXY_LOG_LEVEL` environment variable to control logging verbosity:

```bash
# Linux/macOS
export MCP_PROXY_LOG_LEVEL=DEBUG
uv run -m mcp_proxy

# Windows (PowerShell)
$env:MCP_PROXY_LOG_LEVEL="DEBUG"
uv run -m mcp_proxy

# Windows (CMD)
set MCP_PROXY_LOG_LEVEL=DEBUG
uv run -m mcp_proxy
```

### Log Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `DEBUG` | Detailed diagnostic information | Development, troubleshooting |
| `INFO` | General informational messages | Normal operation (default) |
| `WARNING` | Warning messages for potential issues | Production monitoring |
| `ERROR` | Error messages for failures | Error tracking |
| `CRITICAL` | Critical errors that may stop the server | Critical failure alerts |

### Default Behavior

If `MCP_PROXY_LOG_LEVEL` is not set, the default log level is `INFO`.

## Log Format

All log messages follow this format:

```
[LEVEL] logger_name: message
```

**Examples:**
```
[INFO] mcp_proxy.server: Connected to underlying server: filesystem
[DEBUG] mcp_proxy.server: list_tools called
[WARNING] mcp_proxy.config: Config file config.yaml not found
[ERROR] mcp_proxy.server: Error listing tools from server1: Connection timeout
```

## Usage in Code

### Getting a Logger

```python
from mcp_proxy.logging_config import get_logger

logger = get_logger(__name__)
```

### Logging Messages

```python
# Debug messages (only shown at DEBUG level)
logger.debug("Detailed diagnostic information")

# Info messages (shown at INFO level and above)
logger.info("Server started successfully")

# Warning messages (shown at WARNING level and above)
logger.warning("Configuration file not found, using defaults")

# Error messages (shown at ERROR level and above)
logger.error("Failed to connect to server", exc_info=True)
```

### Exception Logging

Always use `exc_info=True` when logging exceptions to include stack traces:

```python
try:
    result = await session.list_tools()
except Exception as e:
    logger.error(f"Error listing tools: {e}", exc_info=True)
```

## Module-Specific Loggers

Each module has its own logger, making it easy to filter logs:

- `mcp_proxy.server`: Server operations, connections, tool calls
- `mcp_proxy.config`: Configuration loading
- `mcp_proxy.processors`: Transformation operations (projection, grep)
- `mcp_proxy.logging_config`: Logging system initialization

## Log Output

By default, logs are written to `stderr`, which is the standard for MCP servers using stdio transport. This ensures logs don't interfere with the MCP protocol communication on stdout.

## Best Practices

1. **Use Appropriate Log Levels**
   - `DEBUG`: Detailed diagnostic info (development only)
   - `INFO`: Normal operational messages
   - `WARNING`: Potential issues that don't stop operation
   - `ERROR`: Errors that prevent normal operation

2. **Include Context**
   - Always include relevant context in log messages
   - Use f-strings for formatted messages: `logger.info(f"Connected to {server_name}")`

3. **Log Exceptions Properly**
   - Use `exc_info=True` when logging exceptions
   - Include error context: `logger.error(f"Failed to connect to {server_name}: {e}", exc_info=True)`

4. **Avoid Sensitive Data**
   - Never log passwords, API keys, or other sensitive information
   - Be careful with user data in logs

5. **Performance Considerations**
   - Debug logging can be verbose; use it judiciously
   - Consider log volume in production environments

## Examples

### Development Mode

```bash
# Enable debug logging for development
export MCP_PROXY_LOG_LEVEL=DEBUG
uv run -m mcp_proxy
```

**Output:**
```
[DEBUG] mcp_proxy.server: list_tools called
[DEBUG] mcp_proxy.server: underlying_servers keys: ['filesystem', 'git']
[DEBUG] mcp_proxy.server: Fetching tools from 2 server(s) in parallel
[DEBUG] mcp_proxy.server: Fetching tools from filesystem session
[DEBUG] mcp_proxy.server: Got 5 tools from filesystem session
[INFO] mcp_proxy.server: Loaded 5 tools from filesystem
```

### Production Mode

```bash
# Use INFO level for production
export MCP_PROXY_LOG_LEVEL=INFO
uv run -m mcp_proxy
```

**Output:**
```
[INFO] mcp_proxy.server: Connected to underlying server: filesystem
[INFO] mcp_proxy.server: Loaded 5 tools from filesystem
[INFO] mcp_proxy.server: Connected to underlying server: git
[INFO] mcp_proxy.server: Loaded 3 tools from git
```

### Error Monitoring

```bash
# Use WARNING level to focus on issues
export MCP_PROXY_LOG_LEVEL=WARNING
uv run -m mcp_proxy
```

**Output:**
```
[WARNING] mcp_proxy.config: Config file config.yaml not found. Using empty configuration.
[ERROR] mcp_proxy.server: Failed to start connection to server1: Connection timeout
```

## Integration with Monitoring

The structured logging format makes it easy to integrate with log aggregation tools:

- **JSON Format**: Can be configured for structured log parsing
- **Log Aggregation**: Works with tools like ELK, Splunk, Datadog
- **Alerting**: ERROR and CRITICAL levels can trigger alerts

## Troubleshooting

### No Logs Appearing

1. Check that `MCP_PROXY_LOG_LEVEL` is set correctly
2. Verify logs aren't being filtered by your environment
3. Check that the logging module is properly imported

### Too Many Logs

1. Increase log level: `export MCP_PROXY_LOG_LEVEL=WARNING`
2. Filter by module if your log viewer supports it
3. Use log rotation for production deployments

### Missing Stack Traces

Ensure `exc_info=True` is used when logging exceptions:
```python
logger.error("Error message", exc_info=True)  # ✓ Correct
logger.error("Error message")  # ✗ Missing stack trace
```

