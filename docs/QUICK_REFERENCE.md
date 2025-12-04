# Quick Reference Guide

## Logging

### Set Log Level
```bash
# Linux/macOS
export MCP_PROXY_LOG_LEVEL=DEBUG

# Windows PowerShell
$env:MCP_PROXY_LOG_LEVEL="DEBUG"

# Windows CMD
set MCP_PROXY_LOG_LEVEL=DEBUG
```

### Available Log Levels
- `DEBUG` - Detailed diagnostic information
- `INFO` - General informational messages (default)
- `WARNING` - Warning messages
- `ERROR` - Error messages
- `CRITICAL` - Critical errors

## Performance

### Parallel Tool Discovery
- Tools are automatically fetched in parallel from multiple servers
- Provides 2-3x speedup compared to sequential fetching
- No configuration needed - works automatically

### Tool Caching
- Tool definitions are cached after first discovery
- Cached tools are returned instantly (<1ms)
- Cache persists for server lifetime

## Configuration

### Environment Variables
- `MCP_PROXY_LOG_LEVEL` - Control logging verbosity (default: `INFO`)

### Config File
- Location: `config.yaml` (in current directory or project root)
- Format: YAML
- See `config.yaml.example` for template

## Common Operations

### Run Server
```bash
uv run -m mcp_proxy
```

### Run with Debug Logging
```bash
export MCP_PROXY_LOG_LEVEL=DEBUG
uv run -m mcp_proxy
```

### Run Tests
```bash
uv run pytest
```

### Run Specific Test
```bash
uv run pytest tests/test_processors.py -v
```

## Tool Naming

- Format: `{server_name}_{tool_name}`
- Example: `filesystem_read_file`
- Prevents naming conflicts between servers

## Projection

### Include Mode
```json
{
  "_meta": {
    "projection": {
      "mode": "include",
      "fields": ["name", "email"]
    }
  }
}
```

### Exclude Mode
```json
{
  "_meta": {
    "projection": {
      "mode": "exclude",
      "fields": ["password", "ssn"]
    }
  }
}
```

## Grep

### Basic Grep
```json
{
  "_meta": {
    "grep": {
      "pattern": "ERROR",
      "caseInsensitive": true
    }
  }
}
```

### Grep with Context
```json
{
  "_meta": {
    "grep": {
      "pattern": "ERROR",
      "contextLines": {
        "both": 3
      }
    }
  }
}
```

### Grep Multiline
```json
{
  "_meta": {
    "grep": {
      "pattern": "def.*\\n.*return",
      "multiline": true
    }
  }
}
```

## Combined Transformations

```json
{
  "_meta": {
    "projection": {
      "mode": "include",
      "fields": ["name", "email"]
    },
    "grep": {
      "pattern": "gmail",
      "caseInsensitive": true
    }
  }
}
```

## Troubleshooting

### No Logs Appearing
- Check `MCP_PROXY_LOG_LEVEL` is set
- Verify logs aren't filtered
- Check logging module import

### Slow Tool Discovery
- Check underlying server performance
- Verify network connectivity
- Ensure parallel execution is working

### High Memory Usage
- Check for memory leaks
- Verify tool cache size
- Monitor connection count

## File Locations

- Source code: `src/mcp_proxy/`
- Tests: `tests/`
- Documentation: `docs/`
- Config example: `config.yaml.example`
- Main entry: `src/mcp_proxy/__main__.py`

## Key Modules

- `server.py` - Main server implementation
- `processors.py` - Projection and grep processors
- `config.py` - Configuration loading
- `logging_config.py` - Logging infrastructure

## Performance Tips

1. Use `INFO` or `WARNING` log level in production
2. Keep servers running to maintain tool cache
3. Use parallel server configuration (automatic)
4. Monitor transformation performance for large responses

## Links

- [Full Documentation](README.md)
- [Architecture](ARCHITECTURE.md)
- [Logging Guide](LOGGING.md)
- [Performance Guide](PERFORMANCE.md)

