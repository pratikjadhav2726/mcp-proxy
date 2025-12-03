# MCP Proxy Server Examples

This directory contains example code demonstrating how to use the MCP Proxy Server.

## Example Files

- `example_usage.py`: Demonstrates basic usage with projection and grep

## Running Examples

1. **Configure your servers**: Update `config.yaml` with your underlying MCP servers

2. **Run the proxy server** (in one terminal):
```bash
uv run proxy_server.py
```

3. **Run the example** (in another terminal):
```bash
uv run examples/example_usage.py
```

## Example Scenarios

### Scenario 1: Filtering Large API Responses

**Problem**: An API returns 100 user objects, but you only need names and emails.

**Solution**: Use field projection:
```python
result = await session.call_tool(
    "api_server::get_users",
    {
        "_meta": {
            "projection": {
                "mode": "include",
                "fields": ["name", "email"]
            }
        }
    }
)
```

**Result**: 95% token savings by returning only 2 fields per user instead of 20+.

### Scenario 2: Searching Log Files

**Problem**: A log file is 10MB, but you only need ERROR lines.

**Solution**: Use grep:
```python
result = await session.call_tool(
    "filesystem::read_file",
    {
        "path": "/var/log/app.log",
        "_meta": {
            "grep": {
                "pattern": "ERROR",
                "caseInsensitive": False,
                "maxMatches": 100
            }
        }
    }
)
```

**Result**: Only ERROR lines are returned, reducing output from 10MB to ~50KB.

### Scenario 3: Privacy-Sensitive Data

**Problem**: Need user data but want to exclude sensitive fields.

**Solution**: Use exclude projection:
```python
result = await session.call_tool(
    "database::get_user",
    {
        "userId": "123",
        "_meta": {
            "projection": {
                "mode": "exclude",
                "fields": ["password", "ssn", "creditCard"]
            }
        }
    }
)
```

**Result**: All user data except sensitive fields is returned.

## Custom Examples

Create your own examples by:

1. Setting up underlying MCP servers in `config.yaml`
2. Using the proxy server's tool naming convention: `server_name::tool_name`
3. Adding `_meta` to tool arguments for transformations

See `example_usage.py` for the basic structure.

