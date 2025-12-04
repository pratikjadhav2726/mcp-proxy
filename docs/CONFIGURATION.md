# Configuration Documentation

## Overview

The MCP Proxy Server uses Pydantic for robust configuration validation, ensuring that all server configurations are valid before attempting to connect to underlying servers.

## Configuration File

The configuration is stored in a YAML file (default: `config.yaml`). The server looks for this file in:
1. Current working directory
2. Project root (one level up from `src/`)

## Configuration Structure

```yaml
underlying_servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    env:  # Optional
      VAR1: value1
      VAR2: value2
```

## Server Configuration Fields

### `name` (Required)

Unique identifier for the server.

- **Type**: String
- **Constraints**:
  - Must be unique across all servers
  - Alphanumeric characters, underscores (`_`), and hyphens (`-`) only
  - 1-100 characters
  - Leading/trailing whitespace is automatically trimmed
- **Usage**: Used to prefix tool names (e.g., `filesystem_read_file`)

**Valid Examples:**
- `filesystem`
- `my_server`
- `server-1`
- `api_server_v2`

**Invalid Examples:**
- `server@name` (contains invalid character)
- `server name` (contains space)
- Empty string
- Duplicate names across servers

### `command` (Required)

Command to execute the server.

- **Type**: String
- **Constraints**:
  - Cannot be empty
  - Leading/trailing whitespace is automatically trimmed
- **Examples**: `npx`, `python`, `node`, `uv`

### `args` (Optional)

List of arguments to pass to the command.

- **Type**: List of strings
- **Default**: Empty list `[]`
- **Examples**:
  ```yaml
  args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  args: ["-m", "my_mcp_server"]
  args: []  # No arguments needed
  ```

### `env` (Optional)

Environment variables for the server process.

- **Type**: Dictionary of string key-value pairs
- **Default**: `null` (no environment variables)
- **Example**:
  ```yaml
  env:
    API_KEY: "your-api-key"
    DEBUG: "true"
    PATH: "/custom/path"
  ```

## Validation Rules

### Server Name Validation

1. **Uniqueness**: Each server must have a unique name
2. **Format**: Must match pattern `^[a-zA-Z0-9_-]+$`
3. **Length**: Between 1 and 100 characters
4. **Whitespace**: Automatically trimmed

### Command Validation

1. **Required**: Must be provided
2. **Non-empty**: Cannot be empty or whitespace-only
3. **Whitespace**: Automatically trimmed

### Global Validation

1. **No Duplicate Names**: All server names must be unique
2. **Valid YAML**: Configuration file must be valid YAML
3. **Type Safety**: All fields are type-checked

## Error Messages

When validation fails, you'll receive detailed error messages:

### Missing Required Field

```
ValueError: Invalid configuration in config.yaml:
Configuration validation errors:
  underlying_servers -> 0 -> name: Field required
```

### Duplicate Server Names

```
ValueError: Invalid configuration in config.yaml:
Configuration validation errors:
  underlying_servers: Value error, Duplicate server names found: server1. Each server must have a unique name.
```

### Invalid Name Format

```
ValueError: Invalid configuration in config.yaml:
Configuration validation errors:
  underlying_servers -> 0 -> name: String should match pattern '^[a-zA-Z0-9_-]+$'
```

### Invalid YAML

```
ValueError: YAML parsing error in config.yaml: ...
```

## Examples

### Minimal Configuration

```yaml
underlying_servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

### Multiple Servers

```yaml
underlying_servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  
  - name: git
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git", "/path/to/repo"]
  
  - name: python_server
    command: python
    args: ["-m", "my_mcp_server"]
```

### With Environment Variables

```yaml
underlying_servers:
  - name: api_server
    command: python
    args: ["-m", "api_server"]
    env:
      API_KEY: "secret-key"
      DEBUG: "false"
      LOG_LEVEL: "INFO"
```

### Empty Configuration

```yaml
underlying_servers: []
```

Or simply an empty file (returns empty list).

## Programmatic Usage

### Loading Configuration

```python
from mcp_proxy.config import load_config

# Load from default location (config.yaml)
servers = load_config()

# Load from custom location
servers = load_config("/path/to/custom_config.yaml")
```

### Using Pydantic Models Directly

```python
from mcp_proxy.config import ServerConfig, ProxyConfig

# Create a server configuration
server = ServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
)

# Create full configuration
config = ProxyConfig(
    underlying_servers=[server]
)

# Validate and convert to dict
config_dict = config.model_dump()
```

## Best Practices

1. **Use Descriptive Names**: Choose clear, descriptive server names
   - Good: `filesystem`, `git_server`, `api_backend`
   - Bad: `s1`, `server`, `test`

2. **Keep Names Simple**: Use alphanumeric characters, underscores, and hyphens only
   - Avoid special characters that might cause issues

3. **Validate Early**: Test your configuration before deploying
   ```bash
   python -c "from mcp_proxy.config import load_config; load_config('config.yaml')"
   ```

4. **Use Environment Variables**: For sensitive data like API keys, use the `env` field
   ```yaml
   env:
     API_KEY: "${API_KEY}"  # Use environment variable substitution if your YAML parser supports it
   ```

5. **Document Your Configuration**: Add comments to explain server purposes
   ```yaml
   underlying_servers:
     # Filesystem access server
     - name: filesystem
       command: npx
       args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
   ```

## Troubleshooting

### Configuration Not Found

If the config file is not found, the server will:
- Log a warning
- Return an empty list (no servers configured)
- Continue running (useful for testing)

### Validation Errors

If validation fails:
1. Check the error message for the specific field and issue
2. Verify the field name is correct (case-sensitive)
3. Check for duplicate server names
4. Ensure YAML syntax is correct

### Common Issues

**Issue**: "Field required" error
- **Solution**: Ensure all required fields (`name`, `command`) are present

**Issue**: "Duplicate server names" error
- **Solution**: Ensure each server has a unique `name`

**Issue**: "String should match pattern" error
- **Solution**: Check that server names contain only alphanumeric characters, underscores, and hyphens

**Issue**: YAML parsing errors
- **Solution**: Validate your YAML syntax using a YAML validator or linter

## Advanced Configuration

### Custom Validation

You can extend the Pydantic models to add custom validation:

```python
from mcp_proxy.config import ServerConfig
from pydantic import field_validator

class CustomServerConfig(ServerConfig):
    @field_validator("name")
    @classmethod
    def validate_name_prefix(cls, v: str) -> str:
        if not v.startswith("prod_"):
            raise ValueError("Production servers must start with 'prod_'")
        return v
```

### Configuration Schema

The Pydantic models generate JSON schemas that can be used for:
- IDE autocomplete
- Configuration file validation
- Documentation generation

Access the schema:
```python
from mcp_proxy.config import ProxyConfig

schema = ProxyConfig.model_json_schema()
print(schema)
```

## See Also

- [Architecture Documentation](ARCHITECTURE.md) - System architecture
- [Quick Reference Guide](QUICK_REFERENCE.md) - Quick lookup
- [Main README](../README.md) - Project overview

