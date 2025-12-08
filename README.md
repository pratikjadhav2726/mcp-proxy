# MCP Proxy Server

An MCP (Model Context Protocol) proxy server that acts as an intermediary between MCP clients and underlying MCP tool servers. This proxy implements **field projection** and **grep search** capabilities to optimize token usage, enhance privacy, and improve performance.

## Overview

This proxy server builds on the ideas discussed in [GitHub Discussion #1709]([https://github.com/modelcontextprotocol/servers/issues/1709](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1709)) of the `modelcontextprotocol` repository. It enables:

- **Field Projection**: Request only specific fields from tool responses, reducing token usage by 85-95% in many cases
- **Grep Search**: Filter tool outputs using regex patterns, extracting only relevant information
- **Backwards Compatibility**: Works with existing MCP servers without modification
- **Privacy Enhancement**: Ensures only requested data is exposed to clients

## Features

### 1. Field Projection

Request only specific fields from tool responses:

```json
{
  "method": "tools/call",
  "params": {
    "name": "server::get_user_profile",
    "arguments": {
      "userId": "user123",
      "_meta": {
        "projection": {
          "mode": "include",
          "fields": ["name", "email"]
        }
      }
    }
  }
}
```

**Supported Modes:**
- `include`: Only return specified fields
- `exclude`: Return all fields except specified ones
- `view`: Use named preset views (future enhancement)

### 2. Grep Search

Filter tool outputs using regex patterns:

**Basic Grep:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "server::read_file",
    "arguments": {
      "path": "/logs/app.log",
      "_meta": {
        "grep": {
          "pattern": "ERROR",
          "caseInsensitive": true,
          "maxMatches": 10,
          "target": "content"
        }
      }
    }
  }
}
```

**Grep with Context Lines:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "server::read_file",
    "arguments": {
      "path": "/logs/app.log",
      "_meta": {
        "grep": {
          "pattern": "ERROR",
          "contextLines": {
            "both": 3
          }
        }
      }
    }
  }
}
```

**Multiline Pattern Matching:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "server::read_file",
    "arguments": {
      "path": "/code/script.py",
      "_meta": {
        "grep": {
          "pattern": "def.*\\n.*return",
          "multiline": true
        }
      }
    }
  }
}
```

**Grep Options:**
- `pattern`: Regex pattern to search for
- `caseInsensitive`: Case-insensitive matching (default: false)
- `multiline`: Enable multiline pattern matching - allows patterns to span multiple lines (default: false)
- `maxMatches`: Maximum number of matches to return
- `contextLines`: Include context lines around matches (similar to grep -A, -B, -C)
  - `before`: Number of lines before each match (grep -B)
  - `after`: Number of lines after each match (grep -A)
  - `both`: Number of lines both before and after (grep -C, overrides before/after)
- `target`: Search in `content` (text) or `structuredContent` (JSON)

### 3. Combined Transformations

Use both projection and grep together:

```json
{
  "method": "tools/call",
  "params": {
    "name": "server::get_user_profile",
    "arguments": {
      "userId": "user123",
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
  }
}
```

## Installation

### Prerequisites

- Python 3.12 or later
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Installation from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mcp-proxy-server.git
cd mcp-proxy-server
```

2. Install dependencies using `uv`:
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### Installation as Package

```bash
pip install mcp-proxy-server
```

Or using uv:
```bash
uv pip install mcp-proxy-server
```

## Configuration

Configure underlying MCP servers in `config.yaml`:

```yaml
underlying_servers:
  # Example: Filesystem server
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
  
  # Example: Python-based server
  - name: my_server
    command: python
    args: ["-m", "my_mcp_server"]
  
  # Example: Node.js server
  - name: api_server
    command: node
    args: ["/path/to/server.js"]
```

**Configuration Fields:**
- `name`: Unique identifier for the server (used in tool names as `name_tool_name`)
  - Required: Yes
  - Type: String
  - Constraints: Must be unique, alphanumeric with underscores/hyphens only, 1-100 characters
- `command`: Command to run the server (e.g., `npx`, `python`, `node`)
  - Required: Yes
  - Type: String
  - Constraints: Cannot be empty
- `args`: List of arguments to pass to the command
  - Required: No (defaults to empty list)
  - Type: List of strings
- `env`: Optional environment variables dictionary
  - Required: No
  - Type: Dictionary of string key-value pairs

**Configuration Validation:**
The proxy server uses Pydantic for configuration validation, ensuring:
- All required fields are present
- Field types are correct
- Server names are unique
- Names and commands are properly formatted
- Helpful error messages for invalid configurations

If validation fails, the server will log detailed error messages and raise a `ValueError` with specific field-level error information.

## Usage

### Running the Proxy Server

Using uv:
```bash
uv run -m mcp_proxy
```

Or using the installed package:
```bash
mcp-proxy
```

Or using Python directly:
```bash
python -m mcp_proxy
```

The server will:
1. Load configuration from `config.yaml` (in the current directory or project root)
2. Connect to all configured underlying servers
3. Aggregate tools from all servers (prefixed with server name)
4. Start listening for MCP client connections via stdio

### Tool Naming Convention

Tools from underlying servers are prefixed with the server name using an underscore separator:

- Original tool: `read_file` from `filesystem` server
- Proxy tool name: `filesystem_read_file`

This prevents naming conflicts when aggregating tools from multiple servers.

### Example Client Request

```python
# Using MCP Python client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to proxy server
server_params = StdioServerParameters(
    command="uv",
    args=["run", "-m", "mcp_proxy"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Call tool with projection
        result = await session.call_tool(
            "filesystem_read_file",
            {
                "path": "/path/to/file.json",
                "_meta": {
                    "projection": {
                        "mode": "include",
                        "fields": ["name", "status"]
                    }
                }
            }
        )
        
        print(result.content)
```

## Response Metadata

The proxy adds metadata about applied transformations in the response:

```json
{
  "_meta": {
    "transformations": {
      "projection": {
        "applied": true,
        "mode": "include"
      },
      "grep": {
        "applied": true,
        "pattern": "ERROR"
      },
      "token_savings": {
        "original_size": 10000,
        "new_size": 500,
        "savings_percent": 95.0
      }
    }
  }
}
```

This metadata is included as a comment in the first content item, allowing clients to track optimization benefits.

## Architecture

```
┌─────────────┐
│ MCP Client  │
│  (e.g., LLM)│
└──────┬──────┘
       │
       │ Tool calls with _meta
       │ (projection, grep)
       ▼
┌─────────────────┐
│  MCP Proxy      │
│  Server         │
│  - Intercepts   │
│  - Transforms   │
│  - Optimizes    │
└──────┬──────────┘
       │
       │ Forwarded calls
       │ (with/without meta)
       ▼
┌─────────────┬─────────────┐
│  Server 1   │  Server 2   │  ...
│ (filesystem)│  (api)      │
└─────────────┴─────────────┘
```

## Capability Negotiation

The proxy declares its capabilities during initialization:

```json
{
  "capabilities": {
    "tools": {},
    "experimental": {
      "projection": {
        "supported": true,
        "modes": ["include", "exclude", "view"]
      },
      "grep": {
        "supported": true,
        "maxPatternLength": 1000
      }
    }
  }
}
```

## Error Handling

- **Unknown Server**: Returns error if tool name doesn't match `server::tool` format
- **Connection Failures**: Logs errors but continues with other servers
- **Tool Errors**: Propagates errors from underlying servers with context
- **Graceful Degradation**: If underlying server doesn't support projection, applies it client-side

## Token Savings

The proxy tracks and reports token savings:

- **Original Size**: Size of response from underlying server
- **New Size**: Size after transformations
- **Savings Percent**: Percentage reduction

Example: A 10KB response reduced to 500 bytes = 95% token savings.

## Privacy Benefits

- **Field-Level Control**: Only requested fields are exposed
- **Data Minimization**: Reduces risk of exposing sensitive information
- **Client-Side Filtering**: Works even if underlying servers don't support projection

## Performance Considerations

- **Latency**: Proxy adds minimal overhead (typically <10ms)
- **Parallel Tool Discovery**: Tool listing from multiple servers is performed in parallel, providing up to 3x speedup when multiple servers need tool discovery
- **Caching**: Tool definitions are cached to avoid repeated queries
- **Async Processing**: All operations are asynchronous for optimal performance

## Development

### Project Structure

```
mcp-proxy-server/
├── src/
│   └── mcp_proxy/          # Main package
│       ├── __init__.py      # Package initialization
│       ├── __main__.py      # Entry point
│       ├── server.py        # Main server implementation
│       ├── processors.py    # Projection and grep processors
│       ├── config.py        # Configuration loading
│       └── logging_config.py # Logging configuration
├── tests/                   # Test files
├── examples/                # Example usage scripts
├── docs/                    # Additional documentation
├── config.yaml.example      # Example configuration
├── pyproject.toml           # Project configuration and dependencies
├── README.md                # This file
├── CONTRIBUTING.md          # Contribution guidelines
├── CHANGELOG.md             # Version history
└── LICENSE                  # MIT License
```

### Key Components

- **`MCPProxyServer`**: Main server class that manages connections and tool calls (in `server.py`)
- **`ProjectionProcessor`**: Handles field projection operations (in `processors.py`)
- **`GrepProcessor`**: Handles grep search operations (in `processors.py`)
- **`logging_config`**: Provides structured logging infrastructure (in `logging_config.py`)

### Development Setup

1. Clone the repository and install in development mode:
```bash
git clone https://github.com/yourusername/mcp-proxy-server.git
cd mcp-proxy-server
uv sync --group dev
```

2. Run tests:
```bash
uv run pytest
```

3. Run the server in development:
```bash
uv run -m mcp_proxy
```

### Logging Configuration

The proxy server uses structured logging. Control log verbosity using the `MCP_PROXY_LOG_LEVEL` environment variable:

```bash
# Set log level to DEBUG for detailed output
export MCP_PROXY_LOG_LEVEL=DEBUG
uv run -m mcp_proxy

# Set log level to WARNING to reduce output
export MCP_PROXY_LOG_LEVEL=WARNING
uv run -m mcp_proxy
```

**Available log levels:**
- `DEBUG`: Detailed diagnostic information (includes all operations)
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical errors that may cause the server to stop

**Log Format:**
```
[LEVEL] module_name: message
```

Example output:
```
[INFO] mcp_proxy.server: Connected to underlying server: filesystem
[INFO] mcp_proxy.server: Loaded 5 tools from filesystem
[DEBUG] mcp_proxy.server: list_tools called
[WARNING] mcp_proxy.config: Config file config.yaml not found. Using empty configuration.
[ERROR] mcp_proxy.server: Error listing tools from server1: Connection timeout
```

### Testing

Test with MCP Inspector:

```bash
# Terminal 1: Run proxy server
uv run -m mcp_proxy

# Terminal 2: Run inspector
npx -y @modelcontextprotocol/inspector
```

Connect inspector to the proxy server to test tool calls with projections and grep.

## Future Enhancements

- [ ] Response caching for repeated queries
- [ ] Named view presets for common projections
- [ ] Full JSONPath support for nested field selection
- [ ] Deduplication of responses
- [ ] Summarization of large outputs
- [ ] Metrics and monitoring
- [ ] HTTP/WebSocket transport support

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

This project aligns with the MCP protocol specification and the ideas discussed in [issue #1709](https://github.com/modelcontextprotocol/servers/issues/1709).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

- [Architecture Documentation](docs/ARCHITECTURE.md) - System architecture and design
- [Logging Documentation](docs/LOGGING.md) - Logging configuration and usage
- [Performance Documentation](docs/PERFORMANCE.md) - Performance characteristics and optimizations

## References

- [MCP Protocol Specification](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [GitHub Issue #1709 - Field Projection Discussion](https://github.com/modelcontextprotocol/servers/issues/1709)

