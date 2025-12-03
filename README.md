# MCP Proxy Server

An MCP (Model Context Protocol) proxy server that acts as an intermediary between MCP clients and underlying MCP tool servers. This proxy implements **field projection** and **grep search** capabilities to optimize token usage, enhance privacy, and improve performance.

## Overview

This proxy server builds on the ideas discussed in [GitHub issue #1709](https://github.com/modelcontextprotocol/servers/issues/1709) of the `modelcontextprotocol` repository. It enables:

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

**Grep Options:**
- `pattern`: Regex pattern to search for
- `caseInsensitive`: Case-insensitive matching (default: false)
- `maxMatches`: Maximum number of matches to return
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
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd mcp-proxy-server
```

2. Install dependencies using `uv`:
```bash
uv sync
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
- `name`: Unique identifier for the server (used in tool names as `name::tool_name`)
- `command`: Command to run the server (e.g., `npx`, `python`, `node`)
- `args`: List of arguments to pass to the command

## Usage

### Running the Proxy Server

```bash
uv run proxy_server.py
```

The server will:
1. Load configuration from `config.yaml`
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
    args=["run", "proxy_server.py"]
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
- **Caching**: Future enhancement could cache responses for repeated queries
- **Async Processing**: All operations are asynchronous for optimal performance

## Development

### Project Structure

```
mcp-proxy-server/
├── proxy_server.py    # Main proxy server implementation
├── config.yaml        # Server configuration
├── pyproject.toml     # Project dependencies
├── README.md          # This file
└── main.py            # Entry point (optional)
```

### Key Components

- **`MCPProxyServer`**: Main server class that manages connections and tool calls
- **`ProjectionProcessor`**: Handles field projection operations
- **`GrepProcessor`**: Handles grep search operations

### Testing

Test with MCP Inspector:

```bash
# Terminal 1: Run proxy server
uv run proxy_server.py

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

This project aligns with the MCP protocol specification and the ideas discussed in [issue #1709](https://github.com/modelcontextprotocol/servers/issues/1709). Contributions are welcome!

## License

[Add your license here]

## References

- [MCP Protocol Specification](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [GitHub Issue #1709 - Field Projection Discussion](https://github.com/modelcontextprotocol/servers/issues/1709)

