# Quick Start Guide

Get the MCP Proxy Server up and running in 5 minutes.

## Step 1: Install Dependencies

```bash
uv sync
```

## Step 2: Configure Underlying Servers

Copy the example config and edit it:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` to add your MCP servers. For example:

```yaml
underlying_servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
```

## Step 3: Run the Proxy Server

```bash
uv run proxy_server.py
```

The server will:
- Load configuration from `config.yaml`
- Connect to all configured servers
- Start listening for MCP client connections via stdio

## Step 4: Test with MCP Inspector

In another terminal:

```bash
npx -y @modelcontextprotocol/inspector
```

Connect to your proxy server and try calling tools with `_meta` for projections and grep.

## Example Tool Call

When calling tools through the proxy, use the format: `server_name::tool_name`

**With Projection:**
```json
{
  "name": "filesystem::read_file",
  "arguments": {
    "path": "/path/to/file.json",
    "_meta": {
      "projection": {
        "mode": "include",
        "fields": ["name", "status"]
      }
    }
  }
}
```

**With Grep:**
```json
{
  "name": "filesystem::read_file",
  "arguments": {
    "path": "/path/to/logfile.log",
    "_meta": {
      "grep": {
        "pattern": "ERROR",
        "caseInsensitive": true,
        "maxMatches": 10
      }
    }
  }
}
```

## Troubleshooting

**Problem**: "Config file not found"
- **Solution**: Create `config.yaml` from `config.yaml.example`

**Problem**: "Failed to connect to server"
- **Solution**: Check that the server command and args are correct in `config.yaml`

**Problem**: "Tool name must be in format 'server::tool'"
- **Solution**: Use the format `server_name::tool_name` when calling tools

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/](examples/) for usage examples
- Review the architecture and capabilities in the main README

