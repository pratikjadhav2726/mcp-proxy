"""
Example usage of the MCP Proxy Server

This demonstrates how to use the proxy server with field projection and grep search.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def example_with_projection():
    """Example: Using field projection to get only specific fields."""
    print("=== Example: Field Projection ===\n")
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Wait for underlying servers to connect
            await asyncio.sleep(3)
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools[:10]]}")
            if len(tools.tools) > 10:
                print(f"... and {len(tools.tools) - 10} more\n")
            else:
                print()
            
            # Try to find a tool that might return structured data
            # Look for tools from the "everything" server
            everything_tools = [t for t in tools.tools if t.name.startswith("everything_")]
            
            if everything_tools:
                # Try using a tool that returns structured data
                tool_name = everything_tools[0].name
                print(f"Trying tool: {tool_name}")
                print("Note: This example demonstrates projection syntax.\n")
                print("Example projection request:")
                print(f"  Tool: {tool_name}")
                print("  Arguments with projection:")
                print("  {")
                print('    "argument": "value",')
                print('    "_meta": {')
                print('      "projection": {')
                print('        "mode": "include",')
                print('        "fields": ["name", "email", "status"]')
                print("      }")
                print("    }")
                print("  }")
                print("\nThis would return only the specified fields from the response.\n")
            else:
                print("No tools available. Configure underlying servers in config.yaml.\n")


async def example_with_grep():
    """Example: Using grep to filter tool outputs."""
    print("=== Example: Grep Search ===\n")
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Wait for underlying servers to connect
            await asyncio.sleep(3)
            
            # List available tools
            tools = await session.list_tools()
            
            # Look for file reading tools
            file_tools = [t for t in tools.tools if "read" in t.name.lower() or "file" in t.name.lower()]
            
            if file_tools:
                tool_name = file_tools[0].name
                print(f"Example tool: {tool_name}")
                print("Note: This example demonstrates grep syntax.\n")
                print("Example grep request:")
                print(f"  Tool: {tool_name}")
                print("  Arguments with grep:")
                print("  {")
                print('    "path": "/path/to/logfile.log",')
                print('    "_meta": {')
                print('      "grep": {')
                print('        "pattern": "ERROR|WARN",')
                print('        "caseInsensitive": true,')
                print('        "maxMatches": 20,')
                print('        "target": "content"')
                print("      }")
                print("    }")
                print("  }")
                print("\nThis would return only lines matching the pattern.\n")
            else:
                print("Example grep syntax:")
                print("  {")
                print('    "_meta": {')
                print('      "grep": {')
                print('        "pattern": "ERROR",')
                print('        "caseInsensitive": true,')
                print('        "maxMatches": 10')
                print("      }")
                print("    }")
                print("  }\n")


async def example_combined():
    """Example: Using both projection and grep together."""
    print("=== Example: Combined Transformations ===\n")
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Wait for underlying servers to connect
            await asyncio.sleep(3)
            
            print("Example: Combining projection and grep")
            print("This allows you to:")
            print("  1. Filter fields using projection")
            print("  2. Search within the filtered results using grep\n")
            print("Example combined request:")
            print("  {")
            print('    "filter": "active",')
            print('    "_meta": {')
            print('      "projection": {')
            print('        "mode": "include",')
            print('        "fields": ["name", "email", "status"]')
            print("      },")
            print('      "grep": {')
            print('        "pattern": "gmail\\.com",')
            print('        "caseInsensitive": true,')
            print('        "target": "structuredContent"')
            print("      }")
            print("    }")
            print("  }")
            print("\nThis would:")
            print("  - First project to only include name, email, status fields")
            print("  - Then grep for entries containing 'gmail.com'\n")


async def main():
    """Run all examples."""
    print("MCP Proxy Server Usage Examples\n")
    print("=" * 50 + "\n")
    
    # Note: These examples are commented out because they require
    # actual underlying MCP servers to be configured.
    # Uncomment and configure your servers in config.yaml to test.
    
    await example_with_projection()
    await example_with_grep()
    await example_combined()
    
    print("\n" + "=" * 50)
    print("Note: Configure underlying servers in config.yaml to test these examples.")


if __name__ == "__main__":
    asyncio.run(main())

