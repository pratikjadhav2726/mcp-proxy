"""
Test the everything server directly to see if it works.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_everything_direct():
    """Test everything server directly."""
    print("Testing everything server directly...")
    
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything", "stdio"]
    )
    
    try:
        print("Connecting...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Initializing...")
                init_result = await asyncio.wait_for(session.initialize(), timeout=30.0)
                print(f"Connected! Server: {init_result.serverInfo.name if init_result.serverInfo else 'Unknown'}")
                
                print("Listing tools...")
                tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                print(f"Found {len(tools_result.tools)} tools")
                
                if tools_result.tools:
                    print("First 5 tools:")
                    for tool in tools_result.tools[:5]:
                        print(f"  - {tool.name}")
                
                return True
    except asyncio.TimeoutError as e:
        print(f"Timeout: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_everything_direct())
    exit(0 if success else 1)

