"""
Test the proxy server with the everything server configured.
"""

import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_with_everything():
    """Test proxy server with everything server."""
    print("Testing MCP Proxy Server with Everything Server")
    print("=" * 60)
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize session
                init_result = await session.initialize()
                print(f"[OK] Connected to proxy server")
                print(f"    Server: {init_result.serverInfo.name if init_result.serverInfo else 'Unknown'}")
                print(f"    Version: {init_result.serverInfo.version if init_result.serverInfo else 'Unknown'}")
                
                # Wait a bit for underlying servers to connect
                await asyncio.sleep(2)
                
                # List tools
                print("\n[INFO] Listing available tools...")
                tools_result = await session.list_tools()
                print(f"[OK] Found {len(tools_result.tools)} tools")
                
                if tools_result.tools:
                    print("\nAvailable tools:")
                    for i, tool in enumerate(tools_result.tools, 1):
                        print(f"  {i}. {tool.name}")
                        if tool.description:
                            desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
                            print(f"     {desc}")
                    
                    # Test calling a tool with projection
                    if tools_result.tools:
                        test_tool = tools_result.tools[0]
                        print(f"\n[INFO] Testing tool call with projection: {test_tool.name}")
                        print("       (This will fail if tool doesn't accept empty args)")
                        
                        # Try to call with projection meta
                        try:
                            # Get tool schema to see what args it needs
                            if hasattr(test_tool, 'inputSchema') and test_tool.inputSchema:
                                print(f"       Tool schema: {test_tool.inputSchema}")
                        except Exception as e:
                            print(f"       Could not inspect tool schema: {e}")
                else:
                    print("\n[WARNING] No tools available!")
                    print("          This could mean:")
                    print("          1. The everything server didn't connect")
                    print("          2. The everything server has no tools")
                    print("          3. There's a connection error (check proxy logs)")
                
                print("\n" + "=" * 60)
                print("[OK] Test completed")
                
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_with_everything())
    sys.exit(0 if success else 1)

