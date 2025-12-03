"""
Simple test script to verify the proxy server works correctly.
"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_proxy_connection():
    """Test basic connection to proxy server."""
    print("Testing MCP Proxy Server Connection...")
    print("=" * 50)
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize session
                init_result = await session.initialize()
                print("[OK] Successfully connected to proxy server")
                print(f"  Server: {init_result.serverInfo.name if init_result.serverInfo else 'Unknown'}")
                print(f"  Version: {init_result.serverInfo.version if init_result.serverInfo else 'Unknown'}")
                
                # List tools
                tools_result = await session.list_tools()
                print(f"\n[OK] Successfully listed tools")
                print(f"  Available tools: {len(tools_result.tools)}")
                
                if tools_result.tools:
                    print("  Tool names:")
                    for tool in tools_result.tools:
                        print(f"    - {tool.name}")
                else:
                    print("  (No tools available - configure underlying servers in config.yaml)")
                
                # Check capabilities
                print(f"\n[OK] Server capabilities:")
                if init_result.capabilities and hasattr(init_result.capabilities, 'experimental'):
                    exp = init_result.capabilities.experimental
                    if exp:
                        print("  Experimental features:")
                        if isinstance(exp, dict):
                            for key, value in exp.items():
                                print(f"    - {key}: {value}")
                        else:
                            print(f"    - {exp}")
                
                print("\n" + "=" * 50)
                print("[OK] All tests passed! Proxy server is working correctly.")
                print("\nTo test with actual tools:")
                print("1. Configure underlying servers in config.yaml")
                print("2. Use the proxy with _meta for projections and grep")
                
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_proxy_connection())
    exit(0 if success else 1)

