"""
Quick test to verify the proxy server connects and lists tools.
"""

import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def quick_test():
    """Quick test of proxy server."""
    print("Testing MCP Proxy Server...")
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Wait longer for underlying servers to connect and load tools
                print("[INFO] Waiting for underlying servers to connect...")
                await asyncio.sleep(5)
                
                # Initialize
                await session.initialize()
                print("[OK] Connected to proxy")
                
                # List tools multiple times to see if they appear
                for attempt in range(3):
                    tools_result = await session.list_tools()
                    print(f"[INFO] Attempt {attempt + 1}: Found {len(tools_result.tools)} tools")
                    if tools_result.tools:
                        break
                    if attempt < 2:
                        await asyncio.sleep(2)
                
                if tools_result.tools:
                    print(f"\n[OK] Successfully found {len(tools_result.tools)} tools:")
                    for tool in tools_result.tools[:10]:  # Show first 10
                        print(f"  - {tool.name}")
                    if len(tools_result.tools) > 10:
                        print(f"  ... and {len(tools_result.tools) - 10} more")
                else:
                    print("\n[WARNING] No tools found after multiple attempts")
                    print("          This suggests the underlying server connection")
                    print("          may not be established properly.")
                
                return True
                
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)

