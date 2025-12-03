"""
Test projection and grep functionality with actual tool calls.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_projection():
    """Test field projection with a tool call."""
    print("=== Testing Field Projection ===\n")
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await asyncio.sleep(3)  # Wait for underlying servers
            
            # Test with structuredContent tool which returns JSON
            print("Calling everything_structuredContent with projection...")
            result = await session.call_tool(
                "everything_structuredContent",
                {
                    "_meta": {
                        "projection": {
                            "mode": "include",
                            "fields": ["name", "status"]
                        }
                    }
                }
            )
            
            print(f"\nResult ({len(result.content)} content items):")
            for i, item in enumerate(result.content):
                if hasattr(item, 'text'):
                    text = item.text
                    # Check if it's metadata
                    if "MCP Proxy Metadata" in text:
                        print(f"\n  [{i}] Metadata:")
                        print(text[:200] + "..." if len(text) > 200 else text)
                    else:
                        print(f"\n  [{i}] Content:")
                        try:
                            data = json.loads(text)
                            print(json.dumps(data, indent=2))
                        except:
                            print(text[:500] + "..." if len(text) > 500 else text)
            
            print("\n[OK] Projection test completed!\n")


async def test_grep():
    """Test grep functionality with a tool call."""
    print("=== Testing Grep Search ===\n")
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await asyncio.sleep(3)  # Wait for underlying servers
            
            # Test with echo tool which returns text
            print("Calling everything_echo with grep...")
            result = await session.call_tool(
                "everything_echo",
                {
                    "message": "This is a test message with ERROR and WARN keywords",
                    "_meta": {
                        "grep": {
                            "pattern": "ERROR|WARN",
                            "caseInsensitive": True
                        }
                    }
                }
            )
            
            print(f"\nResult ({len(result.content)} content items):")
            for i, item in enumerate(result.content):
                if hasattr(item, 'text'):
                    text = item.text
                    if "MCP Proxy Metadata" in text:
                        print(f"\n  [{i}] Metadata:")
                        print(text[:200] + "..." if len(text) > 200 else text)
                    else:
                        print(f"\n  [{i}] Content:")
                        print(text)
            
            print("\n[OK] Grep test completed!\n")


async def test_combined():
    """Test combined projection and grep."""
    print("=== Testing Combined Projection + Grep ===\n")
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "proxy_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await asyncio.sleep(3)  # Wait for underlying servers
            
            print("Calling everything_structuredContent with both projection and grep...")
            result = await session.call_tool(
                "everything_structuredContent",
                {
                    "_meta": {
                        "projection": {
                            "mode": "include",
                            "fields": ["name", "status"]
                        },
                        "grep": {
                            "pattern": "active|pending",
                            "target": "structuredContent",
                            "caseInsensitive": True
                        }
                    }
                }
            )
            
            print(f"\nResult ({len(result.content)} content items):")
            for i, item in enumerate(result.content):
                if hasattr(item, 'text'):
                    text = item.text
                    if "MCP Proxy Metadata" in text:
                        print(f"\n  [{i}] Metadata:")
                        print(text[:300] + "..." if len(text) > 300 else text)
                    else:
                        print(f"\n  [{i}] Content:")
                        try:
                            data = json.loads(text)
                            print(json.dumps(data, indent=2))
                        except:
                            print(text[:500] + "..." if len(text) > 500 else text)
            
            print("\n[OK] Combined test completed!\n")


async def main():
    """Run all tests."""
    print("MCP Proxy Server - Projection and Grep Tests")
    print("=" * 60 + "\n")
    
    try:
        await test_projection()
        await test_grep()
        await test_combined()
        
        print("=" * 60)
        print("[OK] All tests completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

