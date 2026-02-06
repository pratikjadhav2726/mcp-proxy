"""
Quick test to verify the proxy_get_capabilities tool.
"""
import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)

async def test_capabilities():
    """Test the proxy_get_capabilities tool."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_proxy"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools - should include proxy_get_capabilities
            print("\n=== Listing Tools ===")
            tools_result = await session.list_tools()
            print(f"\nFound {len(tools_result.tools)} tools")
            
            # Find the capabilities tool
            capabilities_tool = None
            for tool in tools_result.tools:
                if tool.name == "proxy_get_capabilities":
                    capabilities_tool = tool
                    break
            
            if capabilities_tool:
                print(f"\n✅ Found proxy_get_capabilities tool")
                print(f"Description: {capabilities_tool.description[:200]}...")
            else:
                print(f"\n❌ proxy_get_capabilities tool NOT FOUND!")
                return
            
            # Call the capabilities tool
            print("\n=== Calling proxy_get_capabilities ===")
            result = await session.call_tool(
                "proxy_get_capabilities",
                {"show_examples": True}
            )
            
            print("\n=== Capabilities Response ===")
            for content in result.content:
                if hasattr(content, 'text'):
                    print(content.text[:1000])  # First 1000 chars
                    print(f"\n... (total {len(content.text)} chars)")
            
            # Check a regular tool to see if it has proxy hints
            print("\n=== Checking Regular Tool Description ===")
            if len(tools_result.tools) > 1:
                regular_tool = tools_result.tools[1]  # Second tool (first is capabilities)
                print(f"Tool: {regular_tool.name}")
                print(f"Description:\n{regular_tool.description[:500]}...")
                
                if "_meta" in regular_tool.description:
                    print("\n✅ Regular tools include _meta hints!")
                else:
                    print("\n⚠️ Regular tools don't have _meta hints")

if __name__ == "__main__":
    asyncio.run(test_capabilities())

