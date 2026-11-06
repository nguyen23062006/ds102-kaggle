"""Kiểm tra MCP handshake và danh sách tool; không chạy notebook/huấn luyện."""
import asyncio
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check():
    root = Path(__file__).resolve().parents[1]
    parameters = StdioServerParameters(
        command=str(root / '.venv-colab-mcp/bin/uvx'),
        args=['--python', '3.13', '--from',
              'git+https://github.com/googlecolab/colab-mcp@b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1',
              'python', str(root / 'kaggle_ds102/colab_bridge_launcher.py')],
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            print('MCP handshake OK; tools:', names)
            assert 'open_colab_browser_connection' in names
            assert 'list_colab_tools' in names
            assert 'call_colab_tool' in names


if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(check(), timeout=40))
