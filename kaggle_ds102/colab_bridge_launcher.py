"""Launch official Colab MCP with one IPv4 loopback listener on macOS.

Binding localhost:0 can choose different ephemeral ports for IPv4 and IPv6.
Keep the upstream token/origin checks intact; bind only 127.0.0.1.
"""
import asyncio
import contextlib
from typing import Any
from colab_mcp.websocket_server import ColabWebSocketServer
from colab_mcp import session as session_module
from colab_mcp import main, mcp

_original_init = ColabWebSocketServer.__init__


def _loopback_init(self, host='127.0.0.1'):
    _original_init(self, host=host)


ColabWebSocketServer.__init__ = _loopback_init


async def _await_without_canceling_proxy(self):
    # A user may accept the browser prompt after the initial tool call times out.
    # Never cancel the persistent MCP initialization task on that timeout.
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(
            asyncio.gather(
                self.wss.connection_live.wait(), asyncio.shield(self._start_task)
            ),
            timeout=session_module.UI_CONNECTION_TIMEOUT,
        )


session_module.ColabProxyClient.await_proxy_connection = _await_without_canceling_proxy

# Some clients retain the initial MCP tool catalog despite list_changed.
# Expose stable delegates from startup; preserve the upstream paired transport.
_active_proxy = None
_original_start = session_module.ColabSessionProxy.start_proxy_server


async def _start_and_track(self):
    global _active_proxy
    await _original_start(self)
    _active_proxy = self


session_module.ColabSessionProxy.start_proxy_server = _start_and_track


def _paired_client():
    if _active_proxy is None:
        raise RuntimeError('Colab proxy has not started.')
    client = _active_proxy.middleware[0].proxy_client
    if not client.is_connected():
        raise RuntimeError('Colab is not paired. Use open_colab_browser_connection first.')
    return client.proxy_mcp_client


async def _list_colab_tools() -> dict[str, Any]:
    """List actual tools and input schemas from the paired Colab notebook.

    Use this after pairing, even if notebook tools are absent from the host
    catalog. This does not execute notebook code or start training.
    """
    tools = await _paired_client().list_tools()
    return {'connected': True, 'tools': [t.model_dump(mode='json') for t in tools]}


async def _call_colab_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool discovered with list_colab_tools on the paired Colab.

    Follow its returned input schema exactly. Code execution runs through the
    paired Colab notebook, not via a local Python interpreter.
    """
    client = _paired_client()
    allowed = {tool.name for tool in await client.list_tools()}
    if tool_name not in allowed:
        raise ValueError('Tool is not exposed by the paired Colab notebook.')
    result = await client.call_tool(tool_name, arguments)
    return {'content': [block.model_dump(mode='json') for block in result.content],
            'structuredContent': result.structured_content, 'isError': result.is_error}


mcp.tool(name='list_colab_tools')(_list_colab_tools)
mcp.tool(name='call_colab_tool')(_call_colab_tool)

if __name__ == '__main__':
    main()
