"""Local protocol simulation only: late frontend connect after timeout.

Does not access Google, open browser, or run notebook/training.
"""
import asyncio
import contextlib
import json
import websockets
import colab_bridge_launcher
from colab_mcp import session as session_module
from colab_mcp.websocket_server import COLAB


async def fake_frontend(wss, peers):
    async with websockets.connect(
        f'ws://127.0.0.1:{wss.port}/?access_token={wss.token}', origin=COLAB,
        subprotocols=['mcp'],
    ) as ws:
        peers.append(ws)
        async for raw in ws:
            message = json.loads(raw)
            if 'id' not in message:
                continue
            if message.get('method') == 'initialize':
                result = {'protocolVersion':message['params']['protocolVersion'],
                          'capabilities':{'tools':{}},'serverInfo':{'name':'LOCAL-TEST-ONLY','version':'1'}}
            elif message.get('method') == 'tools/list':
                result = {'tools':[{'name':'simulation_only','description':'Local test; not a real Colab tool',
                                    'inputSchema':{'type':'object','properties':{}}}]}
            elif message.get('method') == 'tools/call':
                result = {'content':[{'type':'text','text':'LOCAL_SIMULATION_ONLY'}], 'isError':False}
            else:
                result = {}
            await ws.send(json.dumps({'jsonrpc':'2.0','id':message['id'],'result':result}))


async def check():
    proxy = session_module.ColabSessionProxy()
    frontend = None
    peers = []
    try:
        await proxy.start_proxy_server()
        client = proxy.middleware[0].proxy_client
        session_module.UI_CONNECTION_TIMEOUT = .03
        await client.await_proxy_connection()
        assert not client._start_task.cancelled()
        assert not client._start_task.done()
        print('PASS: first timeout leaves initialization task alive.')
        frontend = asyncio.create_task(fake_frontend(proxy.wss, peers))
        session_module.UI_CONNECTION_TIMEOUT = 5.
        await client.await_proxy_connection()
        assert client.is_connected()
        tools = await client.proxy_mcp_client.list_tools()
        assert any(t.name == 'simulation_only' for t in tools)
        print('PASS: late authenticated frontend completes MCP initialization and tools/list.')
        inventory = await colab_bridge_launcher._list_colab_tools()
        assert inventory['connected'] and inventory['tools'][0]['name'] == 'simulation_only'
        called = await colab_bridge_launcher._call_colab_tool('simulation_only', {})
        assert called['content'][0]['text'] == 'LOCAL_SIMULATION_ONLY'
        try:
            await colab_bridge_launcher._call_colab_tool('not_exposed', {})
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown tool must be rejected')
        print('PASS: stable delegates list/call paired tools and reject unknown tool names.')
    finally:
        if frontend is not None:
            if peers:
                await peers[0].close(code=1000)
            else:
                frontend.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await frontend
        await proxy.cleanup()


if __name__ == '__main__':
    asyncio.run(asyncio.wait_for(check(),timeout=12))
