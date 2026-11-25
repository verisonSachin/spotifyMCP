"""MCPServer entrypoint, lifespan management, and transport configuration."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import logging
import sys

from mcp.server import MCPServer

from spotify_mcp.client import SpotifyClient
from spotify_mcp.tools import register_tools

logger = logging.getLogger(__name__)


def create_server(
    client_factory: Callable[[], SpotifyClient] | None = None,
) -> MCPServer:
    """Create and configure the Spotify MCPServer instance."""
    factory = client_factory or SpotifyClient

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[SpotifyClient]:
        client = factory()
        try:
            yield client
        finally:
            await client.aclose()

    server = MCPServer("spotify-mcp", lifespan=lifespan)
    register_tools(server)
    return server


mcp = create_server()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spotify MCP Server (2026 Web API)")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport mode (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for streamable-http transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for streamable-http transport (default: 8000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint (`spotify-mcp`) with stderr-only logging."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    logger.info("Starting spotify-mcp server with transport=%s", args.transport)

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
