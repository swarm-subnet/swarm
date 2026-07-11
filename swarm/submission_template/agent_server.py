"""Compatibility import for the image-owned model-graph RPC implementation."""

from swarm.model_graph.server import ObservationDecoder, make_agent_server, serve

__all__ = ["ObservationDecoder", "make_agent_server", "serve"]
