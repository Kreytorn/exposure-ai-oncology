"""Orchestration plane: the LLM tool-calling agent and its typed tools.

The agent never touches voxels and never invents a measurement - it calls the typed
tools in tools.py, reads their JSON, and narrates, keeping a trace that links every
number to the tool call that produced it.
"""
