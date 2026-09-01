"""The validator's own scripts, tests and documentation.

An explicit package rather than a namespace one: the scripts are imported by
dotted name from the CLI, and a namespace package would resolve them from a
source checkout while an installed copy had nothing to import.
"""
