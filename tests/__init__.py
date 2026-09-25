"""Marks this directory a regular package, which is load-bearing rather than tidy.

ultralytics ships its own top-level ``tests`` package, so with the yolo extra installed a
namespace ``tests`` here loses to it and every ``from tests.conftest import ...`` resolves
into site-packages instead.
"""
