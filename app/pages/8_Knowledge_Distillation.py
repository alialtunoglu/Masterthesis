"""Compatibility route for the shared KD launcher."""
from kd_launcher import render_kd_launcher

render_kd_launcher(multi=False, unified=globals().get("UNIFIED_LAUNCHER", False))
