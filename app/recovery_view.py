"""Shared recovery controls for experiment launcher pages."""

from __future__ import annotations

from typing import Any

import streamlit as st


def add_recovery_flags(command: list[str], *, enabled: bool, backup_dir: str = "") -> list[str]:
    result = list(command)
    if not enabled:
        return [*result, "--no-recovery"]
    return [*result, "--backup-dir", backup_dir] if backup_dir.strip() else result


def render_recovery_settings(defaults: Any = None, *, key: str) -> tuple[bool, str]:
    source = defaults if isinstance(defaults, dict) else getattr(defaults, "__dict__", {})
    with st.expander("Kayıt ve Kurtarma", expanded=False):
        enabled = st.toggle(
            "Epoch sonunda kurtarma kaydı",
            value=bool(source.get("enabled", True)),
            key=f"{key}_recovery_enabled",
        )
        backup_dir = st.text_input(
            "Kalıcı yedek klasörü (isteğe bağlı)",
            value=str(source.get("backup_dir") or ""),
            disabled=not enabled,
            key=f"{key}_recovery_backup",
        )
        st.caption("Kesilen eğitimlere Job Monitor'daki doğrulanmış son epoch kaydından devam edilir.")
        if not enabled:
            st.warning("Kurtarma kapalıysa kesilen eğitime devam edilemez.")
    return enabled, backup_dir.strip()
