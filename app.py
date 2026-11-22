"""
Footballistic — Tactical Football Match Intelligence Platform.
Architectural, ultra-minimal dark analytics workspace designed for clarity and precision.
No sidebar — unified top bar, bespoke telemetry components, and instant match switching.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.heatmap import SpatialAnalytics
from src.analysis.events import MatchEventEngine
from src.analysis.tactical_physical import (
    PhysicalPerformanceEngine,
    TacticalDynamicsEngine,
    SPEED_ZONES,
    render_player_speed_chart,
    render_tactical_shape_chart,
    render_pressing_chart,
)
from src.homography.pitch_view import PitchView2D


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Footballistic — Tactical Intelligence",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# DESIGN SYSTEM — OBSIDIAN PITCH STUDIO (Minimalist Editorial Dark UI)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    :root {
        --bg-canvas: #090B0F;
        --bg-surface: #0F1218;
        --bg-elevated: #151922;
        --bg-hover: #1B202C;
        --border-subtle: rgba(255, 255, 255, 0.068);
        --border-strong: rgba(255, 255, 255, 0.13);
        --text-primary: #F3F5F8;
        --text-secondary: #8A93A5;
        --text-muted: #5A6273;
        --accent: #10B981;
        --accent-soft: rgba(16, 185, 129, 0.12);
        --home: #E5484D;
        --away: #3B82F6;
    }

    html, body, [class*="css"], .stApp, button, input, select, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background:
            radial-gradient(960px 360px at 50% -120px, rgba(16, 185, 129, 0.055), transparent 70%),
            var(--bg-canvas);
        color: var(--text-primary);
    }

    /* Hide default Streamlit chrome & sidebar */
    header[data-testid="stHeader"] { background: transparent; height: 0; }
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"], section[data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1380px !important;
    }

    /* ---------------- BRAND TOP BAR ---------------- */
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 12px;
        height: 40px;
    }
    .brand-mark {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        background: linear-gradient(145deg, #141A24, #0C1016);
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .brand-name {
        font-size: 1.08rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text-primary);
        line-height: 1;
    }
    .brand-divider {
        width: 1px;
        height: 16px;
        background: rgba(255, 255, 255, 0.12);
    }
    .brand-tag {
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .top-hairline {
        height: 1px;
        background: var(--border-subtle);
        margin: 12px 0 6px 0;
    }

    /* ---------------- EDITORIAL UNDERLINE TABS ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        background: transparent;
        border-bottom: 1px solid var(--border-subtle);
        padding: 0;
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 2px;
        background: transparent !important;
        border: none !important;
        transition: color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        letter-spacing: -0.01em;
        margin: 0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover p {
        color: var(--text-primary) !important;
    }
    .stTabs [aria-selected="true"] p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
        height: 2px !important;
        border-radius: 2px;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent !important;
    }

    /* ---------------- FORM CONTROLS ---------------- */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px !important;
    }

    div[data-baseweb="select"] > div {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        min-height: 40px !important;
        height: 40px !important;
        font-size: 0.84rem !important;
        color: var(--text-primary) !important;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: var(--border-strong) !important;
        background: var(--bg-elevated) !important;
    }
    div[data-baseweb="select"] svg {
        color: var(--text-muted) !important;
    }

    /* Buttons & Popover Trigger (pixel-matched 40px height) */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stPopover"] > button {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        min-height: 40px !important;
        height: 40px !important;
        padding: 0 12px !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em;
        transition: all 0.15s ease;
        box-shadow: none !important;
    }
    .stButton > button p,
    .stDownloadButton > button p,
    div[data-testid="stPopover"] > button p {
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        line-height: 1 !important;
        margin: 0 !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stPopover"] > button:hover {
        background: var(--bg-elevated) !important;
        border-color: var(--border-strong) !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #04130D !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"] p {
        color: #04130D !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #34D399 !important;
        border-color: #34D399 !important;
    }

    /* Minimal Segmented Radio Control */
    div[role="radiogroup"] {
        gap: 4px !important;
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        padding: 3px;
        border-radius: 9px;
        display: inline-flex !important;
        width: fit-content !important;
    }
    div[role="radiogroup"] > label {
        background: transparent;
        border: 1px solid transparent;
        padding: 5px 14px !important;
        min-height: 30px;
        display: inline-flex !important;
        align-items: center !important;
        border-radius: 6px;
        margin: 0 !important;
        cursor: pointer;
        user-select: none;
        transition: all 0.15s ease;
    }
    .meta-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 7px 12px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        float: right;
    }
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[role="radiogroup"] > label p {
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        margin: 0 !important;
    }
    div[role="radiogroup"] > label:hover p {
        color: var(--text-primary) !important;
    }
    div[role="radiogroup"] > label:has(input:checked) {
        background: var(--bg-elevated);
        border-color: var(--border-strong);
    }
    div[role="radiogroup"] > label:has(input:checked) p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* ---------------- ARCHITECTURAL COMPONENTS ---------------- */
    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
        margin: 14px 0 20px 0;
    }
    .kpi-cell {
        padding: 16px 20px;
        border-right: 1px solid var(--border-subtle);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .kpi-cell:last-child { border-right: none; }
    .kpi-label {
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .kpi-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.48rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        font-variant-numeric: tabular-nums;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.74rem;
        color: var(--text-secondary);
        font-weight: 400;
    }

    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .dot-home { background: var(--home); }
    .dot-away { background: var(--away); }
    .dot-accent { background: var(--accent); }

    /* Section Header */
    .sec-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 16px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .sec-head-tight { margin-top: 6px; }
    .sec-left {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .sec-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        color: var(--accent);
        letter-spacing: 0.05em;
    }
    .sec-title {
        font-size: 0.90rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }
    .sec-sub {
        font-size: 0.76rem;
        color: var(--text-muted);
    }

    /* Visual Panel Frame */
    .panel-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin: 4px 0 8px 2px;
    }
    .panel-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: none;
        letter-spacing: 0;
    }

    /* Possession Bar */
    .poss-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .poss-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 10px;
    }
    .poss-team {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .poss-name {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-secondary);
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }
    .poss-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }
    .poss-detail {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
    }
    .poss-track {
        display: flex;
        width: 100%;
        height: 6px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.05);
        gap: 3px;
    }
    .poss-fill-home {
        background: var(--home);
        border-radius: 999px 0 0 999px;
    }
    .poss-fill-away {
        background: var(--away);
        border-radius: 0 999px 999px 0;
    }

    /* Single-Pass Telemetry Inspector Card */
    .tele-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 12px;
    }
    .tele-head {
        padding: 14px 18px;
        border-bottom: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }
    .tele-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.04em;
    }
    .status-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 999px;
        text-transform: uppercase;
    }
    .pill-ok {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.30);
    }
    .pill-miss {
        background: rgba(229, 72, 77, 0.14);
        color: #F87171;
        border: 1px solid rgba(229, 72, 77, 0.30);
    }
    .tele-route {
        padding: 14px 18px;
        border-bottom: 1px solid var(--border-subtle);
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .tele-players {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tele-arrow { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.90rem; }
    .tele-meta {
        font-size: 0.75rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tele-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
    }
    .tele-stat {
        padding: 13px 18px;
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
    }
    .tele-stat:nth-child(2n) { border-right: none; }
    .tele-stat:nth-last-child(-n+2) { border-bottom: none; }
    .tele-stat-lbl {
        font-size: 0.64rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .tele-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }
    .tele-stat-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 2px;
    }

    /* Spatial Distribution Bars */
    .dist-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 14px;
    }
    .dist-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 13px 16px;
    }
    .dist-head {
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 9px;
    }
    .dist-bar {
        display: flex;
        height: 5px;
        border-radius: 999px;
        overflow: hidden;
        gap: 2px;
        margin-bottom: 9px;
        background: rgba(255, 255, 255, 0.04);
    }
    .dist-legend {
        display: flex;
        justify-content: space-between;
        font-size: 0.74rem;
        color: var(--text-secondary);
    }
    .dist-legend strong {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Bespoke Obsidian Data Tables */
    .obs-table-wrap {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
        margin: 8px 0 12px 0;
    }
    .obs-table-scroll {
        max-height: 410px;
        overflow-y: auto;
        overflow-x: auto;
    }
    .obs-table-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
    .obs-table-scroll::-webkit-scrollbar-track { background: #0C0F15; }
    .obs-table-scroll::-webkit-scrollbar-thumb { background: #222938; border-radius: 999px; }
    .obs-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.80rem;
    }
    .obs-table thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #131720;
        color: #6B7280;
        font-size: 0.64rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.085em;
        padding: 11px 14px;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        white-space: nowrap;
    }
    .obs-table thead th.num-h { text-align: right; }
    .obs-table tbody td {
        padding: 10px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.045);
        color: #CBD5E1;
        white-space: nowrap;
        vertical-align: middle;
    }
    .obs-table tbody tr:last-child td { border-bottom: none; }
    .obs-table tbody tr:hover td { background: rgba(255, 255, 255, 0.025); }
    .obs-table tbody tr.row-active td { background: rgba(16, 185, 129, 0.085); }
    .obs-num {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
        text-align: right;
        color: #F3F5F8;
        font-weight: 500;
    }
    .obs-muted { color: #6B7280; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; }
    .obs-rank {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748B;
    }
    .obs-rank-top { color: #34D399; }
    .obs-player {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #F8FAFC;
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }
    .obs-team {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
        color: #94A3B8;
    }
    .obs-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 3px 8px;
        border-radius: 999px;
        text-transform: uppercase;
    }
    .obs-pill-ok { background: rgba(16, 185, 129, 0.14); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.28); }
    .obs-pill-miss { background: rgba(229, 72, 77, 0.14); color: #F87171; border: 1px solid rgba(229, 72, 77, 0.28); }
    .obs-pill-tag { background: rgba(255, 255, 255, 0.05); color: #CBD5E1; border: 1px solid rgba(255, 255, 255, 0.08); text-transform: none; letter-spacing: 0; font-family: 'Inter', sans-serif; font-size: 0.72rem; font-weight: 500; }
    .obs-bar-cell {
        display: inline-flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 3px;
        min-width: 64px;
    }
    .obs-bar-track {
        width: 64px;
        height: 3.5px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        overflow: hidden;
    }
    .obs-bar-fill { height: 100%; border-radius: 999px; }

    /* Compact Vertical Speed Zone + Sprint Panel */
    .sz-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 15px 18px;
        height: 100%;
    }
    .sz-row {
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-bottom: 10px;
    }
    .sz-row-top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-size: 0.74rem;
    }
    .sz-lbl { color: #94A3B8; display: inline-flex; align-items: center; gap: 6px; font-weight: 500; }
    .sz-val { font-family: 'JetBrains Mono', monospace; color: #F3F5F8; font-weight: 600; font-size: 0.76rem; }
    .sz-pct { color: #64748B; font-weight: 500; font-size: 0.70rem; }
    .sz-track { height: 5px; width: 100%; border-radius: 999px; background: rgba(255, 255, 255, 0.05); overflow: hidden; }
    .sz-fill { height: 100%; border-radius: 999px; }
    .sz-sprints {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid var(--border-subtle);
        font-size: 0.72rem;
        color: #8A93A5;
    }
    .sz-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #F8FAFC;
        background: rgba(229, 72, 77, 0.14);
        border: 1px solid rgba(229, 72, 77, 0.30);
        border-radius: 6px;
        padding: 2px 7px;
        margin: 4px 5px 0 0;
    }

    /* Media & Dataframes */
    [data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
    }
    [data-testid="stImageCaption"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.71rem !important;
        color: var(--text-muted) !important;
        text-align: left !important;
        margin-top: 6px !important;
    }
    video {
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# UI COMPONENT RENDERERS
# =============================================================================
def render_kpi_strip(items: List[Tuple[str, str, str, Optional[str]]]):
    """Renders a unified 4-column architectural KPI strip with hairline dividers."""
    cells = []
    for label, value, sub, dot_cls in items:
        dot_html = f'<span class="dot {dot_cls}"></span>' if dot_cls else ""
        cells.append(
            f'<div class="kpi-cell">'
            f'<div class="kpi-label">{dot_html}{label}</div>'
            f'<div class="kpi-val">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="kpi-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


def render_section_header(num: str, title: str, subtitle: str = "", tight: bool = False):
    cls = "sec-head sec-head-tight" if tight else "sec-head"
    sub_html = f'<span class="sec-sub">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="sec-left"><span class="sec-num">{num}</span><span class="sec-title">{title}</span></div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_panel_label(title: str, meta: str = ""):
    meta_html = f'<span class="panel-meta">{meta}</span>' if meta else ""
    st.markdown(f'<div class="panel-label"><span>{title}</span>{meta_html}</div>', unsafe_allow_html=True)


def render_distribution_strip(blocks: List[Tuple[str, List[Tuple[str, float, str]]]]):
    """Renders 1 or 2 sleek spatial distribution bar cards side-by-side."""
    cols = f"repeat({len(blocks)}, minmax(0, 1fr))"
    cards = []
    for title, segs in blocks:
        bars = "".join(
            f'<div style="width:{max(1.0, val)}%; background:{col}; border-radius:2px;"></div>'
            for _, val, col in segs
        )
        legends = "".join(
            f'<span>{lbl} <strong>{val:.0f}%</strong></span>'
            for lbl, val, _ in segs
        )
        cards.append(
            f'<div class="dist-card">'
            f'<div class="dist-head">{title}</div>'
            f'<div class="dist-bar">{bars}</div>'
            f'<div class="dist-legend">{legends}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="dist-row" style="grid-template-columns:{cols};">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_speed_zone_panel(profile: Dict):
    """Renders a sleek vertical FIFA 5-zone speed panel + detected sprint bursts."""
    rows_html = []
    for name, lo, hi, col in SPEED_ZONES:
        dist = profile["zone_distance_m"].get(name, 0.0)
        pct = profile["zone_pct"].get(name, 0.0)
        rng = f"≥{int(lo)}" if hi >= 90 else f"{int(lo)}–{int(hi)}"
        rows_html.append(
            f'<div class="sz-row">'
            f'<div class="sz-row-top">'
            f'<span class="sz-lbl"><span style="width:7px;height:7px;border-radius:50%;background:{col};display:inline-block;"></span>{name} <span class="sz-pct">({rng} km/h)</span></span>'
            f'<span class="sz-val">{dist:.1f} m <span class="sz-pct">· {pct:.0f}%</span></span>'
            f'</div>'
            f'<div class="sz-track"><div class="sz-fill" style="width:{min(100.0, pct)}%;background:{col};"></div></div>'
            f'</div>'
        )

    sprints = profile.get("sprint_events", [])
    if sprints:
        chips = "".join(
            f'<span class="sz-chip">{s0:.1f}–{s1:.1f}s · {pk:.1f} km/h</span>'
            for s0, s1, pk in sprints
        )
        sp_html = f'<div class="sz-sprints"><div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748B;font-weight:600;margin-bottom:2px;">Detected Sprint Bursts ({len(sprints)})</div>{chips}</div>'
    else:
        sp_html = '<div class="sz-sprints"><span style="color:#5A6273;">No sustained sprint bursts (≥ 21 km/h for ≥ 0.4s) in this clip.</span></div>'

    st.markdown(
        f'<div class="sz-card">'
        f'<div class="dist-head" style="margin-bottom:12px;">FIFA 5-Zone Intensity Breakdown</div>'
        f'{"".join(rows_html)}'
        f'{sp_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_passes_table_html(all_passes: List[Dict], active_pass_num: Optional[int] = None):
    """Renders the Complete Pass Telemetry Log as a bespoke Obsidian HTML table."""
    max_len = max((p["length_m"] for p in all_passes), default=50.0) or 50.0
    body = []
    for p in all_passes:
        is_ok = bool(p["is_completed"])
        dot_c = "dot-home" if p["team_id"] == 0 else "dot-away"
        t_lbl = "Home" if p["team_id"] == 0 else "Away"
        pill = '<span class="obs-pill obs-pill-ok">Completed</span>' if is_ok else '<span class="obs-pill obs-pill-miss">Intercepted</span>'
        to_txt = f"#{p['target_id']}" if is_ok else f"#{p['target_id']} <span class='obs-muted'>(Opp)</span>"
        bar_w = min(100.0, (p["length_m"] / max_len) * 100.0)
        bar_c = "#10B981" if is_ok else "#E5484D"
        tr_cls = ' class="row-active"' if p["pass_number"] == active_pass_num else ""
        body.append(
            f'<tr{tr_cls}>'
            f'<td class="obs-rank">#{p["pass_number"]:02d}</td>'
            f'<td class="obs-muted">{p["time_sec"]:.1f}s – {p["end_time_sec"]:.1f}s</td>'
            f'<td><span class="obs-team"><span class="dot {dot_c}"></span>T{p["team_id"]} {t_lbl}</span></td>'
            f'<td class="obs-player">#{p["player_id"]}</td>'
            f'<td>{to_txt}</td>'
            f'<td>{pill}</td>'
            f'<td><span class="obs-pill obs-pill-tag">{p["subtype"]}</span></td>'
            f'<td class="obs-num"><div class="obs-bar-cell"><span>{p["length_m"]:.1f} m</span><div class="obs-bar-track"><div class="obs-bar-fill" style="width:{bar_w:.0f}%;background:{bar_c};"></div></div></div></td>'
            f'<td class="obs-num">{p["avg_speed_kmh"]:.1f}</td>'
            f'<td class="obs-num" style="color:#94A3B8;">{p["peak_speed_kmh"]:.1f}</td>'
            f'<td class="obs-num" style="color:#94A3B8;">{p["duration_sec"]:.2f}s</td>'
            f'<td style="color:#94A3B8;font-size:0.75rem;">{p["zone"]}</td>'
            f'</tr>'
        )
    st.markdown(
        '<div class="obs-table-wrap"><div class="obs-table-scroll">'
        '<table class="obs-table"><thead><tr>'
        '<th>Pass</th><th>Window</th><th>Team</th><th>From</th><th>To</th>'
        '<th>Outcome</th><th>Type</th><th class="num-h">Length</th>'
        '<th class="num-h">Avg km/h</th><th class="num-h">Peak km/h</th>'
        '<th class="num-h">Time</th><th>Target Zone</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>',
        unsafe_allow_html=True,
    )


def render_physical_leaderboard_html(rows: List[Dict], active_pid: Optional[int] = None):
    """Renders the Match Physical Leaderboard as a sleek Obsidian HTML table with micro-bars."""
    max_dist = max((r["Distance (m)"] for r in rows), default=80.0) or 80.0
    max_top = max((r["Top Speed (km/h)"] for r in rows), default=30.0) or 30.0
    max_hsr = max((r["HSR >18 (m)"] for r in rows), default=25.0) or 25.0

    body = []
    for idx, r in enumerate(rows, start=1):
        is_home = "Team 0" in r["Team"]
        dot_c = "dot-home" if is_home else "dot-away"
        t_short = "T0 Home" if is_home else "T1 Away"
        t_bar = "#E5484D" if is_home else "#3B82F6"
        pid_num = int(r["Player"].split()[0].replace("#", ""))
        tr_cls = ' class="row-active"' if pid_num == active_pid else ""
        rk_cls = "obs-rank obs-rank-top" if idx <= 3 else "obs-rank"

        d_pct = min(100.0, (r["Distance (m)"] / max_dist) * 100.0)
        v_pct = min(100.0, (r["Top Speed (km/h)"] / max_top) * 100.0)
        h_pct = min(100.0, (r["HSR >18 (m)"] / max_hsr) * 100.0) if max_hsr > 0 else 0.0
        sp_cnt = int(r["Sprints"])
        sp_badge = (
            f'<span class="obs-pill obs-pill-ok">{sp_cnt}</span>'
            if sp_cnt > 0 else '<span class="obs-muted">0</span>'
        )
        body.append(
            f'<tr{tr_cls}>'
            f'<td class="{rk_cls}">{idx:02d}</td>'
            f'<td><span class="obs-player"><span class="dot {dot_c}"></span>{r["Player"]}</span></td>'
            f'<td><span class="obs-team">{t_short}</span></td>'
            f'<td class="obs-num"><div class="obs-bar-cell"><span>{r["Distance (m)"]:.1f} m</span><div class="obs-bar-track"><div class="obs-bar-fill" style="width:{d_pct:.0f}%;background:{t_bar};"></div></div></div></td>'
            f'<td class="obs-num">{r["Avg Speed (km/h)"]:.1f}</td>'
            f'<td class="obs-num"><div class="obs-bar-cell"><span style="color:#F8FAFC;font-weight:600;">{r["Top Speed (km/h)"]:.1f}</span><div class="obs-bar-track"><div class="obs-bar-fill" style="width:{v_pct:.0f}%;background:#10B981;"></div></div></div></td>'
            f'<td class="obs-num"><div class="obs-bar-cell"><span>{r["HSR >18 (m)"]:.1f} m</span><div class="obs-bar-track"><div class="obs-bar-fill" style="width:{h_pct:.0f}%;background:#F59E0B;"></div></div></div></td>'
            f'<td class="obs-num" style="color:#94A3B8;">{r["Sprint >21 (m)"]:.1f} m</td>'
            f'<td class="obs-num">{sp_badge}</td>'
            f'<td class="obs-num" style="color:#94A3B8;">{r["Peak Accel (m/s²)"]:.2f}</td>'
            f'</tr>'
        )
    st.markdown(
        '<div class="obs-table-wrap"><div class="obs-table-scroll">'
        '<table class="obs-table"><thead><tr>'
        '<th>#</th><th>Player</th><th>Team</th><th class="num-h">Distance</th>'
        '<th class="num-h">Avg km/h</th><th class="num-h">Top km/h</th>'
        '<th class="num-h">HSR &gt;18</th><th class="num-h">Sprint &gt;21</th>'
        '<th class="num-h">Sprints</th><th class="num-h">Peak m/s²</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>',
        unsafe_allow_html=True,
    )


def render_spatial_summary_html(rows: List[Dict]):
    """Renders the Spatial Player Dataset in Tab 4 as a sleek Obsidian HTML table."""
    max_d = max((r["Est. Distance (m)"] for r in rows), default=80.0) or 80.0
    body = []
    for r in rows:
        is_home = r["Team"] == "Team 0"
        dot_c = "dot-home" if is_home else "dot-away"
        t_bar = "#E5484D" if is_home else "#3B82F6"
        d_pct = min(100.0, (r["Est. Distance (m)"] / max_d) * 100.0)
        pres = float(r["Pitch Presence (%)"])
        body.append(
            f'<tr>'
            f'<td><span class="obs-player"><span class="dot {dot_c}"></span>{r["Player ID"]}</span></td>'
            f'<td><span class="obs-team">{"Home" if is_home else "Away"}</span></td>'
            f'<td class="obs-num"><div class="obs-bar-cell"><span>{pres:.0f}%</span><div class="obs-bar-track"><div class="obs-bar-fill" style="width:{min(100.0, pres):.0f}%;background:#64748B;"></div></div></div></td>'
            f'<td class="obs-num"><div class="obs-bar-cell"><span>{r["Est. Distance (m)"]:.1f} m</span><div class="obs-bar-track"><div class="obs-bar-fill" style="width:{d_pct:.0f}%;background:{t_bar};"></div></div></div></td>'
            f'<td class="obs-num">{r["Avg Speed (km/h)"]:.1f}</td>'
            f'<td class="obs-num" style="color:#34D399;">{r.get("Top Speed (km/h)", 0.0):.1f}</td>'
            f'<td style="color:#94A3B8;font-size:0.74rem;">{r["Tactical Zone"]}</td>'
            f'</tr>'
        )
    st.markdown(
        '<div class="obs-table-wrap"><div class="obs-table-scroll" style="max-height:360px;">'
        '<table class="obs-table"><thead><tr>'
        '<th>Player</th><th>Team</th><th class="num-h">Presence</th>'
        '<th class="num-h">Distance</th><th class="num-h">Avg km/h</th>'
        '<th class="num-h">Top km/h</th><th>Tactical Role</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>',
        unsafe_allow_html=True,
    )


def render_possession_bar(poss: Dict):
    p0 = poss["team0_possession_pct"]
    p1 = poss["team1_possession_pct"]
    st.markdown(f"""
    <div class="poss-card">
        <div class="poss-top">
            <div class="poss-team">
                <span class="poss-name"><span class="dot dot-home"></span>Team 0 · Home</span>
                <span class="poss-pct">{p0:.1f}%</span>
                <span class="poss-detail">{poss['team0_possession_sec']}s · {poss['team0_passes']}/{poss['team0_passes_attempted']} passes</span>
            </div>
            <div class="poss-team">
                <span class="poss-detail">{poss['team1_passes']}/{poss['team1_passes_attempted']} passes · {poss['team1_possession_sec']}s</span>
                <span class="poss-pct">{p1:.1f}%</span>
                <span class="poss-name">Team 1 · Away<span class="dot dot-away"></span></span>
            </div>
        </div>
        <div class="poss-track">
            <div class="poss-fill-home" style="width:{p0}%;"></div>
            <div class="poss-fill-away" style="width:{p1}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pass_telemetry_card(p: Dict):
    is_ok = bool(p["is_completed"])
    pill_cls = "pill-ok" if is_ok else "pill-miss"
    pill_txt = "COMPLETED" if is_ok else "INTERCEPTED"
    t_dot = "dot-home" if p["team_id"] == 0 else "dot-away"
    t_name = "Team 0 (Home)" if p["team_id"] == 0 else "Team 1 (Away)"
    arrow = "→" if is_ok else "✕"
    target_lbl = f"Player #{p['target_id']}" if is_ok else f"Intercepted by #{p['target_id']}"

    st.markdown(f"""
    <div class="tele-card">
        <div class="tele-head">
            <span class="tele-id">PASS #{p['pass_number']:02d} &nbsp;·&nbsp; {p['time_sec']:.1f}s – {p['end_time_sec']:.1f}s</span>
            <span class="status-pill {pill_cls}">{pill_txt}</span>
        </div>
        <div class="tele-route">
            <div class="tele-players">
                <span class="dot {t_dot}"></span>
                <span>Player #{p['player_id']}</span>
                <span class="tele-arrow">{arrow}</span>
                <span>{target_lbl}</span>
            </div>
            <div class="tele-meta">
                <span>{t_name}</span>
                <span>·</span>
                <span>{p['subtype']}</span>
                <span>·</span>
                <span>{p['start_zone']} to {p['zone']}</span>
            </div>
        </div>
        <div class="tele-grid">
            <div class="tele-stat">
                <div class="tele-stat-lbl">Pass Length</div>
                <div class="tele-stat-val">{p['length_m']:.1f} m</div>
                <div class="tele-stat-sub">Arc {p['arc_length_m']:.1f} m</div>
            </div>
            <div class="tele-stat">
                <div class="tele-stat-lbl">Avg Ball Speed</div>
                <div class="tele-stat-val">{p['avg_speed_kmh']:.0f} km/h</div>
                <div class="tele-stat-sub">Peak {p['peak_speed_kmh']:.0f} km/h ({p['avg_speed_ms']:.1f} m/s)</div>
            </div>
            <div class="tele-stat">
                <div class="tele-stat-lbl">Flight Time</div>
                <div class="tele-stat-val">{p['duration_sec']:.2f} s</div>
                <div class="tele-stat-sub">Frames {p['frame']}–{p['end_frame']}</div>
            </div>
            <div class="tele-stat">
                <div class="tele-stat-lbl">Forward Gain</div>
                <div class="tele-stat-val">{p['forward_gain_m']:+.1f} m</div>
                <div class="tele-stat-sub">Attacking progression</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE & INSTANT PRE-COMPUTED LOAD
# =============================================================================
def load_match_data(match_key: str):
    if match_key == "sample2":
        spatial_json = "output_sample2/spatial_data.json"
        video_file = "output_sample2/analyzed_match.mp4"
    else:
        spatial_json = "output/spatial_data.json"
        video_file = "output/analyzed_match.mp4"

    if os.path.exists(spatial_json):
        try:
            sa = SpatialAnalytics.load_from_json(spatial_json)
            sa.enforce_team_squad_limits(max_players_per_team=11)
            pv = PitchView2D(pitch_width=1050, pitch_height=680)
            return {
                "spatial_analytics": sa,
                "pitch_view": pv,
                "output_video_path": video_file if os.path.exists(video_file) else None,
                "total_players": len(sa.player_tracks),
                "video_info": {
                    "width": sa.width,
                    "height": sa.height,
                    "total_frames": sa.total_processed_frames,
                },
            }
        except Exception:
            pass
    return None


def init_session():
    if "current_match" not in st.session_state:
        st.session_state["current_match"] = "sample1"
    if "results" not in st.session_state or st.session_state["results"] is None:
        st.session_state["results"] = load_match_data(st.session_state["current_match"])


init_session()


# =============================================================================
# UNIFIED SINGLE-ROW TOP NAVIGATION BAR
# =============================================================================
bar_brand, bar_match, bar_model, bar_upload = st.columns(
    [3.3, 2.2, 1.4, 1.3], vertical_alignment="center"
)

with bar_brand:
    st.markdown("""
    <div class="brand-lockup">
        <div class="brand-mark">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="4" width="20" height="16" rx="2"/>
                <line x1="12" y1="4" x2="12" y2="20"/>
                <circle cx="12" cy="12" r="3.2"/>
            </svg>
        </div>
        <span class="brand-name">Footballistic</span>
        <span class="brand-divider"></span>
        <span class="brand-tag">Match Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

with bar_match:
    match_labels = {
        "sample1": "Match 01 — Broadcast (sample.mp4)",
        "sample2": "Match 02 — Wide Tactical (sample2.mp4)",
    }
    match_keys = list(match_labels.keys())
    cur_match = st.session_state.get("current_match", "sample1")
    selected_match_idx = match_keys.index(cur_match) if cur_match in match_keys else 0

    chosen_match = st.selectbox(
        "Match Footage",
        options=match_keys,
        index=selected_match_idx,
        format_func=lambda k: match_labels[k],
        label_visibility="collapsed",
    )
    if chosen_match != st.session_state.get("current_match"):
        st.session_state["current_match"] = chosen_match
        st.session_state["results"] = load_match_data(chosen_match)
        st.session_state.pop("pass_sel_num", None)
        st.rerun()

with bar_model:
    model_choice = st.selectbox(
        "Detection Model",
        ["YOLO11m", "YOLOv8m"],
        index=0,
        help="Detection backbone used when running full video analysis",
        label_visibility="collapsed",
    )
    selected_model_path = "weights/yolo11m.pt" if "11" in model_choice else "weights/yolov8m.pt"

with bar_upload:
    with st.popover("Upload / Run", use_container_width=True):
        st.markdown("**Run Video Analysis**")
        st.caption("Upload custom match footage or re-run the full CV pipeline on the active clip.")
        uploaded_custom = st.file_uploader("Match Video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])

        target_video = "data/sample_videos/sample2.mp4" if chosen_match == "sample2" else "data/sample_videos/sample.mp4"
        output_dest = "output_sample2" if chosen_match == "sample2" else "output"
        if uploaded_custom is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_custom.read())
            target_video = tfile.name
            output_dest = "output_custom"

        run_btn = st.button("Run Full Pipeline", type="primary", use_container_width=True)

st.markdown('<div class="top-hairline"></div>', unsafe_allow_html=True)

# ----------------- RE-RUN ANALYSIS IF TRIGGERED -----------------
if "run_btn" in locals() and run_btn:
    prog_bar = st.progress(0)
    status_box = st.empty()

    def progress_callback(c, t, msg):
        pct = min(1.0, max(0.0, c / t))
        prog_bar.progress(pct)
        status_box.caption(f"{msg} ({int(pct * 100)}%)")

    try:
        with st.spinner("Running high-resolution detection, GMC BoT-SORT tracking, and ball telemetry..."):
            from main import run_pipeline
            pipeline_res = run_pipeline(
                video_path=target_video,
                output_dir=output_dest,
                max_frames=0,
                model_name=selected_model_path,
                confidence=0.14,
                imgsz=1280,
                progress_callback=progress_callback,
            )
            st.session_state["results"] = pipeline_res
            prog_bar.empty()
            status_box.empty()
            st.success("Analysis completed across all frames.")
            st.rerun()
    except Exception as e:
        st.error(f"Error during video processing: {e}")


# =============================================================================
# MAIN WORKSPACE
# =============================================================================
results = st.session_state.get("results")
if results is None or results.get("spatial_analytics") is None:
    st.info("No match dataset loaded. Open Upload / Run in the top bar to process a video.")
    st.stop()

spatial_analytics: SpatialAnalytics = results["spatial_analytics"]
pitch_view: PitchView2D = results.get("pitch_view") or PitchView2D(1050, 680)
avg_pos_2d = spatial_analytics.get_average_positions_2d(min_frames=1, max_per_team=11)

# Shared Phase 2 & Phase 3 engines (computed once per rerun)
event_engine = MatchEventEngine(spatial_analytics, fps=25.0)
physical_engine = PhysicalPerformanceEngine(spatial_analytics, fps=25.0)
tactical_engine = TacticalDynamicsEngine(spatial_analytics, event_engine=event_engine, fps=25.0)

tab_team, tab_events, tab_player, tab_media = st.tabs([
    "Team Analytics",
    "Passing & Events",
    "Player Deep-Dive",
    "Match Footage & Data",
])


# =============================================================================
# TAB 1: TEAM TACTICAL ANALYTICS
# =============================================================================
with tab_team:
    total_f = max(1, spatial_analytics.total_processed_frames)
    ball_cov = min(100.0, 100.0 * len(spatial_analytics.ball_track) / total_f)
    t_row_l, t_row_r = st.columns([2.2, 2.0], vertical_alignment="center")
    with t_row_l:
        team_view_mode = st.radio(
            "Team Perspective",
            ["Team 0 (Home)", "Team 1 (Away)", "Comparative 11v11"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with t_row_r:
        cal_tag = "Optical GMC · FIFA Calibrated" if spatial_analytics.has_metric_calibration() else "2D Perspective Est."
        st.markdown(
            f'<div class="meta-pill"><span class="dot dot-accent"></span>'
            f'{total_f / 25.0:.1f}s &nbsp;·&nbsp; {ball_cov:.0f}% ball lock &nbsp;·&nbsp; {cal_tag}</div>',
            unsafe_allow_html=True,
        )

    if "Comparative" in team_view_mode:
        t0 = spatial_analytics.get_team_metrics(0, min_frames=1)
        t1 = spatial_analytics.get_team_metrics(1, min_frames=1)

        render_kpi_strip([
            ("Team 0 Squad", f"{t0['player_count']}", "1 GK + 10 Outfield · Home", "dot-home"),
            ("Team 0 Compactness", f"{t0['team_depth_m']} m", f"Width {t0['team_width_m']} m", "dot-home"),
            ("Team 1 Squad", f"{t1['player_count']}", "1 GK + 10 Outfield · Away", "dot-away"),
            ("Team 1 Compactness", f"{t1['team_depth_m']} m", f"Width {t1['team_width_m']} m", "dot-away"),
        ])

        t_left, t_right = st.columns(2, gap="medium")
        with t_left:
            render_panel_label("Average 2D Formations", "11 vs 11 · FIFA 105×68m")
            all_form = pitch_view.draw_2d_formations(avg_pos_2d, spatial_analytics.player_teams)
            st.image(cv2.cvtColor(all_form, cv2.COLOR_BGR2RGB), use_column_width=True)

        with t_right:
            render_panel_label("Collective Pitch Heatmap", "All tracked players")
            all_heat = pitch_view.generate_2d_heatmap(
                spatial_analytics.get_2d_positions(min_frames=1), sigma=28.0
            )
            st.image(cv2.cvtColor(all_heat, cv2.COLOR_BGR2RGB), use_column_width=True)

        # ---- Phase 3: Comparative Tactical Shape Dynamics & Pressing ----
        td0, td1 = tactical_engine.summary[0], tactical_engine.summary[1]
        render_section_header(
            "02",
            "Structural Dynamics & Pressing Intensity",
            "Time-series defensive line height, outfield compactness, and nearest-opponent ball pressure",
        )
        render_kpi_strip([
            ("Team 0 Def. Line", f"{td0['avg_def_line_m']} m", f"Block {td0['avg_block_length_m']}m · Gap {td0['avg_def_mid_gap_m']}m", "dot-home"),
            ("Team 1 Def. Line", f"{td1['avg_def_line_m']} m", f"Block {td1['avg_block_length_m']}m · Gap {td1['avg_def_mid_gap_m']}m", "dot-away"),
            ("Team 0 Pressing", f"{td0['avg_press_dist_m']} m", f"High press (≤4.5m): {td0['high_press_pct']}%", "dot-home"),
            ("Team 1 Pressing", f"{td1['avg_press_dist_m']} m", f"High press (≤4.5m): {td1['high_press_pct']}%", "dot-away"),
        ])
        dyn_l, dyn_r = st.columns(2, gap="medium")
        with dyn_l:
            st.image(render_tactical_shape_chart(tactical_engine, target_team=None), use_column_width=True)
        with dyn_r:
            st.image(render_pressing_chart(tactical_engine, target_team=None), use_column_width=True)

    else:
        target_team = 0 if "Team 0" in team_view_mode else 1
        tm = spatial_analytics.get_team_metrics(target_team, min_frames=1)
        dot_c = "dot-home" if target_team == 0 else "dot-away"
        t_lbl = "Team 0 (Home)" if target_team == 0 else "Team 1 (Away)"

        render_kpi_strip([
            ("Active Squad", f"{tm['player_count']}", f"{t_lbl} · 1 GK + 10 Outfield", dot_c),
            ("Team Centroid", f"{tm['centroid_m'][0]}m, {tm['centroid_m'][1]}m", "Average pitch X / Y anchor", None),
            ("Block Depth", f"{tm['team_depth_m']} m", "Vertical compactness", None),
            ("Block Width", f"{tm['team_width_m']} m", "Horizontal spread", None),
        ])

        t_col1, t_col2 = st.columns(2, gap="medium")
        with t_col1:
            render_panel_label("Tactical Shape & Centroid", t_lbl)
            team_form_img = pitch_view.draw_2d_team_formation(
                target_team, avg_pos_2d, spatial_analytics.player_teams
            )
            st.image(cv2.cvtColor(team_form_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        with t_col2:
            render_panel_label("Collective Occupancy Heatmap", t_lbl)
            team_heat_img = pitch_view.generate_2d_heatmap(
                spatial_analytics.get_2d_positions(team_id=target_team, min_frames=1), sigma=28.0
            )
            st.image(cv2.cvtColor(team_heat_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        bar_col = "#E5484D" if target_team == 0 else "#3B82F6"
        render_distribution_strip([
            ("Channel Preference", [
                ("Left Flank", tm["left_flank_pct"], "rgba(255,255,255,0.22)"),
                ("Central", tm["center_pct"], bar_col),
                ("Right Flank", tm["right_flank_pct"], "rgba(255,255,255,0.45)"),
            ]),
            ("Pitch Depth Distribution", [
                ("Defensive", tm["defensive_pct"], "rgba(255,255,255,0.22)"),
                ("Middle", tm["middle_pct"], bar_col),
                ("Attacking", tm["attacking_pct"], "#10B981"),
            ]),
        ])

        # ---- Phase 3: Team Structural Dynamics & Pressing ----
        td_s = tactical_engine.summary[target_team]
        phys_t = physical_engine.get_team_physical_summary(target_team)
        render_section_header(
            "02",
            f"{t_lbl} — Structural Dynamics & Pressing",
            f"Team distance {phys_t['total_distance_m']}m · HSR {phys_t['hsr_distance_m']}m · {phys_t['total_sprints']} sprints",
        )
        render_kpi_strip([
            ("Defensive Line", f"{td_s['avg_def_line_m']} m", "Mean height from own goal", dot_c),
            ("Outfield Block", f"{td_s['avg_block_length_m']} m", f"Avg width {td_s['avg_block_width_m']} m", None),
            ("Def–Mid Gap", f"{td_s['avg_def_mid_gap_m']} m", "Vertical line separation", None),
            ("Press Distance", f"{td_s['avg_press_dist_m']} m", f"High press (≤4.5m): {td_s['high_press_pct']}%", "dot-accent"),
        ])
        dyn_l, dyn_r = st.columns(2, gap="medium")
        with dyn_l:
            st.image(render_tactical_shape_chart(tactical_engine, target_team=target_team), use_column_width=True)
        with dyn_r:
            st.image(render_pressing_chart(tactical_engine, target_team=target_team), use_column_width=True)


# =============================================================================
# TAB 2: PASSING & MATCH EVENTS
# =============================================================================
with tab_events:
    poss = event_engine.get_possession_summary()

    render_possession_bar(poss)

    render_kpi_strip([
        ("Total Passes", f"{poss['total_passes']} / {poss['total_passes_attempted']}",
         f"{poss['total_incomplete_passes']} intercepted or incomplete", "dot-accent"),
        ("Pass Accuracy", f"{poss['total_pass_accuracy']}%",
         f"Home {poss['team0_pass_accuracy']}% · Away {poss['team1_pass_accuracy']}%", None),
        ("Avg Pass Length", f"{poss['avg_pass_length_m']} m",
         "Calibrated on FIFA 105×68m pitch", None),
        ("Avg Ball Speed", f"{poss['avg_pass_speed_kmh']} km/h",
         "Kick-to-reception flight velocity", None),
    ])

    all_passes = event_engine.all_passes
    if not all_passes:
        st.info("No passes detected in this clip.")
    else:
        render_section_header(
            "01",
            "Pass-by-Pass Tactical Inspector",
            "Freeze-frame 2D pitch state paired with broadcast verification",
            tight=True,
        )

        # Clean single-row toolbar: Team | Outcome | Prev | Pass Selector | Next
        c_team, c_out, c_prev, c_sel, c_next = st.columns(
            [1.15, 1.15, 0.36, 3.05, 0.36], vertical_alignment="bottom"
        )
        with c_team:
            pass_team_filter = st.selectbox(
                "Team",
                ["All Teams", "Team 0 (Home)", "Team 1 (Away)"],
                key="pass_team_filter",
            )
        with c_out:
            pass_outcome_filter = st.selectbox(
                "Outcome",
                ["All Outcomes", "Completed", "Incomplete"],
                key="pass_outcome_filter",
            )

        filtered_passes = [
            p for p in all_passes
            if (
                pass_team_filter == "All Teams"
                or (p["team_id"] == 0 and "Team 0" in pass_team_filter)
                or (p["team_id"] == 1 and "Team 1" in pass_team_filter)
            )
            and (
                pass_outcome_filter == "All Outcomes"
                or (p["is_completed"] and pass_outcome_filter == "Completed")
                or ((not p["is_completed"]) and pass_outcome_filter == "Incomplete")
            )
        ]

        if not filtered_passes:
            st.warning("No passes match the active filter combination.")
        else:
            def _fmt_pass(p: Dict) -> str:
                arrow = f"#{p['player_id']} → #{p['target_id']}" if p["is_completed"] else f"#{p['player_id']} ✕ #{p['target_id']}"
                status = "Completed" if p["is_completed"] else "Intercepted"
                return f"Pass #{p['pass_number']:02d}  ·  {p['time_sec']:.1f}s  ·  T{p['team_id']} {arrow}  ·  {status}  ·  {p['length_m']}m  ·  {p['avg_speed_kmh']:.0f} km/h"

            pass_by_num = {p["pass_number"]: p for p in filtered_passes}
            nums = list(pass_by_num.keys())
            if st.session_state.get("pass_sel_num") not in pass_by_num:
                st.session_state["pass_sel_num"] = nums[0]

            def _step_pass(delta: int):
                cur = st.session_state.get("pass_sel_num")
                idx = nums.index(cur) if cur in nums else 0
                st.session_state["pass_sel_num"] = nums[(idx + delta) % len(nums)]

            cur_idx = nums.index(st.session_state["pass_sel_num"]) + 1

            with c_prev:
                st.button("←", on_click=_step_pass, args=(-1,), key="pass_prev", help="Previous pass", use_container_width=True)
            with c_sel:
                sel_num = st.selectbox(
                    f"Inspect Pass ({cur_idx} of {len(nums)})",
                    nums,
                    format_func=lambda n: _fmt_pass(pass_by_num[n]),
                    key="pass_sel_num",
                )
            with c_next:
                st.button("→", on_click=_step_pass, args=(1,), key="pass_next", help="Next pass", use_container_width=True)

            selected_pass = pass_by_num[sel_num]

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            col_p2d, col_pinfo = st.columns([1.68, 1.0], gap="medium")
            with col_p2d:
                single_pass_img = pitch_view.draw_2d_single_pass(selected_pass)
                st.image(cv2.cvtColor(single_pass_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            with col_pinfo:
                render_pass_telemetry_card(selected_pass)

                v_path = results.get("output_video_path")
                if v_path and os.path.exists(v_path):
                    cap_snap = cv2.VideoCapture(v_path)
                    cap_snap.set(cv2.CAP_PROP_POS_FRAMES, selected_pass["mid_frame"])
                    ok_fr, cam_snap = cap_snap.read()
                    cap_snap.release()
                    if ok_fr and cam_snap is not None:
                        render_panel_label("Broadcast Verification", f"t = {selected_pass['mid_frame']/25.0:.1f}s · Frame {selected_pass['mid_frame']}")
                        st.image(cv2.cvtColor(cam_snap, cv2.COLOR_BGR2RGB), use_column_width=True)

        # ---------------- 02 / ALL PASSES OVERVIEW MAP ----------------
        render_section_header(
            "02",
            "All Passes — Match Overview Map",
            "Numbered circles match Pass # in the log · Green = completed · Dashed red ✕ = intercepted",
        )
        ov_ctrl, ov_spacer = st.columns([1.6, 3.4])
        with ov_ctrl:
            ov_team = st.radio(
                "Overview Filter",
                ["Both Teams", "Team 0 (Home)", "Team 1 (Away)"],
                horizontal=True,
                key="overview_team",
                label_visibility="collapsed",
            )
        ov_filter = None if ov_team == "Both Teams" else (0 if "Team 0" in ov_team else 1)
        ov_img = pitch_view.draw_2d_all_passes_overview(all_passes, team_filter=ov_filter)

        ov_l, ov_c, ov_r = st.columns([0.18, 2.64, 0.18])
        with ov_c:
            st.image(cv2.cvtColor(ov_img, cv2.COLOR_BGR2RGB), use_column_width=True)

        # ---------------- 03 / COMPLETE PASS LOG TABLE ----------------
        render_section_header(
            "03",
            "Complete Pass Telemetry Log",
            f"{len(all_passes)} passes recorded across the match clip",
        )
        pass_rows = event_engine.get_passes_table()
        if pass_rows:
            pass_df = pd.DataFrame(pass_rows)
            if "Outcome" in pass_df.columns:
                pass_df["Outcome"] = pass_df["Outcome"].str.replace("✅ ", "", regex=False).str.replace("❌ ", "", regex=False)
            active_pnum = selected_pass["pass_number"] if filtered_passes else None
            render_passes_table_html(all_passes, active_pass_num=active_pnum)

            dl_col, _ = st.columns([1.2, 4.8])
            with dl_col:
                st.download_button(
                    "Export Passes CSV",
                    data=pass_df.to_csv(index=False).encode("utf-8"),
                    file_name="footballistic_passes_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# =============================================================================
# TAB 3: PLAYER DEEP-DIVE ANALYTICS
# =============================================================================
with tab_player:
    all_pids = sorted(list(avg_pos_2d.keys()))

    if not all_pids:
        st.warning("No tracked players found in active dataset.")
    else:
        col_filter, col_select = st.columns([1.1, 2.4], gap="medium")
        with col_filter:
            team_filter = st.selectbox(
                "Filter Team",
                ["All Teams", "Team 0 (Home)", "Team 1 (Away)"],
                index=0,
            )

        if "Team 0" in team_filter:
            filtered_ids = [p for p in all_pids if spatial_analytics.player_teams.get(p) == 0]
        elif "Team 1" in team_filter:
            filtered_ids = [p for p in all_pids if spatial_analytics.player_teams.get(p) == 1]
        else:
            filtered_ids = all_pids

        with col_select:
            if not filtered_ids:
                st.warning("No players found for this team.")
                selected_pid = None
            else:
                def is_player_gk(pid: int) -> bool:
                    if pid in (1, 13) or str(pid) in ("1", "13"):
                        return True
                    pts = spatial_analytics.player_tracks_2d.get(pid, [])
                    if pts:
                        avg_px = np.mean(pts, axis=0)
                        xm = (avg_px[0] - spatial_analytics.FIELD_X1) * (105.0 / max(1.0, spatial_analytics.FIELD_W))
                        ym = (avg_px[1] - spatial_analytics.FIELD_Y1) * (68.0 / max(1.0, spatial_analytics.FIELD_H))
                        tid = spatial_analytics.player_teams.get(pid, 0)
                        if (tid == 0 and xm <= 16.5 and abs(ym - 34.0) <= 16.0) or (tid == 1 and xm >= 88.5 and abs(ym - 34.0) <= 16.0):
                            return True
                    return False

                def format_player_label(pid: int) -> str:
                    t = spatial_analytics.player_teams.get(pid, 0)
                    t_str = "Team 0 · Home" if t == 0 else "Team 1 · Away"
                    role = " (GK)" if is_player_gk(pid) else ""
                    return f"Player #{pid}{role}  —  {t_str}"

                selected_pid = st.selectbox(
                    "Select Player",
                    filtered_ids,
                    format_func=format_player_label,
                )

        if selected_pid is not None:
            pm = spatial_analytics.get_player_metrics(selected_pid)
            pp = physical_engine.get_player_profile(selected_pid)
            team_id = pm["team_id"]
            is_keeper = pm.get("is_goalkeeper", is_player_gk(selected_pid))
            p_dot = "dot-home" if team_id == 0 else "dot-away"
            p_label = f"#{selected_pid} GK" if is_keeper else f"#{selected_pid}"

            render_kpi_strip([
                ("Player Profile", p_label, f"Team {team_id} · {pm['tactical_zone']}", p_dot),
                ("Distance Covered", f"{pm['distance_m']} m", f"{pm['frames_active']} frames · {pm['presence_pct']}% presence", None),
                ("Avg / Top Speed", f"{pp['avg_speed_kmh']} / {pp['top_speed_kmh']} km/h", f"Peak accel {pp['peak_accel_ms2']} m/s²", None),
                ("High-Speed Load", f"{pp['hsr_distance_m']} m HSR", f"{pp['sprint_distance_m']} m sprint · {pp['sprint_count']} sprints", "dot-accent"),
            ])

            p_col1, p_col2 = st.columns(2, gap="medium")
            with p_col1:
                render_panel_label("Movement Trajectory & Anchor", f"Player {p_label}")
                focus_pitch = pitch_view.draw_2d_player_focus(
                    tracker_id=selected_pid,
                    avg_pos_2d=pm["avg_pos_2d"],
                    team_id=team_id,
                    trail_2d=pm["positions_2d"],
                    teammate_positions=None,
                )
                st.image(cv2.cvtColor(focus_pitch, cv2.COLOR_BGR2RGB), use_column_width=True)

            with p_col2:
                render_panel_label("Individual Spatial Heatmap", f"Player {p_label}")
                p_heat = pitch_view.generate_2d_heatmap(pm["positions_2d"], sigma=22.0)
                st.image(cv2.cvtColor(p_heat, cv2.COLOR_BGR2RGB), use_column_width=True)

            bar_col = "#E5484D" if team_id == 0 else "#3B82F6"
            render_distribution_strip([
                ("Pitch Thirds Breakdown", [
                    ("Defensive", pm["defensive_pct"], "rgba(255,255,255,0.22)"),
                    ("Middle Third", pm["middle_pct"], bar_col),
                    ("Attacking", pm["attacking_pct"], "#10B981"),
                ]),
            ])

            # ---- Phase 3: Speed Zones & Velocity Profile (Compact Side-by-Side) ----
            cal_note = "Optical-Flow GMC + FIFA Height Scale" if spatial_analytics.has_metric_calibration() else "Smoothed 2D Pitch Scale"
            render_section_header(
                "02",
                "Speed Profile & Intensity Zones",
                f"{cal_note} · HSR > 18 km/h · Sprint ≥ 21 km/h",
            )
            sp_left, sp_right = st.columns([1.35, 1.0], gap="medium")
            with sp_left:
                st.image(render_player_speed_chart(pp, team_id), use_column_width=True)
            with sp_right:
                render_speed_zone_panel(pp)

            # ---- Phase 3: Match Physical Leaderboard ----
            lb_filter = 0 if "Team 0" in team_filter else (1 if "Team 1" in team_filter else None)
            lb_rows = physical_engine.get_physical_leaderboard(team_filter=lb_filter)
            render_section_header(
                "03",
                "Match Physical Leaderboard",
                f"Active player (#{selected_pid}) highlighted · {len(lb_rows)} players tracked",
            )
            if lb_rows:
                lb_s1, lb_s2 = st.columns([1.3, 4.7], vertical_alignment="bottom")
                with lb_s1:
                    sort_col = st.selectbox(
                        "Sort Leaderboard",
                        ["Top Speed (km/h)", "Distance (m)", "HSR >18 (m)", "Sprints", "Avg Speed (km/h)"],
                        index=0,
                        key="phys_lb_sort",
                    )
                lb_sorted = sorted(lb_rows, key=lambda r: (-float(r[sort_col]), -float(r["Distance (m)"])))
                render_physical_leaderboard_html(lb_sorted, active_pid=selected_pid)

                lb_df = pd.DataFrame(lb_sorted)
                dl_p, _ = st.columns([1.3, 4.7])
                with dl_p:
                    st.download_button(
                        "Export Physical CSV",
                        data=lb_df.to_csv(index=False).encode("utf-8"),
                        file_name="footballistic_physical_metrics.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


# =============================================================================
# TAB 4: MATCH FOOTAGE & DATASET
# =============================================================================
with tab_media:
    col_v, col_d = st.columns([1.15, 1.0], gap="medium")

    with col_v:
        render_panel_label("Annotated Match Replay", "H.264 · GMC Tracking + White Ball Trail")
        v_path = results.get("output_video_path")
        if v_path and os.path.exists(v_path):
            with open(v_path, "rb") as f:
                st.video(f.read())
        else:
            st.info("Annotated video has not been rendered yet.")

    with col_d:
        render_panel_label("Spatial Player Dataset", "11 vs 11 FIFA Standard")
        summary_rows = spatial_analytics.get_player_summary(min_frames=1)
        if summary_rows:
            render_spatial_summary_html(summary_rows)
            df = pd.DataFrame(summary_rows)

            dl_c, _ = st.columns([1.4, 2.6])
            with dl_c:
                st.download_button(
                    "Export Analytics CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="footballistic_match_analytics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
