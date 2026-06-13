"""
Streamlit UI for the License Plate Detection Pipeline.

Connects to the FastAPI backend at API_BASE_URL.
Run with: streamlit run app.py
"""

import os
import time

import requests
import streamlit as st

st.set_page_config(
    page_title="Plate Detection Pipeline",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# THEME / GLOBAL STYLES
# ──────────────────────────────────────────────
ACCENT = "#22D3EE"        # cyan accent
ACCENT_DIM = "#0E7490"
BG = "#0B0F14"            # near-black terminal bg
PANEL = "#11161D"
PANEL_BORDER = "#1F2A36"
TEXT = "#D7E1E8"
TEXT_DIM = "#7C8B99"
GOOD = "#34D399"
WARN = "#FBBF24"
BAD = "#F87171"

ICON = {
    "status": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/></svg>',
    "ocr": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7V4h16v3M9 20h6M12 4v16"/></svg>',
    "layers": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "settings": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    "upload": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "video": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>',
    "broadcast": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.93 19.07A10 10 0 1 1 19.07 19.07"/><path d="M7.76 16.24a6 6 0 1 1 8.49 0"/><circle cx="12" cy="12" r="2"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "car": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 17h14M5 17a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm14 0a2 2 0 1 1-4 0 2 2 0 0 1 4 0zM3 17V9l2-5h14l2 5v8"/></svg>',
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "stop": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>',
    "play": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "terminal": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>',
    "x": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    "moon": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
}


def icon(name, color=None):
    color = color or ACCENT
    return f'<span style="display:inline-flex;vertical-align:middle;color:{color};margin-right:6px;">{ICON[name]}</span>'


CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {PANEL_BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {{
    color: {TEXT};
}}

/* Headings */
h1, h2, h3 {{
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}}
h1 {{
    color: {TEXT};
    font-weight: 700;
    border-bottom: 1px solid {PANEL_BORDER};
    padding-bottom: 0.6rem;
}}
h2, h3 {{
    color: {ACCENT};
    font-weight: 600;
    font-size: 1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* Top brand bar */
.brand-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.2rem;
    background: linear-gradient(180deg, #131A22 0%, {PANEL} 100%);
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    margin-bottom: 1.2rem;
}}
.brand-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: {TEXT};
    display: flex;
    align-items: center;
    gap: 0.6rem;
    letter-spacing: 0.04em;
}}
.brand-sub {{
    color: {TEXT_DIM};
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
}}
.status-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border: 1px solid {PANEL_BORDER};
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.pill-ok {{ color: {GOOD}; background: rgba(52,211,153,0.08); border-color: rgba(52,211,153,0.25); }}
.pill-bad {{ color: {BAD}; background: rgba(248,113,113,0.08); border-color: rgba(248,113,113,0.25); }}
.pill-warn {{ color: {WARN}; background: rgba(251,191,36,0.08); border-color: rgba(251,191,36,0.25); }}
.pill-neutral {{ color: {ACCENT}; background: rgba(34,211,238,0.08); border-color: rgba(34,211,238,0.25); }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_DIM};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-radius: 8px;
    padding: 0.5rem 1rem;
}}
.stTabs [aria-selected="true"] {{
    background-color: rgba(34,211,238,0.10) !important;
    color: {ACCENT} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {ACCENT};
}}
.stTabs [data-baseweb="tab-border"] {{
    background-color: {PANEL_BORDER};
}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background-color: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 0.9rem 1rem;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_DIM};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT};
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.8rem !important;
    line-height: 1.8 !important;
    letter-spacing: 0.01em;
    word-break: break-word;
}}

/* Panels / containers */
.panel {{
    background-color: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
}}
.panel-title {{
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
}}
.kv-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.32rem 0;
    border-bottom: 1px dashed {PANEL_BORDER};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}}
.kv-row:last-child {{ border-bottom: none; }}
.kv-key {{ color: {TEXT_DIM}; }}
.kv-val {{ color: {TEXT}; font-weight: 600; }}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {{
    background-color: rgba(34,211,238,0.10);
    color: {ACCENT};
    border: 1px solid {ACCENT_DIM};
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    transition: all 0.15s ease;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    background-color: {ACCENT};
    color: #06222B;
    border-color: {ACCENT};
}}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
    background-color: #0E1420 !important;
    color: {TEXT} !important;
    border: 1px solid {PANEL_BORDER} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 1px {ACCENT}33 !important;
}}

/* Checkbox label */
.stCheckbox label p {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: {TEXT};
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background-color: #0E1420;
    border: 1px dashed {PANEL_BORDER};
    border-radius: 10px;
}}

/* Expander */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {{
    background-color: #0E1420 !important;
    border: 1px solid {PANEL_BORDER} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: {TEXT} !important;
    font-size: 0.82rem !important;
}}
[data-testid="stExpander"] {{
    border: none !important;
    background: transparent !important;
}}

/* Captions / small text */
.stCaption, .css-1aehpvj {{
    font-family: 'JetBrains Mono', monospace !important;
    color: {TEXT_DIM} !important;
    font-size: 0.72rem !important;
}}

/* Divider */
hr {{ border-color: {PANEL_BORDER} !important; }}

/* Plate tag */
.plate-tag {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    color: #06222B;
    background-color: {ACCENT};
    padding: 0.18rem 0.55rem;
    border-radius: 6px;
}}

/* Reduce top padding */
.block-container {{
    padding-top: 1.4rem;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {PANEL_BORDER}; border-radius: 4px; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def pill(text, kind="neutral"):
    return f'<span class="status-pill pill-{kind}">{text}</span>'


def kv_row(key, val):
    return f'<div class="kv-row"><span class="kv-key">{key}</span><span class="kv-val">{val}</span></div>'


def panel_open(title, icon_name=None):
    title_html = f"{icon(icon_name) if icon_name else ''}{title}"
    return f'<div class="panel"><div class="panel-title">{title_html}</div>'


PANEL_CLOSE = "</div>"


# SIDEBAR

with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">'
        f'{icon("settings")}<span style="font-family:\'JetBrains Mono\',monospace;'
        f'font-weight:700;font-size:0.95rem;letter-spacing:0.06em;color:{TEXT};">'
        f'CONFIGURATION</span></div>',
        unsafe_allow_html=True,
    )

    api_url = st.text_input(
        "API base URL",
        value=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="The FastAPI backend URL",
    )

    auto_refresh = st.checkbox(
        "Auto-refresh dashboard",
        value=False,
        help="Automatically refresh the dashboard every 5 seconds",
    )

    dark_mode = st.checkbox(
        "Reduced motion",
        value=False,
        help="Disable auto-refresh spinners and animations",
    )

    st.markdown("---")
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.72rem;'
        f'color:{TEXT_DIM};line-height:1.6;">'
        f'CONNECTED TO<br><span style="color:{ACCENT};">{api_url}</span></div>',
        unsafe_allow_html=True,
    )

# BRAND BAR

st.markdown(
    f'<div class="brand-bar">'
    f'<div>'
    f'<div class="brand-title">{icon("terminal")}License Plate Detection</div>'
    f'<div class="brand-sub">License plate detection &amp; recognition console</div>'
    f'</div>'
    f'{pill(icon("status", "currentColor") + "LIVE", "neutral")}'
    f'</div>',
    unsafe_allow_html=True,
)

tab_dashboard, tab_upload, tab_stream, tab_results = st.tabs(
    ["Dashboard", "Upload Video", "Stream Ingestion", "Results"]
)

# DASHBOARD TAB

with tab_dashboard:
    st.markdown(
        f'<h3>{icon("status")}System Status</h3>',
        unsafe_allow_html=True,
    )

    status_placeholder = st.empty()

    while True:
        with status_placeholder.container():
            with st.spinner("Fetching system status..."):
                try:
                    resp = requests.get(f"{api_url}/status", timeout=5)
                    resp.raise_for_status()
                    status = resp.json()
                except requests.RequestException as e:
                    st.markdown(
                        f'<div class="panel" style="border-color:rgba(248,113,113,0.35);">'
                        f'<div class="panel-title" style="color:{BAD};">'
                        f'{icon("alert", BAD)}Connection error</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;'
                        f'color:{TEXT_DIM};">Could not reach API at <span style="color:{TEXT};">'
                        f'{api_url}</span><br>{e}</div></div>',
                        unsafe_allow_html=True,
                    )
                    if not auto_refresh:
                        break
                    time.sleep(5)
                    continue

            status_kind = "ok" if status.get("status", "").lower() in ("ok", "healthy", "running") else "warn"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Status", status.get("status", "—"))
            col2.metric("Inference Mode", status.get("inference_mode", "—"))
            col3.metric("OCR Engine", status.get("ocr_engine", "—"))
            col4.metric("Processing Mode", status.get("processing_mode", "—"))

            st.write("")

            comp_col, cfg_col = st.columns([1, 1])

            with comp_col:
                rows = ""
                for key, val in status.get("components", {}).items():
                    state = str(val).lower() in ("true", "ok", "running", "healthy", "1", "up")
                    if isinstance(val, bool):
                        badge = pill(
                            f'{icon("check", "currentColor")}UP' if state else f'{icon("x", "currentColor")}DOWN',
                            "ok" if state else "bad",
                        )
                    else:
                        badge = f'<span class="kv-val">{val}</span>'
                    label = key.replace("_", " ").title()
                    rows += f'<div class="kv-row"><span class="kv-key">{label}</span>{badge}</div>'
                st.markdown(
                    panel_open("Components", "layers") + rows + PANEL_CLOSE,
                    unsafe_allow_html=True,
                )

            with cfg_col:
                rows = ""
                for key, val in status.get("config", {}).items():
                    rows += kv_row(key.replace("_", " ").title(), val)
                st.markdown(
                    panel_open("Configuration", "settings") + rows + PANEL_CLOSE,
                    unsafe_allow_html=True,
                )

        if not auto_refresh:
            break
        time.sleep(5)


# UPLOAD VIDEO TAB

with tab_upload:
    st.markdown(
        f'<h3>{icon("upload")}Upload a Video File</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;'
        f'color:{TEXT_DIM};margin-bottom:0.8rem;">'
        f'Submit a recorded clip to the ingestion pipeline for plate detection.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "mpg", "mpeg", "ts", "3gp"],
    )

    if uploaded_file is not None:
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;'
            f'color:{TEXT_DIM};margin:0.4rem 0;">{icon("video")}{uploaded_file.name} '
            f'&nbsp;·&nbsp; {uploaded_file.size / (1024*1024):.2f} MB</div>',
            unsafe_allow_html=True,
        )

        if st.button("Process video", type="primary"):
            with st.spinner("Uploading and processing video..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{api_url}/ingest/file", files=files, timeout=600)
                    resp.raise_for_status()
                    result = resp.json()
                except requests.RequestException as e:
                    st.markdown(
                        f'<div class="panel" style="border-color:rgba(248,113,113,0.35);">'
                        f'<div class="panel-title" style="color:{BAD};">'
                        f'{icon("alert", BAD)}Processing failed</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;'
                        f'color:{TEXT_DIM};">{e}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()

            st.markdown(
                f'<div class="panel" style="border-color:rgba(52,211,153,0.3);">'
                f'<div class="panel-title" style="color:{GOOD};">'
                f'{icon("check", GOOD)}Done</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.85rem;'
                f'color:{TEXT};">{result.get("message", "Processing complete.")}</div></div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Job ID", result.get("job_id", "—"))
            col2.metric("Frames Sampled", result.get("frames_sampled", 0))
            col3.metric("Plates Detected", result.get("count", len(result.get("plates", []))))

            if result.get("timings"):
                st.markdown(
                    f'<h3 style="margin-top:1rem;">{icon("clock")}Timings</h3>',
                    unsafe_allow_html=True,
                )
                t = result["timings"]
                t_cols = st.columns(3)
                if t.get("detection_ms"):
                    t_cols[0].metric("Detection", f"{t['detection_ms']:.0f} ms")
                if t.get("ocr_ms"):
                    t_cols[1].metric("OCR", f"{t['ocr_ms']:.0f} ms")
                if t.get("llm_ms"):
                    t_cols[2].metric("LLM", f"{t['llm_ms']:.0f} ms")

            plates = result.get("plates", [])
            if plates:
                st.markdown(
                    f'<h3 style="margin-top:1rem;">{icon("car")}Detected Plates ({result.get("count", len(plates))})</h3>',
                    unsafe_allow_html=True,
                )
                for i, plate in enumerate(plates, 1):
                    label = plate.get("text", "N/A")
                    with st.expander(f"Plate #{i}  —  {label}"):
                        rows = "".join(kv_row(k, v) for k, v in plate.items())
                        st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="panel"><div style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:0.82rem;color:{TEXT_DIM};">{icon("info")}'
                    f'No plates detected in the video.</div></div>',
                    unsafe_allow_html=True,
                )

# ──────────────────────────────────────────────
# STREAM INGESTION TAB
# ──────────────────────────────────────────────
with tab_stream:
    st.markdown(
        f'<h3>{icon("broadcast")}Start a New Stream</h3>',
        unsafe_allow_html=True,
    )

    with st.form("stream_form"):
        stream_url = st.text_input(
            "RTSP / RTMP URL",
            placeholder="rtsp://camera_ip:554/stream",
        )
        continuous = st.checkbox("Continuous stream", value=False)
        submitted = st.form_submit_button("Start stream", type="primary")

        if submitted:
            if not stream_url:
                st.markdown(
                    f'<div class="panel" style="border-color:rgba(248,113,113,0.35);">'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;color:{BAD};">'
                    f'{icon("alert", BAD)}Enter a stream URL.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Starting stream ingestion..."):
                    try:
                        resp = requests.post(
                            f"{api_url}/ingest/stream",
                            json={"url": stream_url, "continuous": continuous},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        st.markdown(
                            f'<div class="panel" style="border-color:rgba(52,211,153,0.3);">'
                            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;color:{GOOD};">'
                            f'{icon("check", GOOD)}Stream started — Job ID: <span class="plate-tag">{result["job_id"]}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                    except requests.RequestException as e:
                        st.markdown(
                            f'<div class="panel" style="border-color:rgba(248,113,113,0.35);">'
                            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;color:{BAD};">'
                            f'{icon("alert", BAD)}Failed to start stream: {e}</div></div>',
                            unsafe_allow_html=True,
                        )

    st.markdown(
        f'<h3 style="margin-top:1.4rem;">{icon("layers")}Active Streams</h3>',
        unsafe_allow_html=True,
    )

    try:
        resp = requests.get(f"{api_url}/ingest/streams", timeout=5)
        resp.raise_for_status()
        streams = resp.json()
    except requests.RequestException:
        streams = {}

    if streams:
        for sid, info in streams.items():
            state = str(info.get("status", "")).lower()
            kind = "ok" if state in ("running", "active", "ok") else "warn" if state else "neutral"
            with st.expander(f"{sid}  —  {info.get('status', 'unknown')}"):
                rows = "".join(kv_row(k, v) for k, v in info.items())
                st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)
                if st.button(f"Stop stream", key=f"stop_{sid}"):
                    try:
                        resp = requests.delete(
                            f"{api_url}/ingest/stream/{sid}", timeout=10
                        )
                        resp.raise_for_status()
                        st.markdown(
                            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;color:{GOOD};">'
                            f'{icon("check", GOOD)}Stream {sid} stopped.</div>',
                            unsafe_allow_html=True,
                        )
                        st.rerun()
                    except requests.RequestException as e:
                        st.markdown(
                            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;color:{BAD};">'
                            f'{icon("alert", BAD)}Failed to stop stream: {e}</div>',
                            unsafe_allow_html=True,
                        )
    else:
        st.markdown(
            f'<div class="panel"><div style="font-family:\'JetBrains Mono\',monospace;'
            f'font-size:0.82rem;color:{TEXT_DIM};">{icon("info")}No active streams.</div></div>',
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────
# RESULTS TAB
# ──────────────────────────────────────────────
with tab_results:
    st.markdown(
        f'<h3>{icon("search")}Query Detection Results</h3>',
        unsafe_allow_html=True,
    )

    job_id = st.text_input("Job ID", placeholder="e.g. job_8f23a1")

    if job_id:
        with st.spinner("Fetching results..."):
            try:
                resp = requests.get(f"{api_url}/results/{job_id}", timeout=30)
                if resp.status_code == 404:
                    st.markdown(
                        f'<div class="panel" style="border-color:rgba(251,191,36,0.3);">'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;color:{WARN};">'
                        f'{icon("alert", WARN)}No results found for job {job_id}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.markdown(
                    f'<div class="panel" style="border-color:rgba(248,113,113,0.35);">'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;color:{BAD};">'
                    f'{icon("alert", BAD)}Failed to fetch results: {e}</div></div>',
                    unsafe_allow_html=True,
                )
                st.stop()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", data.get("status", "—"))
        col2.metric("Frames Sampled", data.get("frames_sampled", 0))
        col3.metric("Frames Processed", data.get("frames_processed", 0))
        col4.metric("Plates", data.get("count", len(data.get("plates", []))))

        plates = data.get("plates", [])
        if plates:
            st.markdown(
                f'<h3 style="margin-top:1rem;">{icon("car")}Detected Plates ({data.get("count", len(plates))})</h3>',
                unsafe_allow_html=True,
            )
            for i, plate in enumerate(plates, 1):
                label = plate.get("text", "N/A")
                with st.expander(f"Plate #{i}  —  {label}"):
                    rows = "".join(kv_row(k, v) for k, v in plate.items())
                    st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="panel"><div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:0.82rem;color:{TEXT_DIM};">{icon("info")}No plates in result.</div></div>',
                unsafe_allow_html=True,
            )