from typing import Dict, List, Optional, Tuple
from datetime import date

import streamlit as st

import settings


def configure_page() -> None:
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon=settings.APP_PAGE_ICON,
        layout="wide",
    )


def render_header() -> None:
    st.title(f"{settings.APP_PAGE_ICON} {settings.APP_TITLE}")
    st.markdown(
        """
        Design a region, select dates, tune cloud tolerance, and generate a Sentinel-2 time-lapse.
        """
    )


def sidebar_controls() -> Dict:
    with st.sidebar:
        st.subheader("Configuration")

        start_date: date = st.date_input(
            "Start date",
            value=settings.DEFAULT_START_DATE
        )
        end_date: date = st.date_input(
            "End date",
            value=settings.DEFAULT_END_DATE
        )

        st.markdown("### Cloud tolerance")
        ignore_cloud_filter = st.checkbox(
            "Do not filter by cloud coverage",
            value=False,
            help="If checked, all scenes are considered regardless of cloud percentage."
        )

        max_cloud_cover = st.slider(
            "Maximum cloud cover (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="Upper limit for eo:cloud_cover when querying the catalog."
        )

        st.markdown("### Visual mode")
        mode = st.selectbox(
            "Visualization",
            settings.VISUAL_MODES
        )

        custom_bands: Optional[List[str]] = None
        if mode == "Custom RGB":
            custom_bands = [
                st.selectbox("Red band", settings.BAND_OPTIONS, index=3, key="red_band"),
                st.selectbox("Green band", settings.BAND_OPTIONS, index=2, key="green_band"),
                st.selectbox("Blue band", settings.BAND_OPTIONS, index=1, key="blue_band"),
            ]

        st.markdown("### Time-lapse settings")
        fps = st.slider(
            "Frames per second",
            min_value=1,
            max_value=settings.MAX_FPS,
            value=settings.DEFAULT_FPS,
            help="Playback speed of the exported video."
        )

        apply_cv = st.checkbox(
            "Run NDVI analysis per frame",
            value=True,
            help="Computes a simple NDVI-based time series as a basic computer vision analytics layer."
        )

    return {
        "start_date": start_date,
        "end_date": end_date,
        "ignore_cloud_filter": ignore_cloud_filter,
        "max_cloud_cover": max_cloud_cover,
        "mode": mode,
        "custom_bands": custom_bands,
        "fps": fps,
        "apply_cv": apply_cv,
    }


def two_column_layout() -> Tuple:
    return st.columns([2, 1])
