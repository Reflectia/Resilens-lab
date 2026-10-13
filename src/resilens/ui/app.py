"""Streamlit Web UI"""

import streamlit as st

from resilens.ui.api_client import ApiClient

MODES = ["A", "B", "C", "D", "E"]
PROFILES = ["minimal", "standard", "performance"]


def render_run_form(client: ApiClient) -> None:
    pass


def render_results(client: ApiClient) -> None:
    pass


def render_comparison(client: ApiClient) -> None:
    pass


def main() -> None:
    st.set_page_config(page_title="Resilens", layout="wide")
    st.title("Resilens v2.0")
    client = ApiClient()
    run_tab, results_tab, compare_tab = st.tabs(["Run", "Results", "Comparison"])
    with run_tab:
        render_run_form(client)
    with results_tab:
        render_results(client)
    with compare_tab:
        render_comparison(client)


if __name__ == "__main__":
    main()
