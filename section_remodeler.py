"""
section_remodeler.py
═════════════════════
Section 12: Intelligent Code Remodeler UI

Allows users to upload broken .py / .cpp / .hpp strategy files,
automatically fixes structural issues (missing Strategy class, wrong base,
etc.), and outputs a compliant file ready for the backtest engine.
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from config import COLORS
from ui_components import (
    section_header, banner_success, banner_warning,
    banner_error, banner_info, terminal_log,
)
from code_remodeler import remodel_strategy, RemodelerReport

_G = COLORS["green"]
_R = COLORS["red"]
_A = COLORS["amber"]
_B = COLORS["blue"]
_W = COLORS["text"]
_C = COLORS["card_bg"]


def _score_bar(score: float) -> str:
    filled = int(score / 5)
    empty  = 20 - filled
    bar    = "█" * filled + "░" * empty
    color  = _G if score >= 70 else _A if score >= 40 else _R
    return (
        f'<span style="font-family:monospace;color:{color};">'
        f"{bar}  {score:.0f}%</span>"
    )


def _render_report(report: RemodelerReport):
    """Display the full parse / transformation report."""
    status_color = _G if report.success else _R
    status_icon  = "✅" if report.success else "❌"

    st.markdown(
        f'<div style="background:#050505;border:1px solid {"#2A2A2A"};'
        f'border-radius:8px;padding:14px 18px;margin-bottom:10px;">'
        f'<div style="font-family:monospace;font-size:0.82rem;color:{status_color};">'
        f'{status_icon} REMODEL REPORT — {report.strategy_name}'
        f' &nbsp;|&nbsp; {report.source_language.upper()}'
        f' &nbsp;|&nbsp; {report.source_lines:,} lines</div>'
        f'<div style="margin-top:6px;">{_score_bar(report.parse_score)}</div>',
        unsafe_allow_html=True,
    )

    # Transformations
    if report.transformations:
        for t in report.transformations:
            icon  = "✅" if t.startswith("✅") else "⚠️" if t.startswith("⚠") \
                    else "❌" if t.startswith("❌") else "  →"
            color = _G if "✅" in t else _A if "⚠" in t else _R if "❌" in t else "#888"
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.74rem;'
                f'padding:1px 0;color:{color};">{t}</div>',
                unsafe_allow_html=True,
            )

    if report.used_llm_assist:
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.74rem;'
            f'color:{_B};padding:4px 0;">🤖 LLM Assist: {report.llm_assist_notes}</div>',
            unsafe_allow_html=True,
        )

    if report.errors:
        for e in report.errors:
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.72rem;'
                f'color:{_R};padding:1px 0;">{e[:200]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Components table
    if report.components:
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.78rem;'
            f'color:{_G};margin:8px 0 4px 0;">◈ EXTRACTED COMPONENTS</div>',
            unsafe_allow_html=True,
        )
        comp_data = [
            {
                "Type":       c.component_type,
                "Name":       c.name,
                "Confidence": f"{c.confidence:.0%}",
                "Notes":      c.notes[:60] if c.notes else "",
            }
            for c in report.components
        ]
        df = pd.DataFrame(comp_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=200)


def render_section_remodeler(SS: dict):
    section_header(
        "CODE REMODELER",
        "Intelligent auto-fix pipeline — .py / .cpp / .hpp → Strategy(PyStrategyBase)",
    )

    banner_info(
        "Upload any strategy file that failed the terminal's parser. "
        "The remodeler uses **AST analysis + LLM assist** to automatically "
        "extract your trading logic and wrap it into the compliant "
        "`class Strategy(PyStrategyBase):` structure."
    )

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("**Upload Broken Strategy File**")
        st.caption("Accepts .py / .cpp / .hpp / .h up to 10,000 lines. "
                   "All structural issues are auto-resolved.")
        up = st.file_uploader(
            "Drop strategy file here",
            type=["py", "cpp", "hpp", "h", "cxx"],
            key="remodeler_upload",
        )

        use_llm = st.toggle(
            "Enable Claude LLM Assist (for complex files with score < 60%)",
            value=True, key="remodeler_llm",
        )

        if up:
            raw_src = up.read().decode("utf-8", errors="replace")
            size_kb = len(raw_src.encode()) / 1024
            line_count = len(raw_src.splitlines())

            st.markdown(
                f'<span class="pill pill-blue">{up.name}</span>'
                f'<span class="pill">{size_kb:.1f} KB</span>'
                f'<span class="pill">{line_count:,} lines</span>',
                unsafe_allow_html=True,
            )

            if line_count > 10_000:
                banner_warning(
                    f"File has {line_count:,} lines — processing first 10,000. "
                    "Full logic extraction may be partial for very large files."
                )

            # Source preview
            with st.expander("📄 Source Preview (first 60 lines)"):
                preview = "\n".join(raw_src.splitlines()[:60])
                lang = "python" if up.name.endswith(".py") else "cpp"
                st.code(preview, language=lang)

        # Run button
        st.markdown("---")
        run_btn = st.button(
            "🔧  REMODEL STRATEGY",
            type="primary",
            use_container_width=True,
            key="remodel_run",
            disabled=(up is None),
        )

        if run_btn and up:
            with st.spinner(
                f"Analysing {up.name} ({line_count:,} lines) — "
                "extracting logic and remodeling…"
            ):
                report = remodel_strategy(raw_src, up.name, use_llm=use_llm)
                SS["remodeler_report"] = report
                SS["remodeler_source"] = raw_src

            if report.success:
                banner_success(
                    f"**{report.strategy_name}** remodeled successfully — "
                    f"score: **{report.parse_score:.0f}%** | "
                    f"components: **{len(report.components)}** | "
                    f"LLM assist: **{'Yes' if report.used_llm_assist else 'No'}**"
                )
            else:
                banner_error(
                    f"Remodeling failed — score: {report.parse_score:.0f}%. "
                    "See report for details."
                )

    with col2:
        report: RemodelerReport = SS.get("remodeler_report")
        if report:
            _render_report(report)

    # ── Output panel ───────────────────────────────────────────────────────
    report = SS.get("remodeler_report")
    if report and report.remodeled_code:
        st.markdown("---")
        section_header("REMODELED OUTPUT")

        out_tab1, out_tab2 = st.tabs(["📋 Remodeled Code", "📊 Side-by-side Diff"])

        with out_tab1:
            st.code(report.remodeled_code, language="python")

            # ── PRIMARY DOWNLOAD — the fully remodeled, backtest-ready file ──────
            st.markdown(
                f'''<div style="background:#003322;border:2px solid {COLORS["green"]};
                border-radius:10px;padding:16px 20px;margin-bottom:12px;text-align:center;">
                <div style="font-family:monospace;font-size:0.80rem;
                color:{COLORS["green"]};margin-bottom:8px;">
                ✅ REMODEL COMPLETE — {report.strategy_name} · Score {report.parse_score:.0f}%
                · {report.source_lines:,} lines processed
                </div></div>''',
                unsafe_allow_html=True,
            )

            st.download_button(
                label="🔥  download the remodeled file young bull",
                data=report.remodeled_code.encode("utf-8"),
                file_name=f"{report.strategy_name}_remodeled.py",
                mime="text/x-python",
                use_container_width=True,
                type="primary",
                key="dl_young_bull",
                help=(
                    f"Downloads '{report.strategy_name}_remodeled.py' — "
                    "fully compliant class Strategy(PyStrategyBase) file "
                    "ready to upload in Section 2 and backtest immediately."
                ),
            )

            st.markdown("---")
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button(
                    "⬇️  Download Remodeled .py",
                    data=report.remodeled_code.encode(),
                    file_name=f"{report.strategy_name}_remodeled.py",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl2:
                # One-click load into Strategy Loader (Section 2)
                if st.button("⚡  Load into Strategy Loader", use_container_width=True,
                              key="remodel_load"):
                    # Import and load via py_strategy
                    try:
                        from py_strategy import load_py_strategy
                        py_rep = load_py_strategy(report.remodeled_code,
                                                   f"{report.strategy_name}.py")
                        SS["py_source"]         = report.remodeled_code
                        SS["py_parse_report"]   = py_rep
                        SS["strategy_filename"] = f"{report.strategy_name}_remodeled.py"
                        SS["strategy_type"]     = "python"
                        SS["use_demo_strategy"] = False
                        if py_rep.is_valid:
                            SS["parse_report"] = type("R", (), {
                                "strategy":    py_rep.strategy_def,
                                "line_count":  py_rep.line_count,
                                "parse_score": py_rep.parse_score,
                                "summary":     py_rep.summary,
                                "warnings":    py_rep.warnings,
                            })()
                            banner_success(
                                f"**{report.strategy_name}** loaded into Strategy Loader! "
                                "Go to Section 4 to run the backtest."
                            )
                        else:
                            banner_warning("Loaded with warnings — check Strategy Loader parse report.")
                    except Exception as exc:
                        banner_error(f"Load failed: {exc}")

            with dl3:
                # Export full report as text
                report_text = "\n".join([
                    f"REMODEL REPORT — {report.strategy_name}",
                    f"Source: {report.source_file}  |  Language: {report.source_language.upper()}",
                    f"Lines: {report.source_lines:,}  |  Score: {report.parse_score:.0f}%",
                    f"LLM Assist: {'Yes' if report.used_llm_assist else 'No'}",
                    "",
                    "=== TRANSFORMATIONS ===",
                ] + report.transformations + [
                    "",
                    "=== WARNINGS ===",
                ] + report.warnings + [
                    "",
                    "=== ERRORS ===",
                ] + report.errors)
                st.download_button(
                    "📋  Download Parse Report",
                    data=report_text.encode(),
                    file_name=f"{report.strategy_name}_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with out_tab2:
            raw_src = SS.get("remodeler_source", "")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div style="font-family:monospace;font-size:0.74rem;'
                    f'color:{_R};margin-bottom:4px;">ORIGINAL SOURCE</div>',
                    unsafe_allow_html=True,
                )
                st.code(raw_src[:3000] + ("\n…[truncated]" if len(raw_src) > 3000 else ""),
                        language="python" if report.source_language == "python" else "cpp")
            with c2:
                st.markdown(
                    f'<div style="font-family:monospace;font-size:0.74rem;'
                    f'color:{_G};margin-bottom:4px;">REMODELED OUTPUT</div>',
                    unsafe_allow_html=True,
                )
                st.code(report.remodeled_code[:3000] +
                        ("\n…[truncated]" if len(report.remodeled_code) > 3000 else ""),
                        language="python")

    # ── Example error types handled ─────────────────────────────────────
    st.markdown("---")
    section_header("WHAT THE REMODELER FIXES")
    st.markdown(
        f"""<div style="font-family:monospace;font-size:0.76rem;
        background:{_C};border:1px solid #2A2A2A;border-radius:8px;padding:14px;">
<span style="color:{_R};">BEFORE (broken):</span>
<span style="color:#888;">  ❌ No Strategy class found</span>
<span style="color:#888;">  ❌ Missing on_bar() method</span>
<span style="color:#888;">  ❌ Missing stop_distance()</span>
<span style="color:#888;">  ❌ C++ class with no Python equivalent</span>
<span style="color:#888;">  ❌ Bare script-style if/else entry logic</span>

<span style="color:{_G};">AFTER (compliant):</span>
<span style="color:{_G};">  ✅ class Strategy(PyStrategyBase):</span>
<span style="color:{_G};">  ✅ on_bar(self, bar, indicators) → signal</span>
<span style="color:{_G};">  ✅ stop_distance() → ATR × extracted_mult</span>
<span style="color:{_G};">  ✅ target_distance() → extracted_R × stop</span>
<span style="color:{_G};">  ✅ All constants preserved as class attributes</span>
<span style="color:{_G};">  ✅ Ready for backtest engine immediately</span>
</div>""",
        unsafe_allow_html=True,
    )
