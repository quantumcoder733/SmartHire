"""Streamlit presentation layer for the artifact-backed SmartHire pipeline."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hashlib
from html import escape
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import streamlit as st

from src.analyzer import (
    analyze_resume_unsupervised,
    load_classifier,
    load_job_cluster_assignments,
    load_job_kmeans,
    load_job_metadata,
    load_job_tfidf_matrix,
    load_job_vectorizer,
)


PAGE_TITLE = "SmartHire — Resume-to-Job Matching & Career Guidance Engine"
ANALYSIS_STATE_KEYS = (
    "analysis_result",
    "artifact_summary",
    "analysis_error",
)

st.set_page_config(
    page_title="SmartHire",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --sh-ink: #172f2a;
        --sh-muted: #40534d;
        --sh-line: #cbd9d2;
        --sh-surface: #ffffff;
        --sh-page: #f5f8f6;
        --sh-accent: #178b4c;
        --sh-expander: #27473f;
    }
    .stApp { background: var(--sh-page); color: var(--sh-ink); }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #172f2a !important;
    }
    .stApp [data-testid="stMarkdownContainer"] p {
        color: #243a34 !important;
        opacity: 1;
    }
    .stApp [data-testid="stCaptionContainer"] p {
        color: #40534d !important;
        opacity: 1;
    }
    .stApp label { color: #263d36 !important; }
    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .sh-eyebrow {
        color: var(--sh-accent);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .sh-supporting {
        color: #40534d !important;
        max-width: 760px;
        line-height: 1.55;
    }
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #cbd9d2;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        min-height: 6.5rem;
        box-sizing: border-box;
        color: #172f2a !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p {
        color: #40534d !important;
        opacity: 1 !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
        color: #172f2a !important;
        opacity: 1 !important;
        font-weight: 700;
    }
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #cbd9d2;
        border-radius: 8px;
        overflow: hidden;
        color: #243a34 !important;
    }
    [data-testid="stExpander"] details > summary {
        background: #27473f !important;
        color: #ffffff !important;
    }
    [data-testid="stExpander"] details > summary * {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
        color: #243a34 !important;
        opacity: 1 !important;
    }
    [data-testid="stExpander"] [data-testid="stMetricLabel"],
    [data-testid="stExpander"] [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p {
        color: #40534d !important;
    }
    [data-testid="stExpander"] [data-testid="stMetricValue"],
    [data-testid="stExpander"] [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {
        color: #172f2a !important;
    }
    [data-testid="stExpander"] details > summary [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border-color: #aabbb2 !important;
        color: #243a34 !important;
    }
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] button {
        color: #263d36 !important;
        opacity: 1 !important;
    }
    /* Analyze Resume button */
    [data-testid="stBaseButton-primary"] {
        background-color: #176B5B !important;
        border-color: #176B5B !important;
        color: #FFFFFF !important;
        min-height: 2.75rem;
        font-weight: 650;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stBaseButton-primary"] *,
    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span,
    [data-testid="stBaseButton-primary"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stBaseButton-primary"]:hover {
        background-color: #13594c !important;
        border-color: #13594c !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
        .summary-category-card {
        background: #ffffff;
        border: 1px solid #cbd9d2;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        min-height: 6.5rem;
        box-sizing: border-box;
        color: #172f2a;
    }
    .summary-category-label {
        color: #40534d;
        font-size: 0.875rem;
        font-weight: 600;
        line-height: 1.25;
        margin-bottom: 0.45rem;
    }
    .summary-category-value {
        color: #172f2a;
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.25;
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
        word-break: normal;
        overflow-wrap: anywhere;
    }
    .skill-chip-group { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .skill-chip {
        display: inline-block;
        border: 1px solid #aebfb6;
        border-radius: 999px;
        padding: 0.25rem 0.62rem;
        font-size: 0.82rem;
        line-height: 1.35;
        background: #f5f8f6 !important;
        color: #172f2a !important;
    }
    .skill-chip.have { border-color: #8eb6a7; background: #e5f2ec !important; }
    .skill-chip.develop { border-color: #a9b99a; background: #f0f4e8 !important; }
    .sh-footer {
        background: #0b0f0e !important;
        border-top: 1px solid #0b0f0e;
        border-radius: 8px;
        color: #ffffff !important;
        font-size: 0.82rem;
        line-height: 1.5;
        padding: 1rem 1.2rem;
        margin-top: 2rem;
    }

    .sh-footer p {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading saved SmartHire artifacts...")
def _load_persisted_artifact_summary() -> dict[str, int]:
    """Validate and warm immutable artifacts once per Streamlit process."""
    load_classifier()
    load_job_vectorizer()
    job_matrix = load_job_tfidf_matrix()
    metadata = load_job_metadata()
    assignments = load_job_cluster_assignments(expected_rows=len(metadata))
    load_job_kmeans()

    if job_matrix.shape[0] != len(metadata):
        raise ValueError(
            "Job metadata and TF-IDF matrix must contain the same number of rows."
        )

    return {
        "job_count": int(job_matrix.shape[0]),
        "cluster_count": len(set(assignments.tolist())),
    }


def _resume_signature(uploaded_resume: Any | None) -> str | None:
    """Identify the uploaded content without storing it in shared cache."""
    if uploaded_resume is None:
        return None
    return hashlib.sha256(uploaded_resume.getbuffer()).hexdigest()


def _reset_analysis_for_changed_upload(signature: str | None) -> None:
    """Clear session-local results when the selected resume changes."""
    if "resume_signature" not in st.session_state:
        st.session_state["resume_signature"] = signature
        return
    if st.session_state["resume_signature"] != signature:
        for key in ANALYSIS_STATE_KEYS:
            st.session_state.pop(key, None)
        st.session_state["resume_signature"] = signature


def _display_skill_tags(
    label: str,
    skills: list[str],
    style: str,
    empty_message: str,
) -> None:
    """Render canonical skill values as compact, escaped tags."""
    st.markdown(f"**{label}**")
    if not skills:
        st.caption(empty_message)
        return
    tags = "".join(
        f'<span class="skill-chip {style}">{escape(skill)}</span>'
        for skill in skills
    )
    st.markdown(f'<div class="skill-chip-group">{tags}</div>', unsafe_allow_html=True)


def _display_summary(
    analysis: dict[str, Any],
    artifact_summary: dict[str, int],
) -> None:
    """Render classifier and corpus summary metrics."""
    resume = analysis["resume"]
    st.subheader("Candidate Summary")
    summary_columns = st.columns(4)
    summary_columns[0].markdown(
        "<div class='summary-category-card'>"
        "<div class='summary-category-label'>Predicted Category</div>"
        f"<div class='summary-category-value'>{escape(str(analysis['predicted_category']))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    summary_columns[1].metric("Job Matches", len(analysis["recommendations"]))
    summary_columns[2].metric("Jobs in Index", f"{artifact_summary['job_count']:,}")
    summary_columns[3].metric("Job Clusters", artifact_summary["cluster_count"])
    st.caption(
        f"Resume: {resume['filename']} · {resume['text_length']:,} characters"
    )


def _display_recommendations(recommendations: list[dict[str, Any]]) -> None:
    """Render ranked job results without changing their scores or order."""
    st.subheader("Recommended Jobs")
    if not recommendations:
        st.info("No job recommendations were returned for this resume.")
        return

    for recommendation in recommendations:
        with st.expander(
            f"{recommendation['rank']:02d}  ·  {recommendation['title']}",
            expanded=recommendation["rank"] == 1,
        ):
            details = st.columns(3)
            details[0].markdown(f"**Company**\n\n{recommendation['company']}")
            details[1].markdown(f"**Location**\n\n{recommendation['location']}")
            details[2].markdown(f"**Experience**\n\n{recommendation['experience']}")
            metrics = st.columns(2)
            metrics[0].metric(
                "Similarity",
                f"{recommendation['similarity_score']:.3f}",
            )
            metrics[1].metric("Cluster ID", recommendation["cluster"])


def _display_skill_gap(recommendations: list[dict[str, Any]]) -> None:
    """Show existing per-job matched and missing skills."""
    st.subheader("Technical Skill Gap")
    st.caption(
        "Recognized technical skills in the resume are compared with skills "
        "identified in each recommended job's text."
    )
    recognized_anywhere = any(
        recommendation["matched_skills"] or recommendation["missing_skills"]
        for recommendation in recommendations
    )
    if not recognized_anywhere:
        st.info("No recognized technical skills found in this resume.")
        return

    for recommendation in recommendations:
        with st.container(border=True):
            st.markdown(
                f"**Rank {recommendation['rank']} · {recommendation['title']}**"
            )
            skill_columns = st.columns(2)
            with skill_columns[0]:
                _display_skill_tags(
                    "Skills you have",
                    recommendation["matched_skills"],
                    "have",
                    "No recognized matches for this job.",
                )
            with skill_columns[1]:
                _display_skill_tags(
                    "Skills to develop",
                    recommendation["missing_skills"],
                    "develop",
                    "No missing recognized technical skills",
                )


def _display_job_market(clusters: list[dict[str, Any]]) -> None:
    """Render persisted cluster counts and broad source labels."""
    st.subheader("Job Market Exploration")
    st.caption(
        "Clustering is exploratory and groups jobs by job-text similarity. "
        "Cluster IDs are arbitrary labels and do not represent fixed career categories."
    )
    st.caption(
        "Common dataset labels are broad source categories, not extracted technical skills."
    )
    if not clusters:
        st.info("No cluster information is available.")
        return

    chart_data = pd.DataFrame(
    {"Jobs": [cluster["job_count"] for cluster in clusters]},
    index=[f"Cluster {cluster['cluster_id']}" for cluster in clusters],
)

    for start in range(0, len(clusters), 2):
        cluster_columns = st.columns(2)
        for column, cluster in zip(cluster_columns, clusters[start : start + 2]):
            with column:
                with st.container(border=True):
                    st.markdown(f"**Cluster ID {cluster['cluster_id']}**")
                    st.metric("Job count", f"{cluster['job_count']:,}")
                    st.markdown("**Common dataset labels**")
                    if cluster["top_skills"]:
                        for label in cluster["top_skills"]:
                            st.write(
                                f"{label['skill']} · {label['job_count']:,} jobs"
                            )
                    else:
                        st.caption("No dataset labels available")

    # Cluster chart comes AFTER the cluster information
    st.markdown("### Job Distribution by Cluster")
    st.bar_chart(chart_data)

def _display_analysis(
    analysis: dict[str, Any],
    artifact_summary: dict[str, int],
) -> None:
    """Render the current analyzer result using Streamlit components."""
    recommendations = analysis["recommendations"]
    _display_summary(analysis, artifact_summary)
    st.divider()
    _display_recommendations(recommendations)
    st.info(
        "SmartHire represents resume and job text using TF-IDF and ranks jobs "
        "using cosine similarity. The score measures textual similarity; it is "
        "not a probability of getting hired and is not model accuracy."
    )
    st.divider()
    _display_skill_gap(recommendations)
    st.divider()
    _display_job_market(analysis["clusters"])



st.title("SmartHire")
st.markdown("### Resume-to-Job Matching & Career Guidance")
st.markdown(
    "<div class='sh-supporting'>Upload a resume and SmartHire analyzes the "
    "candidate profile, identifies suitable roles from the indexed job dataset, "
    "highlights technical skill gaps, and provides exploratory job-market groupings.</div>",
    unsafe_allow_html=True,
)
st.divider()

with st.container(border=True):
    st.subheader("Upload your resume")
    st.caption("Supported format: PDF. Analysis starts only when you select Analyze Resume.")
    uploaded_resume = st.file_uploader(
        "Resume PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_resume is not None:
        st.caption(f"Selected file: {uploaded_resume.name}")
    analyze_clicked = st.button(
        "Analyze Resume",
        type="primary",
        use_container_width=True,
    )

resume_signature = _resume_signature(uploaded_resume)
_reset_analysis_for_changed_upload(resume_signature)

if analyze_clicked:
    if uploaded_resume is None:
        st.session_state["analysis_error"] = "Please upload a PDF resume to continue."
    elif Path(uploaded_resume.name).suffix.lower() != ".pdf":
        st.session_state["analysis_error"] = "This file is not a PDF. Please upload a PDF resume."
    else:
        temporary_path: Path | None = None
        st.session_state.pop("analysis_error", None)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as file:
                file.write(uploaded_resume.getbuffer())
                temporary_path = Path(file.name)

            with st.spinner("Analyzing your resume..."):
                artifact_summary = _load_persisted_artifact_summary()
                analysis = analyze_resume_unsupervised(
                    temporary_path,
                    top_n=5,
                )

            st.session_state["analysis_result"] = analysis
            st.session_state["artifact_summary"] = artifact_summary
        except FileNotFoundError as error:
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("artifact_summary", None)
            st.session_state["analysis_error"] = str(error)
        except ValueError:
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("artifact_summary", None)
            st.session_state["analysis_error"] = (
                "We couldn't analyze this PDF. Check that it contains readable "
                "resume text and try again."
            )
        except Exception:
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("artifact_summary", None)
            st.session_state["analysis_error"] = (
                "We couldn't read this PDF or complete the analysis. "
                "Please try another PDF."
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

if st.session_state.get("analysis_error"):
    st.error(st.session_state["analysis_error"])

analysis_result = st.session_state.get("analysis_result")
artifact_summary = st.session_state.get("artifact_summary")
if analysis_result is not None and artifact_summary is not None:
    st.success("Resume analysis complete.")
    _display_analysis(analysis_result, artifact_summary)

st.markdown(
    "<div class='sh-footer'>SmartHire uses classical machine-learning methods "
    "for resume classification, job-text similarity, skill-gap analysis, and "
    "exploratory clustering. Results are based on available public datasets "
    "and are career guidance, not a hiring decision.</div>",
    unsafe_allow_html=True,
)
