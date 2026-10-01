"""Controlled-vocabulary technical skill extraction and gap analysis."""

from collections.abc import Collection, Mapping
import re
from typing import Any


# Add aliases under their canonical display name as the job corpus evolves.
TECHNICAL_SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": (),
    "Java": (),
    "JavaScript": (),
    "TypeScript": (),
    "C": (),
    "C++": (),
    "C#": (),
    "SQL": (),
    "HTML": (),
    "CSS": (),
    "React": (),
    "Angular": (),
    "Node.js": (r"node(?:\s*\.\s*|\s+)js", r"nodejs"),
    "Django": (),
    "Flask": (),
    "Spring": (),
    "Spring Boot": (r"spring\s*boot",),
    "Hibernate": (),
    "PHP": (),
    "Android": (),
    "Kotlin": (),
    "Swift": (),
    "REST API": (r"restful\s+api",),
    "GraphQL": (),
    "AWS": (),
    "Azure": (),
    "Docker": (),
    "Kubernetes": (),
    "Git": (),
    "Linux": (),
    "Machine Learning": (),
    "Deep Learning": (),
    "TensorFlow": (),
    "PyTorch": (),
    "Pandas": (),
    "NumPy": (),
    "scikit-learn": (r"scikit[\s-]+learn", r"sklearn"),
    "Power BI": (),
    "Tableau": (),
    "Excel": (),
    "MongoDB": (),
    "MySQL": (),
    "PostgreSQL": (),
    "Redis": (),
    "Hadoop": (),
    "Spark": (),
    "Kafka": (),
    "Big Data": (),
    "Data Analysis": (),
    "Data Science": (),
    "NLP": (),
    "Computer Vision": (),
    "DevOps": (),
    "Jenkins": (),
}


def _skill_pattern(skill: str, aliases: tuple[str, ...]) -> re.Pattern[str]:
    canonical_pattern = r"\s+".join(
        re.escape(part) for part in skill.split()
    )
    alternatives = [canonical_pattern, *aliases]
    if skill == "C":
        return re.compile(
            r"(?<![A-Za-z0-9_])(?:C)(?![A-Za-z0-9_+#])",
            re.IGNORECASE,
        )
    return re.compile(
        r"(?<![A-Za-z0-9_])(?:"
        + "|".join(f"(?:{alternative})" for alternative in alternatives)
        + r")(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


_SKILL_PATTERNS = tuple(
    (
        canonical,
        _skill_pattern(canonical, aliases),
    )
    for canonical, aliases in TECHNICAL_SKILL_ALIASES.items()
)
_CANONICAL_SKILLS = {skill.casefold(): skill for skill in TECHNICAL_SKILL_ALIASES}


def extract_technical_skills(text: str) -> list[str]:
    """Extract canonical, deduplicated skills from free text."""
    if not isinstance(text, str) or not text:
        return []

    matches: list[tuple[int, int, str]] = []
    for canonical, pattern in _SKILL_PATTERNS:
        matches.extend(
            (match.start(), match.end(), canonical)
            for match in pattern.finditer(text)
        )

    selected: list[tuple[int, int, str]] = []
    for match in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
        start, end, _ = match
        if any(start < selected_end and end > selected_start for selected_start, selected_end, _ in selected):
            continue
        selected.append(match)

    return sorted({skill for _, _, skill in selected}, key=str.casefold)


def extract_job_technical_skills(job: Mapping[str, Any]) -> list[str]:
    """Extract skills from job text and useful metadata, excluding broad labels."""
    text_fields = (
        "title",
        "description",
        "text",
        "company",
        "location",
        "experience",
        "source",
    )
    searchable_text = " ".join(
        str(job[field])
        for field in text_fields
        if job.get(field) is not None
    )
    return extract_technical_skills(searchable_text)


def calculate_skill_gap(
    candidate_skills: Collection[str],
    target_skills: Collection[str],
) -> dict[str, list[str]]:
    """Return canonical technical skills matched by and missing from a candidate.

    The collection-based API is retained. Values outside the controlled
    vocabulary are ignored so broad job-category labels are never reported.
    """
    normalized_candidate_skills = {
        skill
        for value in candidate_skills
        for skill in extract_technical_skills(str(value))
    }
    normalized_target_skills = {
        skill
        for value in target_skills
        for skill in extract_technical_skills(str(value))
    }

    return {
        "matched_skills": sorted(
            normalized_candidate_skills & normalized_target_skills,
            key=str.casefold,
        ),
        "missing_skills": sorted(
            normalized_target_skills - normalized_candidate_skills,
            key=str.casefold,
        ),
    }