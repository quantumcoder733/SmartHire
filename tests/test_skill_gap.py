"""Focused tests for controlled-vocabulary skill extraction."""

from src.models.skill_gap import (
    calculate_skill_gap,
    extract_job_technical_skills,
    extract_technical_skills,
)


def test_python_job_example_extracts_matched_and_missing_skills() -> None:
    resume = "Python developer with SQL, Pandas, NumPy and React experience."
    job = {
        "title": "Python Developer",
        "description": (
            "Required Python, SQL, Django and REST API experience."
        ),
        "skills": "Information Technology, Sales",
    }

    gap = calculate_skill_gap(
        extract_technical_skills(resume),
        extract_job_technical_skills(job),
    )

    assert gap == {
        "matched_skills": ["Python", "SQL"],
        "missing_skills": ["Django", "REST API"],
    }


def test_java_spring_boot_example_keeps_canonical_order() -> None:
    resume = "Java developer with Spring Boot and Hibernate."
    job = {
        "title": "Java backend developer",
        "description": (
            "Using Java, Spring Boot, Hibernate and Docker."
        ),
    }

    gap = calculate_skill_gap(
        extract_technical_skills(resume),
        extract_job_technical_skills(job),
    )

    assert gap == {
        "matched_skills": ["Hibernate", "Java", "Spring Boot"],
        "missing_skills": ["Docker"],
    }


def test_aliases_are_case_insensitive_and_return_canonical_names() -> None:
    text = "NODE JS, node.js, sklearn, scikit learn, springboot, RESTful API"

    assert extract_technical_skills(text) == [
        "Node.js",
        "REST API",
        "scikit-learn",
        "Spring Boot",
    ]


def test_programming_language_boundaries_and_broad_labels() -> None:
    assert extract_technical_skills("JavaScript and C, C++, C#") == [
        "C",
        "C#",
        "C++",
        "JavaScript",
    ]
    assert extract_technical_skills(
        "Information Technology, Sales, Management, Engineering"
    ) == []


def test_job_without_recognized_text_skills_returns_empty_gap() -> None:
    job = {
        "title": "Sales Manager",
        "description": "Management and business development experience.",
        "skills": "Information Technology, Engineering",
    }

    assert calculate_skill_gap(
        extract_technical_skills("Experienced sales professional"),
        extract_job_technical_skills(job),
    ) == {"matched_skills": [], "missing_skills": []}