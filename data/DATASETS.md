# Dataset and Artifact Guide

Dataset files and generated artifacts are local inputs/outputs. The repository ignores the data directories, so the names below describe the project's expected files and current local layout; they are not included in a normal Git checkout.

## Raw Data

| File | Role in SmartHire |
| --- | --- |
| `data/raw/UpdatedResumeDataSet.csv` | Public resume-category source used to prepare classifier data. Resume text can contain personal information; do not publish copies or notebook outputs without reviewing them. |
| `data/raw/naukri_com-job_sample.csv` | Public Naukri job source, one documented input to the prepared job corpus. |
| `data/raw/linkedin/postings.csv` | Public LinkedIn job-posting source. |
| `data/raw/linkedin/jobs/job_skills.csv` and `data/raw/linkedin/mappings/skills.csv` | LinkedIn job skill codes and their labels. |

The files currently present locally are excluded from Git by `.gitignore`. Source dataset names and historical download guidance are provided for provenance; check each dataset's current license and terms before use or redistribution. SmartHire does not download or scrape these sources at runtime.

## Processed Data

| File | Role | Recreated by this repository? |
| --- | --- | --- |
| `data/processed/resumes_clean.csv` | Prepared resume text and `Category` labels for classifier training. | No raw-to-prepared command is implemented. |
| `data/processed/jobs_clean.csv` | Prepared job rows used to build the searchable index. | No raw-to-prepared command is implemented. |

Both files are excluded from Git. The current preprocessing module normalizes text; it does not transform the raw datasets into these complete prepared CSVs. The existing small `data/processed/jobs.csv` was an unused mock/demo dataset and is not an application input.

## Generated Model and Index Artifacts

The Streamlit app requires all six artifacts. Small artifacts may be intentionally tracked; the large index files are ignored. No external artifact storage is configured.

| File | Purpose | Build command |
| --- | --- | --- |
| `models/resume_classifier.pkl` | Complete resume TF-IDF/classifier pipeline | `python -m src.train_classifier` |
| `models/job_vectorizer.pkl` | Fitted job TF-IDF vectorizer | `python -m src.build_job_index` |
| `models/job_tfidf.npz` | Sparse job TF-IDF matrix (large; ignored) | `python -m src.build_job_index` |
| `models/job_metadata.csv` | Job rows aligned with the matrix (large; ignored) | `python -m src.build_job_index` |
| `models/job_kmeans.pkl` | Fitted K-Means model | `python -m src.build_job_clusters` |
| `models/job_clusters.csv` | Cluster assignments aligned with metadata and matrix | `python -m src.build_job_clusters` |

The commands require the corresponding prepared CSVs described above. Run them offline from the project root; the Streamlit app does not train models or rebuild indexes. A clean checkout alone cannot regenerate the full artifacts until the prepared datasets are separately provided or their preparation pipeline is implemented.
