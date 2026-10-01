# SmartHire — Resume-to-Job Matching & Career Guidance Engine

SmartHire is a Streamlit application that extracts text from an uploaded PDF resume, predicts a resume dataset category, ranks prepared job descriptions by TF-IDF cosine similarity, reports technical skills found or missing in recommended jobs, and summarizes exploratory job clusters. It does not predict hiring decisions.

## Main Components

- **Resume classification:** TF-IDF and Logistic Regression predict one of the source resume dataset's categories.
- **Job recommendations:** a saved TF-IDF index ranks job descriptions by cosine similarity. Similarity is a text-ranking score, not a probability.
- **Skill-gap analysis:** a controlled vocabulary identifies recognized technical skills in the resume and recommendations.
- **Job clustering:** saved K-Means assignments provide exploratory group counts and common source labels.

The classifier category does not filter or change the similarity ranking. Skill gaps also do not affect ranking. Clusters are exploratory and have no intrinsic career meaning.

## Technology

Python, Streamlit, Pandas, NumPy, SciPy, scikit-learn, joblib, pypdf, pytest, and Jupyter with Matplotlib/Seaborn for notebook work.

## Project Structure

```text
app/                 Streamlit application
src/                 Analysis, data helpers, models, and offline artifact builders
tests/               Fixture-based automated tests
notebooks/           EDA and model experiments (not needed by the app)
data/DATASETS.md     Dataset roles, provenance, and preparation status
data/raw/            Local source datasets; excluded from Git
data/processed/      Local prepared datasets; excluded from Git
models/              Small model artifacts and locally generated job index
requirements.txt
```

## Install

From the repository root in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Large Runtime Artifacts

A normal Git clone does **not** contain the large generated job-index files. The app requires all six artifacts below. No external artifact host or download mechanism is configured in this project.

| Artifact | Git status | How to provide it |
| --- | --- | --- |
| `models/resume_classifier.pkl` | Small artifact; eligible for intentional Git tracking, but generated locally by default | `python -m src.train_classifier` |
| `models/job_vectorizer.pkl` | Small artifact; eligible for intentional Git tracking, but generated locally by default | `python -m src.build_job_index` |
| `models/job_tfidf.npz` | Large; intentionally ignored | `python -m src.build_job_index` |
| `models/job_metadata.csv` | Large; intentionally ignored | `python -m src.build_job_index` |
| `models/job_kmeans.pkl` | Small artifact; eligible for intentional Git tracking, but generated locally by default | `python -m src.build_job_clusters` |
| `models/job_clusters.csv` | Small artifact; eligible for intentional Git tracking, but generated locally by default | `python -m src.build_job_clusters` |

The offline commands expect these prepared inputs, which are also excluded from Git:

- `data/processed/resumes_clean.csv` with `Category` and `text` columns.
- `data/processed/jobs_clean.csv` with `title`, `company`, `location`, `skills`, `description`, `experience`, `source`, and `text` columns.

With those prepared files available, run the following from the repository root. These commands train/build artifacts locally; they do not run when Streamlit starts. Job-index construction processes the large job corpus and needs sufficient disk space and memory.

```powershell
python -m src.train_classifier
python -m src.build_job_index
python -m src.build_job_clusters
```

This repository contains no implemented raw-to-prepared dataset builder. The source datasets and prepared CSVs must be obtained or prepared separately; see [data/DATASETS.md](data/DATASETS.md). In particular, downloading the raw Naukri/LinkedIn sources alone is not enough to reproduce `jobs_clean.csv` using the current scripts. No URLs or credentials for external artifact storage are configured.

## Clean Checkout

- **Source-code checkout:** contains code, tests, notebooks, documentation, and dependency declarations. It intentionally excludes raw/prepared datasets and the large job index, so it cannot run resume analysis by itself.
- **Training/build environment:** has the prepared CSVs above and runs the three offline commands to create all required model/index artifacts. The raw-to-prepared transformation is not currently implemented in this repository.
- **Deployment/runtime environment:** has the six runtime artifacts under `models/` in addition to the source checkout and installed requirements. Provide the large matrix and metadata through deployment storage or by building them in a suitable environment, then place them at the configured paths. No automatic download or storage integration exists.

If a required artifact is missing, the application reports its path and the relevant offline build command. It never silently trains or rebuilds data at startup.

## Run the App

After installing dependencies and providing all six runtime artifacts:

```powershell
python -m streamlit run app/streamlit_app.py
```

## Tests and Compilation

```powershell
python -m pytest -q
python -m compileall src app
```

The fixture-based tests use small temporary data and do not need the full corpus or saved production artifacts.

## Data and Limitations

The resume dataset contains 962 resumes across 25 categories. Existing recorded holdout scores are high, but the dataset is duplicate-heavy and those scores do not establish real-world generalization. Job descriptions can contain noise and company boilerplate. Skill extraction covers only a controlled technical vocabulary. Cosine similarity is not hiring probability. Clustering has low silhouette and imbalanced groups, so it is exploratory. The application uses locally prepared public datasets; it does not scrape job sites live and does not use generative AI.

Dataset source names, roles, local preparation status, and download guidance are documented in [data/DATASETS.md](data/DATASETS.md). Review each source's terms before redistributing its files.
