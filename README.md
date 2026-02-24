# ArchEHR-QA 2026 — Subtask 1: Question Interpretation

Pipeline for **ArchEHR-QA 2026** (Archer) **Subtask 1**: convert patient-authored questions into ≤15-word clinician-interpreted questions.

- **Competition:** [CL4Health @ LREC 2026](https://archehr-qa.github.io/) — ArchEHR-QA shared task.
- **Input:** Patient question (free text).
- **Output:** Single clinician-interpreted question (≤15 words).

## Leaderboard results (Codabench)

**Team:** [Data Mining Lab @ Yale](https://github.com/Data-Mining-Lab-Yale)  
**Participant:** [Elyasirankhah](https://github.com/Elyasirankhah)  
**Place:** 5th out of 84 submissions (36 participants).

We made the **official leaderboard**, placing among the top 5 strong systems in the competition.

Results are publicly available at: [Codabench — Subtask 1 leaderboard](https://www.codabench.org/competitions/12865/#/results-tab).

| Metric       | Score |
|-------------|-------|
| **Overall** | **27.1** |
| ROUGELsum   | 28.2  |
| BERTScore   | 40.6  |
| AlignScore  | 19.7  |
| MEDCON (UMLS) | 19.8 |

Submission used the **pipeline** in `pipeline.py` (clinical reasoning, dual-API Claude + GPT-4o, note-driven EHR query type, gold-pattern scoring).

**Reproducing the submission:**  
1. Put ArchEHR-QA XML (e.g. `archehr-qa.xml`) in this directory or pass its path.  
2. `python run_leaderboard_submission.py [path_to_archehr-qa.xml]` → writes `submission_clinical_reasoning_1-20.json` and `submission_clinical_reasoning_21-120.json`.  
3. `python build_codabench_subtask1_submission.py` → writes `codabench_subtask1/submission.json` and `codabench_subtask1.zip`.  
4. Zip and upload to Codabench. Requires `.env` with `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Optional: `gold_patterns.json` (not in repo — dataset terms; generate from dev data if you have access) for tie-breaking; optional `lamar_exemplars.json` for few-shot.

## Setup

1. **Clone and install**
   ```bash
   pip install -r requirements.txt
   ```

2. **API keys**  
   Create a `.env` in this repo root with:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`

## Usage

- **Run pipeline on sample:**  
  `python pipeline.py`

- **Build Codabench submission** (after you have predictions):  
  `python build_codabench_subtask1_submission.py`  
  Output: `codabench_subtask1/submission.json` (zip and upload to Codabench).

## Data

Dev/test data are not included (access via [PhysioNet](https://physionet.org/) per the [ArchEHR-QA 2026](https://archehr-qa.github.io/) instructions). Place your XML/JSON in the paths expected by the scripts or adjust paths in the code. **Gold patterns:** The pipeline can use a `gold_patterns.json` file for candidate tie-breaking; this file is not distributed (it would contain dataset content; PhysioNet DUA does not allow sharing restricted data). If you have lawful access to the dev set, you can build your own patterns; without the file, the pipeline still runs (tie-breaking uses word count and note-based scoring only).

## License

MIT License. See [LICENSE](LICENSE). Dataset terms apply via PhysioNet per ArchEHR-QA 2026.

