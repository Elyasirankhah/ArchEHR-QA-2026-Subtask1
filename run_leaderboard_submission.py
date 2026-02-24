
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import pipeline_clinical_reasoning_dual_api


def load_cases_from_xml(xml_path: Path, case_ids: list = None):
    """Load cases from ArchEHR-QA XML. Optionally filter by case_ids (e.g. 1-20 or 21-120)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases = []
    for case_elem in root.findall("case"):
        cid = case_elem.get("id", "") or case_elem.get("case_id", "")
        if case_ids is not None and cid not in {str(i) for i in case_ids}:
            continue
        pq_elem = case_elem.find("patient_question")
        patient_question = ""
        if pq_elem is not None:
            phrases = [ph.text.strip() for ph in (pq_elem.findall("phrase") or []) if ph is not None and ph.text]
            patient_question = " ".join(phrases).strip()
        if not patient_question and case_elem.find("patient_narrative") is not None and case_elem.find("patient_narrative").text:
            patient_question = case_elem.find("patient_narrative").text.strip()
        note_sentences = []
        for sent in case_elem.findall("note_excerpt_sentences/sentence") or []:
            if sent is not None and sent.text:
                note_sentences.append(sent.text.strip())
        note_excerpt = " ".join(note_sentences) if note_sentences else None
        if not note_excerpt and case_elem.find("note_excerpt") is not None and case_elem.find("note_excerpt").text:
            note_excerpt = case_elem.find("note_excerpt").text.strip()
        if patient_question:
            cases.append({"case_id": cid, "patient_question": patient_question, "note_excerpt": note_excerpt})
    return cases


def run_range(cases, out_path: Path, desc: str):
    """Run pipeline on cases and write submission JSON."""
    results = []
    for i, c in enumerate(cases, 1):
        try:
            pred = pipeline_clinical_reasoning_dual_api(c["patient_question"], c.get("note_excerpt"))
            results.append({"case_id": c["case_id"], "prediction": pred or ""})
        except Exception as e:
            print(f"  Error case {c['case_id']}: {e}")
            results.append({"case_id": c["case_id"], "prediction": ""})
        if i % 10 == 0:
            print(f"  {desc}: {i}/{len(cases)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {len(results)} predictions to {out_path}")
    return results


def main():
    xml_path = None
    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
    for candidate in [Path("data/archehr-qa.xml"), ROOT / "data" / "archehr-qa.xml", ROOT / "archehr-qa.xml", Path("archehr-qa.xml")]:
        if xml_path is None and candidate.exists():
            xml_path = candidate
            break
    if xml_path is None or not xml_path.exists():
        print("Usage: python run_leaderboard_submission.py [path_to_archehr-qa.xml]")
        print("No XML file found. Place archehr-qa.xml in this directory or pass path.")
        sys.exit(1)
    print(f"Loading cases from {xml_path}")
    cases_1_20 = load_cases_from_xml(xml_path, list(range(1, 21)))
    cases_21_120 = load_cases_from_xml(xml_path, list(range(21, 121)))
    print(f"Cases 1-20: {len(cases_1_20)}; Cases 21-120: {len(cases_21_120)}")
    if not cases_1_20 and not cases_21_120:
        print("No cases found. Check XML has <case id='1'> ... <case id='120'> (or similar).")
        sys.exit(1)
    if cases_1_20:
        run_range(cases_1_20, ROOT / "submission_clinical_reasoning_1-20.json", "1-20")
    if cases_21_120:
        run_range(cases_21_120, ROOT / "submission_clinical_reasoning_21-120.json", "21-120")
    print("Done. Next: python build_codabench_subtask1_submission.py")


if __name__ == "__main__":
    main()

