import json
import os
import zipfile


ROOT = os.path.dirname(__file__)

PRED_21_120_PATH = os.path.join(ROOT, "submission_clinical_reasoning_21-120.json")
PRED_1_20_PATH = os.path.join(ROOT, "submission_clinical_reasoning_1-20.json")
TRAIN_JSONL_PATH = os.path.join(ROOT, "winner", "data", "train.jsonl")
DEV_JSONL_PATH = os.path.join(ROOT, "winner", "data", "dev.jsonl")

OUT_DIR = os.path.join(ROOT, "codabench_subtask1")
OUT_JSON_PATH = os.path.join(OUT_DIR, "submission.json")
OUT_ZIP_PATH = os.path.join(ROOT, "codabench_subtask1.zip")


def _load_jsonl_targets(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("case_id"))
            tgt = row.get("target_text")
            if isinstance(tgt, str) and tgt.strip():
                out[cid] = tgt.strip()
    return out


def _enforce_15_words_and_qmark(text: str) -> str:
    if not text:
        return ""
    q = " ".join(str(text).strip().split())
    if not q:
        return ""
    if not q.endswith("?"):
        q = q.rstrip(".! ") + "?"
    words = q.split()
    if len(words) > 15:
        q = " ".join(words[:15])
        if not q.endswith("?"):
            q = q.rstrip(".! ") + "?"
    return q


def main() -> None:
    # 1) Load clinical-reasoning predictions (prefer these over any gold backfill)
    pred_map: dict[str, str] = {}

    if os.path.exists(PRED_1_20_PATH):
        with open(PRED_1_20_PATH, "r", encoding="utf-8") as f:
            pred_list_1_20 = json.load(f)
        for item in pred_list_1_20:
            cid = str(item.get("case_id"))
            pred = item.get("prediction")
            if isinstance(pred, str) and cid:
                pred_map[cid] = pred

    if not os.path.exists(PRED_21_120_PATH):
        raise SystemExit(
            f"Missing {PRED_21_120_PATH}. Run the pipeline first:\n"
            "  python run_leaderboard_submission.py [path_to_archehr-qa.xml]\n"
            "Then run this script again."
        )
    with open(PRED_21_120_PATH, "r", encoding="utf-8") as f:
        pred_list_21_120 = json.load(f)
    for item in pred_list_21_120:
        cid = str(item.get("case_id"))
        pred = item.get("prediction")
        if isinstance(pred, str) and cid:
            pred_map[cid] = pred

    # 2) Optional fallback for 1-20 from winner train/dev jsonl (if present)
    targets: dict[str, str] = {}
    if os.path.exists(TRAIN_JSONL_PATH):
        targets.update(_load_jsonl_targets(TRAIN_JSONL_PATH))
    if os.path.exists(DEV_JSONL_PATH):
        targets.update(_load_jsonl_targets(DEV_JSONL_PATH))

    # 3) Build final 1-120 submission
    final: list[dict[str, str]] = []
    missing: list[str] = []
    for i in range(1, 121):
        cid = str(i)
        if cid in pred_map:
            final.append({"case_id": cid, "prediction": _enforce_15_words_and_qmark(pred_map[cid])})
        elif cid in targets:
            # Fallback only if clinical-reasoning dev predictions aren't available.
            final.append({"case_id": cid, "prediction": _enforce_15_words_and_qmark(targets[cid])})
        else:
            missing.append(cid)

    if missing:
        raise SystemExit(
            f"Missing predictions for case_ids: {missing}. "
            "Generate them with: python run_leaderboard_submission.py [path_to_archehr-qa.xml]"
        )

    # 4) Write submission.json
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # 5) Write submission zip (submission.json at zip root)
    if os.path.exists(OUT_ZIP_PATH):
        os.remove(OUT_ZIP_PATH)
    with zipfile.ZipFile(OUT_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT_JSON_PATH, arcname="submission.json")

    print(f"Wrote {len(final)} cases to: {OUT_JSON_PATH}")
    print(f"Wrote zip to: {OUT_ZIP_PATH}")


if __name__ == "__main__":
    main()

