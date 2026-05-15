import json
import sys
from pathlib import Path
from typing import Any

from evaluation.pipeline import run_retrieval_pipeline, run_generation_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def summarize_generation_result(result: dict[str, Any]) -> dict[str, Any]:
    """压缩 generation evaluation 输出，只保留核心指标。"""
    return {
        "total": result["total"],
        "mean_claim_faithfulness": result["mean_claim_faithfulness"],
        "mean_answer_relevance": result["mean_answer_relevance"],
        "items": [
            {
                "id": item["id"],
                "question": item["question"],
                "claim_faithfulness": item["claim_faithfulness"]["claim_faithfulness"],
                "answer_relevance": item["answer_relevance"]["score"],
            }
            for item in result["items"]
        ],
    }


def main() -> None:
    retrieval_result = run_retrieval_pipeline(
        dataset_path="evaluation/dataset/test.json",
    )
    print(json.dumps(retrieval_result, ensure_ascii=False, indent=2))

    generation_result = run_generation_pipeline(
        dataset_path="evaluation/dataset/test.json",
    )
    summary = summarize_generation_result(generation_result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
