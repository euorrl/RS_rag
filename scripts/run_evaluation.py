import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.pipeline import (  # noqa: E402
    run_generation_pipeline,
    run_retrieval_pipeline,
)

DATASET_PATH = Path("evaluation/dataset/eval_dataset.json")
RESULT_PATH = Path("evaluation/results/evaluation_result.json")


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
    """运行检索和生成评估，并将结果覆盖写入 JSON 文件。"""
    retrieval_result = run_retrieval_pipeline(
        dataset_path=DATASET_PATH,
    )

    generation_result = run_generation_pipeline(
        dataset_path=DATASET_PATH,
    )
    result = {
        "dataset_path": str(DATASET_PATH),
        "retrieval": retrieval_result,
        "generation": summarize_generation_result(generation_result),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"评估结果已写入：{RESULT_PATH}")


if __name__ == "__main__":
    main()
