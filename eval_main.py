from evaluation.pipeline import run_retrieval_pipeline


def main() -> None:
    result = run_retrieval_pipeline(
        dataset_path="evaluation/dataset/eval_dataset.json",
    )
    print(result)


if __name__ == "__main__":
    main()
