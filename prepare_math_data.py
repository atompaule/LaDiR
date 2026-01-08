import json
import os

from datasets import load_dataset
from tqdm import tqdm


def format_example(example):
    """
    Formats a MATH dataset example into the format expected by ThoughtDataset.
    MATH dataset has: 'problem', 'solution', 'level', 'type'
    ThoughtDataset expects: 'input', 'output', 'solutions' (list)
    """
    return {
        "input": example["problem"],
        "output": example["solution"],
        "solutions": [
            example["solution"]
        ],  # MATH solutions usually contain the reasoning and final answer
        "level": example["level"],
        "type": example["type"],
    }


def main():
    # Create data directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading MATH dataset...")
    # Load Hendrycks MATH dataset
    dataset = load_dataset("qwedsacf/competition_math", trust_remote_code=True)

    # Process train set
    train_data = dataset["train"]

    # Split train data into train/val if test split is missing or empty
    if "test" not in dataset or not dataset["test"]:
        print(
            "No test split found. Splitting training data into train (90%) and val (10%)..."
        )
        split = train_data.train_test_split(test_size=0.1, seed=42)
        train_data = split["train"]
        test_data = split["test"]
    else:
        test_data = dataset["test"]

    train_output_path = os.path.join(output_dir, "train.jsonl")
    vae_train_output_path = os.path.join(output_dir, "vae_train.jsonl")
    print(f"Processing training data ({len(train_data)} examples)...")

    with open(train_output_path, "w", encoding="utf-8") as f, open(
        vae_train_output_path, "w", encoding="utf-8"
    ) as f_vae:
        for example in tqdm(train_data):
            formatted = format_example(example)
            json_line = json.dumps(formatted) + "\n"
            f.write(json_line)
            f_vae.write(json_line)

    print(f"Saved training data to {train_output_path} and {vae_train_output_path}")

    # Process validation set
    val_output_path = os.path.join(output_dir, "val.jsonl")
    vae_val_output_path = os.path.join(output_dir, "vae_val.jsonl")
    print(f"Processing validation data ({len(test_data)} examples)...")

    with open(val_output_path, "w", encoding="utf-8") as f, open(
        vae_val_output_path, "w", encoding="utf-8"
    ) as f_vae:
        for example in tqdm(test_data):
            formatted = format_example(example)
            json_line = json.dumps(formatted) + "\n"
            f.write(json_line)
            f_vae.write(json_line)

    print(f"Saved validation data to {val_output_path} and {vae_val_output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
