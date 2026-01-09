from datasets import load_dataset
import csv
import sys

def extract_lines(n: int, output_path: str = "input.csv"):
    ds = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        samples.append((i + 1, item["text"].replace("\n", " ")))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        writer.writerows(samples)

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_dataset.py <num_lines> [input.csv]")
        sys.exit(1)

    n = int(sys.argv[1])
    input_path = sys.argv[2] if len(sys.argv) > 2 else "input.csv"
    extract_lines(n, input_path)
    print(f"Saved {n} lines to {input_path}")

if __name__ == "__main__":
    main()
