# src/utils/audit.py

import random
import re
import json
import pandas as pd
from pathlib import Path


# -------------------------------------------
# 1️⃣ Generate Statistics Summary
# -------------------------------------------
def generate_chunk_stats(chunks, output_dir="audit"):

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if not chunks:
        print("No chunks found.")
        return None

    data = [c["metadata"] for c in chunks]
    df = pd.DataFrame(data)

    metrics = ["word_count", "token_count"]

    if "char_count" in df.columns:
        metrics.append("char_count")

    stats_df = df[metrics].agg(
        ["count", "mean", "std", "min", "max"]
    ).transpose()

    stats_df.columns = ["Count", "Mean", "Std Dev", "Min", "Max"]
    stats_df = stats_df.round(2)

    # Save CSV
    stats_df.to_csv(output_path / "chunk_stats_summary.csv")

    print("\nCHUNK DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(stats_df.to_string())
    print("=" * 60)

    return stats_df


# -------------------------------------------
# 2️⃣ Sample Random Chunks
# -------------------------------------------
def sample_chunks_for_audit(
    chunks,
    sample_size=5,
    output_dir="audit"
):

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    total = len(chunks)
    samples = random.sample(chunks, min(sample_size, total))

    audit_records = []

    for idx, chunk in enumerate(samples, 1):

        metadata = chunk["metadata"]

        clean_text = re.sub(
            r"<span.*?>.*?</span>",
            "",
            chunk["text"],
            flags=re.DOTALL
        ).strip()

        record = {
            "sample_id": idx,
            "source": metadata.get("source", "N/A"),
            "page": metadata.get("page", "N/A"),
            "word_count": metadata.get("word_count"),
            "token_count": metadata.get("token_count"),
            "section": metadata.get("section", "N/A"),
            "subsection": metadata.get("subsection", "N/A"),
            "text_preview": clean_text[:1000],  # limit size
            "latex_delimiters": chunk["text"].count("$"),
        }

        audit_records.append(record)

    # Save JSON
    with open(output_path / "audit_samples.json", "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)

    # Save readable TXT
    with open(output_path / "audit_samples.txt", "w", encoding="utf-8") as f:
        for record in audit_records:
            f.write("=" * 80 + "\n")
            for k, v in record.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"\nSaved {len(audit_records)} audit samples to '{output_dir}/'")

    return audit_records
