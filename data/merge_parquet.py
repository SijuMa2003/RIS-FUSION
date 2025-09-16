import os
import pandas as pd


def merge_parquet_parts(part_files, output_file: str):
    """
    Merge multiple parquet part files into a single parquet file.

    Args:
        part_files (list of str): List of parquet file paths to merge.
        output_file (str): Path to save the merged parquet file.
    """
    dataframes = []
    for file in part_files:
        if os.path.exists(file):
            print(f"[INFO] Reading: {file}")
            df = pd.read_parquet(file)
            dataframes.append(df)
        else:
            print(f"[WARNING] File not found: {file}")

    if not dataframes:
        print("[ERROR] No files to merge.")
        return

    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"[INFO] Total merged rows: {len(merged_df)}")

    merged_df.to_parquet(output_file, index=False)
    print(f"[INFO] Merged parquet saved to: {output_file}")


# Example usage
if __name__ == "__main__":
    merge_parquet_parts(
        [
            "RIS-Fusion/data/mm_ris_train_part2.parquet",
            "RIS-Fusion/data/mm_ris_train_part1.parquet",
        ],
        "RIS-Fusion/data/mm_ris_train.parquet",
    )