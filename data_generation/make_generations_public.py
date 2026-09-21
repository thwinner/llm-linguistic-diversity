"""
Builds an ID-only, uploadable mirror of data_generation/generations/ into data_generation/generations_public/.

Strips fields that carry copyright-protected reference/source text
(prompt, human_story, human_stories, source_text, gt_translation,
human_translations), while keeping every ID field (prompt_id, book_id,
paragraph_index, source_lang, model, sample_idx, n_qualifying_translators)
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data_generation" / "generations"
DST = ROOT / "data_generation" / "generations_public"


STORY_DOMAINS = ["storytelling_n200"]
PAR3_DOMAINS = ["translation_ref3", "translation_ref4"]


# Function to strip specified fields from a JSONL file and write the result to a new JSONL file
def strip_jsonl(src_path: Path, dst_path: Path, drop_fields: set[str]) -> int:

    # Create the destination directory if it doesn't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    # Read the source JSONL file, strip the specified fields and write to the destination JSONL file
    with src_path.open("r", encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec = {k: v for k, v in rec.items() if k not in drop_fields}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# Copy all JSON config files from the source domain directory to the destination domain directory
def copy_configs(domain_src: Path, domain_dst: Path) -> None:
    for cfg in domain_src.glob("*.json"):
        shutil.copy2(cfg, domain_dst / cfg.name)


# Main function to create the public mirror of the generations data
def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)

    for domain in STORY_DOMAINS:
        src_dir = SRC / domain
        dst_dir = DST / domain

        # Strip fields from prompts_sample.jsonl and write to prompts_sample_ids.jsonl
        n = strip_jsonl(
            src_dir / "prompts_sample.jsonl",
            dst_dir / "prompts_sample_ids.jsonl",
            drop_fields={"prompt", "human_story", "human_stories"},
        )
        print(f"[{domain}] prompts_sample_ids.jsonl: {n} rows (prompt/human_story/human_stories stripped)")

        # Strip fields from all writingprompts_*_n5.jsonl files and write to the destination directory
        for gen_file in sorted(src_dir.glob("writingprompts_*_n5.jsonl")):
            n = strip_jsonl(gen_file, dst_dir / gen_file.name, drop_fields={"prompt"})

            print(f"[{domain}] {gen_file.name}: {n} rows (prompt stripped)")

        copy_configs(src_dir, dst_dir)

    # Repeat the same process for the PAR3 domains, stripping fields from par3_paragraphs_sample.jsonl and par3_*_n5.jsonl files
    for domain in PAR3_DOMAINS:
        src_dir = SRC / domain
        dst_dir = DST / domain
        if not src_dir.exists():
            continue

        # Strip fields from par3_paragraphs_sample.jsonl and write to par3_paragraphs_sample_ids.jsonl
        n = strip_jsonl(
            src_dir / "par3_paragraphs_sample.jsonl",
            dst_dir / "par3_paragraphs_sample_ids.jsonl",
            drop_fields={"source_text", "gt_translation", "human_translations"},
        )
        print(f"[{domain}] par3_paragraphs_sample_ids.jsonl: {n} rows (source_text/gt_translation/human_translations stripped)")

        # Strip fields from all par3_*_n5.jsonl files and write to the destination directory
        for gen_file in sorted(src_dir.glob("par3_*_n5.jsonl")):
            n = strip_jsonl(
                gen_file,
                dst_dir / gen_file.name,
                drop_fields={"source_text", "gt_translation", "human_translations"},
            )
            print(f"[{domain}] {gen_file.name}: {n} rows (source_text/gt_translation/human_translations stripped)")

        copy_configs(src_dir, dst_dir)

    print(f"\nDone. Public mirror written to {DST}")


if __name__ == "__main__":
    main()
