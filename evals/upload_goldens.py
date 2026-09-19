import argparse, json
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

GOLDENS_DIR = Path("goldens")

DATASET_MAP = {
    "rag_triad_golden_dataset.json":        "rag-triad",
    "retriever_golden_dataset.json":         "retriever",
    "leakage_golden_dataset.json":           "leakage",
    "application_level_golden_dataset.json": "application-level",
    "scope_golden_dataset.json":             "scope-safety",
    "toxicity_golden_dataset.json":          "toxicity",
    "generator_golden_dataset.json":         "generator",
}

OUTPUT_KEYS = {"expected_action", "expected_answer", "ideal_answer", "success_criteria"}

client = Client()

def _load(path):
    with open(path) as f:
        return json.load(f)

def _get_or_create_dataset(name):
    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        return existing[0]
    return client.create_dataset(name, description=f"RAG pipeline golden dataset: {name}")

def _dataset_count(dataset_id):
    return sum(1 for _ in client.list_examples(dataset_id=dataset_id))

def _split_record(record):
    record_id = record.get("id", "")
    inputs, outputs, metadata = {}, {}, {"golden_id": record_id}
    for k, v in record.items():
        if k == "id":
            continue
        if k in OUTPUT_KEYS:
            outputs[k] = v
        else:
            inputs[k] = v
    return inputs, outputs, metadata

def _clear_dataset(dataset_id):
    example_ids = [e.id for e in client.list_examples(dataset_id=dataset_id)]
    if example_ids:
        client.delete_examples(example_ids)

def upload_dataset(filename, dataset_name, force=False):
    path = GOLDENS_DIR / filename
    if not path.exists():
        print(f"missing: {path} — skipping")
        return
    records = _load(path)
    dataset = _get_or_create_dataset(dataset_name)
    existing_count = _dataset_count(dataset.id)
    if existing_count > 0 and not force:
        print(f"skip {dataset_name} — {existing_count} examples already exist (use --force to overwrite)")
        return
    if force and existing_count > 0:
        _clear_dataset(dataset.id)
        print(f"cleared {existing_count} existing examples from {dataset_name}")
    inputs_list, outputs_list, metadata_list = [], [], []
    for r in records:
        inp, out, meta = _split_record(r)
        inputs_list.append(inp)
        outputs_list.append(out)
        metadata_list.append(meta)
    client.create_examples(
        inputs=inputs_list,
        outputs=outputs_list,
        metadata=metadata_list,
        dataset_id=dataset.id,
    )
    print(f"uploaded {len(records)} examples → {dataset_name}")

def main():
    ap = argparse.ArgumentParser(description="Upload golden datasets to LangSmith.")
    ap.add_argument("--force", action="store_true", help="clear existing examples and re-upload")
    args = ap.parse_args()
    for filename, dataset_name in DATASET_MAP.items():
        upload_dataset(filename, dataset_name, force=args.force)
    print("done.")

if __name__ == "__main__":
    main()