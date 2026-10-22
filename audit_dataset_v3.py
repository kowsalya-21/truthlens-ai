import os
import json
import urllib.request
import urllib.error
import ssl

MANIFEST_PATH = "E:/truthlens-ai/dataset_v3/manifest.json"

def audit_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: {MANIFEST_PATH} does not exist!")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Total entries in manifest: {len(items)}")
    
    # 1. Basic counts
    splits = {}
    labels = {}
    categories = {}
    generators = {}
    licenses = {}
    
    ids = set()
    filenames = set()
    urls = set()
    scenes = set()
    
    split_scenes = {"train": set(), "validation": set(), "held-out test": set()}
    split_labels = {"train": {"REAL": 0, "AI-GENERATED / MANIPULATED": 0}, 
                    "validation": {"REAL": 0, "AI-GENERATED / MANIPULATED": 0}, 
                    "held-out test": {"REAL": 0, "AI-GENERATED / MANIPULATED": 0}}

    duplicate_ids = []
    duplicate_filenames = []
    duplicate_urls = []
    invalid_entries = []

    for item in items:
        iid = item.get("id")
        fname = item.get("filename")
        lbl = item.get("label")
        surl = item.get("source_url")
        sdataset = item.get("source_dataset")
        lic = item.get("license")
        cat = item.get("category")
        gen = item.get("ai_generator")
        split = item.get("split")
        scene = item.get("original_scene_source_identifier")

        # Check required fields
        if not all([iid, fname, lbl, surl, sdataset, lic, cat, gen, split, scene]):
            invalid_entries.append((iid, "Missing required fields"))

        if iid in ids:
            duplicate_ids.append(iid)
        ids.add(iid)

        if fname in filenames:
            duplicate_filenames.append(fname)
        filenames.add(fname)

        if surl in urls:
            duplicate_urls.append(surl)
        urls.add(surl)

        splits[split] = splits.get(split, 0) + 1
        labels[lbl] = labels.get(lbl, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
        licenses[lic] = licenses.get(lic, 0) + 1
        generators[gen] = generators.get(gen, 0) + 1

        if split in split_scenes:
            split_scenes[split].add(scene)
            if lbl in split_labels[split]:
                split_labels[split][lbl] += 1

    print("\n--- SPLIT & LABEL DISTRIBUTION ---")
    for s, count in splits.items():
        print(f"Split '{s}': {count} items -> {split_labels.get(s)}")

    print("\n--- OVERALL LABELS ---")
    for l, count in labels.items():
        print(f"Label '{l}': {count}")

    print("\n--- DUPLICATES AUDIT ---")
    print(f"Duplicate IDs: {len(duplicate_ids)} ({duplicate_ids})")
    print(f"Duplicate Filenames: {len(duplicate_filenames)} ({duplicate_filenames})")
    print(f"Duplicate URLs: {len(duplicate_urls)} ({duplicate_urls})")
    print(f"Invalid Entries: {len(invalid_entries)} ({invalid_entries})")

    # Check source leakage across splits
    print("\n--- SOURCE-LEVEL SPLIT INTEGRITY ---")
    train_scenes = split_scenes["train"]
    val_scenes = split_scenes["validation"]
    test_scenes = split_scenes["held-out test"]

    leakage_train_test = train_scenes.intersection(test_scenes)
    leakage_val_test = val_scenes.intersection(test_scenes)
    leakage_train_val = train_scenes.intersection(val_scenes)

    print(f"Train scenes count: {len(train_scenes)}")
    print(f"Val scenes count: {len(val_scenes)}")
    print(f"Test scenes count: {len(test_scenes)}")
    print(f"Train <-> Test Scene Overlap: {len(leakage_train_test)} {leakage_train_test if leakage_train_test else '(None - 100% Isolated)'}")
    print(f"Val <-> Test Scene Overlap: {len(leakage_val_test)} {leakage_val_test if leakage_val_test else '(None - 100% Isolated)'}")
    print(f"Train <-> Val Scene Overlap: {len(leakage_train_val)} {leakage_train_val if leakage_train_val else '(None - 100% Isolated)'}")

    # Check URL verification
    print("\n--- URL VERIFICATION (HTTP PROBING) ---")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    unverifiable_urls = []
    verified_count = 0

    for idx, item in enumerate(items, 1):
        url = item["source_url"]
        fname = item["filename"]
        iid = item["id"]
        
        if url.upper().startswith("LOCAL:"):
            local_path = url.split(":", 1)[1].strip()
            if os.path.exists(local_path):
                size = os.path.getsize(local_path)
                verified_count += 1
            else:
                unverifiable_urls.append((iid, fname, url, f"Local file not found: {local_path}"))
            continue

        req = urllib.request.Request(
            url, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TruthLensResearch/1.0",
                     "Range": "bytes=0-1024"}
        )
        try:
            with urllib.request.urlopen(req, timeout=12, context=ctx) as response:
                status = response.getcode()
                if status in (200, 206):
                    verified_count += 1
                else:
                    unverifiable_urls.append((iid, fname, url, f"HTTP status {status}"))
        except urllib.error.HTTPError as e:
            # Try HEAD request if Range GET failed
            try:
                head_req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TruthLensResearch/1.0"}
                )
                head_req.get_method = lambda: "HEAD"
                with urllib.request.urlopen(head_req, timeout=12, context=ctx) as head_resp:
                    if head_resp.getcode() in (200, 206):
                        verified_count += 1
                    else:
                        unverifiable_urls.append((iid, fname, url, f"HTTP status {head_resp.getcode()}"))
            except Exception as e2:
                unverifiable_urls.append((iid, fname, url, f"HTTP Error: {e.code} / Head Error: {e2}"))
        except Exception as e:
            unverifiable_urls.append((iid, fname, url, f"Network Error: {str(e)}"))

    print(f"Verified items: {verified_count} / {len(items)}")
    if unverifiable_urls:
        print(f"Unverifiable / Failed entries ({len(unverifiable_urls)}):")
        for entry in unverifiable_urls:
            print(f"  ID: {entry[0]} | File: {entry[1]} | Error: {entry[3]} | URL: {entry[2]}")
    else:
        print("All 60 entries verified 100% active and accessible!")

if __name__ == "__main__":
    audit_manifest()
