#!/usr/bin/env python3
"""Search ENCODE Portal API for epigenetic datasets across cell types.

Uses biosample_ontology.term_name as the cell-type filter. Captures metadata
for every bigWig file matching each (cell, assay) combination.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE_URL = "https://www.encodeproject.org"
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("EPIGENETIC_DATA_DIR") or os.environ.get("CRISPR_DATA_DIR", DEFAULT_DATA_DIR)
META_DIR = os.path.join(DATA_DIR, "metadata")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

os.makedirs(META_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Cell-type ontology terms. ENCODE uses precise ontology labels, so we try
# several closely related names.
CELL_TYPES = {
    "K562": ["K562"],
    "HepG2": ["HepG2"],
    "CD14_monocytes": [
        "CD14-positive monocyte",
        "CD14-positive, CD16-positive monocyte",
        "CD14-positive, CD16-negative monocyte",
        "monocyte",
    ],
    "ARPE19": ["ARPE-19", "ARPE19"],
    "CD34_HSC": [
        "CD34-positive, CD38-negative hematopoietic stem cell",
        "CD34-positive hematopoietic stem cell",
        "CD34-positive stem cell",
        "hematopoietic stem cell",
        "CD34-positive",
    ],
    "iPSC": [
        "induced pluripotent stem cell",
    ],
}

# Histone marks (ChIP-seq) and accessibility assays
ASSAYS = {
    "H3K4me3": {"target.label": "H3K4me3", "assay_title": "ChIP-seq"},
    "H3K27ac": {"target.label": "H3K27ac", "assay_title": "ChIP-seq"},
    "H3K27me3": {"target.label": "H3K27me3", "assay_title": "ChIP-seq"},
    "H3K9me3": {"target.label": "H3K9me3", "assay_title": "ChIP-seq"},
    "ATAC-seq": {"assay_title": "ATAC-seq"},
    "DNase-seq": {"assay_title": "DNase-seq"},
}


def fetch(url, retries=3, timeout=60):
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "CRISPR-research/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Combination has no results
                return {"@graph": [], "total": 0}
            last_err = e
        except Exception as e:  # noqa
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"[ERR] failed {url}: {last_err}", file=sys.stderr)
    return None


def search_files(cell_key, search_terms, assay_key, assay_params, out_path):
    results = []
    for term in search_terms:
        params = {
            "type": "File",
            "file_type": "bigWig",
            "assembly": "GRCh38",
            "status": "released",
            "format": "json",
            "limit": "200",
            "biosample_ontology.term_name": term,
        }
        params.update(assay_params)
        qs = urllib.parse.urlencode(params)
        url = f"{BASE_URL}/search/?{qs}"
        print(f"  query {cell_key}/{assay_key} term={term}")
        data = fetch(url)
        if data and "@graph" in data:
            for item in data["@graph"]:
                otype = item.get("output_type", "")
                # Prefer fold change over control bigWigs for ChIP-seq
                if assay_key in ("ATAC-seq", "DNase-seq"):
                    pass
                else:
                    if "fold change" not in otype.lower():
                        continue
                results.append({
                    "accession": item.get("accession"),
                    "title": item.get("title"),
                    "output_type": otype,
                    "target": (item.get("target") or {}).get("label"),
                    "assay": item.get("assay_term_name"),
                    "biosample": item.get("biosample_ontology", {}).get("term_name"),
                    "simple_biosample": item.get("simple_biosample_summary"),
                    "assembly": item.get("assembly"),
                    "file_size": item.get("file_size"),
                    "href": item.get("href"),
                    "preferred_default": item.get("preferred_default"),
                    "lab": (item.get("lab") or {}).get("title"),
                    "dataset": item.get("dataset"),
                })
        time.sleep(0.4)
    # Dedup
    seen = {}
    for r in results:
        acc = r["accession"]
        if acc not in seen:
            seen[acc] = r
        elif r.get("preferred_default") and not seen[acc].get("preferred_default"):
            seen[acc] = r
    final = list(seen.values())
    # Sort: preferred_default first, then size descending
    final.sort(key=lambda x: (not x.get("preferred_default", False), -(x.get("file_size") or 0)))
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"  -> {len(final)} unique files for {cell_key}/{assay_key}")
    return final


def main():
    summary = {}
    for cell_key, terms in CELL_TYPES.items():
        summary[cell_key] = {}
        for assay_key, assay_params in ASSAYS.items():
            safe_cell = cell_key.replace("/", "_")
            out_path = os.path.join(META_DIR, f"{safe_cell}_{assay_key}.json")
            files = search_files(cell_key, terms, assay_key, assay_params, out_path)
            summary[cell_key][assay_key] = len(files)
    summary_path = os.path.join(META_DIR, "search_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
