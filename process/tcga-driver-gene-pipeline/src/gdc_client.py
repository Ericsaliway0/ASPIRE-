"""Query and download somatic mutation (MAF) files from the GDC API."""

from pathlib import Path
from typing import List, Tuple

import requests

from config import GDC_API_DATA, GDC_API_FILES, TCGA_MAF_DIR


def get_maf_file_ids(project: str, size: int = 200) -> List[Tuple[str, str]]:
    """Return (file_id, file_name) pairs for a TCGA project's masked somatic
    mutation files.

    Args:
        project: TCGA project abbreviation, e.g. "BRCA" (not "TCGA-BRCA").
        size: Maximum number of files to request from the GDC API.
    """
    params = {
        "filters": {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [f"TCGA-{project}"],
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": ["Masked Somatic Mutation"],
                    },
                },
            ],
        },
        "fields": "file_id,file_name",
        "format": "JSON",
        "size": size,
    }

    response = requests.post(
        GDC_API_FILES,
        json=params,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"GDC request failed: {response.status_code}")

    hits = response.json().get("data", {}).get("hits", [])
    return [(f["file_id"], f["file_name"]) for f in hits]


def download_maf_files(file_ids: List[Tuple[str, str]], project: str) -> None:
    """Download MAF files into `TCGA_MAF_DIR/{project}/`.

    Existing files are skipped. Downloads are written to a `.tmp` file and
    atomically renamed on success, so an interrupted download never leaves a
    corrupt file at the final path.
    """
    project_dir: Path = TCGA_MAF_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)

    for file_id, file_name in file_ids:
        file_path = project_dir / file_name
        tmp_path = file_path.with_suffix(".tmp")

        if file_path.exists():
            print(f"[SKIP] {file_name}")
            continue

        url = f"{GDC_API_DATA}/{file_id}"
        print(f"[DOWNLOAD] {file_name}")

        try:
            response = requests.get(url, stream=True, timeout=60)

            if response.status_code != 200:
                print(f"[ERROR] Failed {file_id} ({response.status_code})")
                continue

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            tmp_path.rename(file_path)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {file_name}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
