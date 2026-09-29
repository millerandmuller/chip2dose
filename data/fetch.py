"""Download every input listed in sources.json, verify checksums, write provenance.csv.

Usage:
    python data/fetch.py            # verify cache, download what is missing
    python data/fetch.py --pin      # record sha256 of the current files in sources.json

A file whose checksum does not match the pinned value is an error, never a warning:
it means the upstream file changed and every downstream number would silently move.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
MANIFEST = DATA_DIR / "sources.json"
PROVENANCE = DATA_DIR / "provenance.csv"

# Some publishers answer scripted requests with an HTML challenge page; a browser
# user agent avoids that, and the checksum check catches it if it still happens.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, timeout: int = 120) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(target)


def try_download(source: dict, target: Path, attempts: int = 3) -> str:
    """Try each URL in order, retrying transient failures. Returns the URL that worked."""
    errors = []
    for url in source["urls"]:
        for attempt in range(1, attempts + 1):
            try:
                download(url, target)
                return url
            except Exception as exc:  # network errors differ by platform; all are reported
                errors.append(f"{url} (attempt {attempt}): {exc}")
                if attempt < attempts:
                    time.sleep(2 ** attempt)
    raise RuntimeError(
        f"all URLs failed for {source['id']} and no verified cached copy exists:\n  "
        + "\n  ".join(errors)
    )


def ensure_file(source: dict) -> tuple[Path, str]:
    """Make sure the file exists and matches its pinned checksum. Returns (path, status)."""
    target = DATA_DIR / source["file"]
    pinned = source.get("sha256")

    if target.exists() and pinned and sha256_of(target) == pinned:
        return target, "cached, checksum ok"

    url = try_download(source, target)
    actual = sha256_of(target)
    if pinned and actual != pinned:
        raise ChecksumError(
            f"{source['id']}: downloaded from {url} but sha256 {actual} != pinned {pinned}. "
            "The upstream file changed; results are not reproducible until this is resolved."
        )
    return target, "downloaded" if pinned else "downloaded, NOT PINNED"


def write_provenance(rows: list[dict]) -> None:
    fields = ["id", "file", "url", "license", "access_date", "sha256", "redistributed", "citation"]
    with PROVENANCE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pin", action="store_true", help="write current sha256 values into sources.json")
    args = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows, unpinned = [], []
    for source in manifest["sources"]:
        path, status = ensure_file(source)
        digest = sha256_of(path)
        if args.pin:
            source["sha256"] = digest
        elif not source.get("sha256"):
            unpinned.append(source["id"])
        print(f"  {source['id']:<20} {status:<24} {path.relative_to(DATA_DIR)}")
        rows.append(
            {
                "id": source["id"],
                "file": source["file"],
                "url": source["urls"][0],
                "license": source["license"],
                "access_date": source["access_date"],
                "sha256": digest,
                "redistributed": "yes" if source["redistribute"] else "no (fetched at pinned URL)",
                "citation": source["citation"],
            }
        )

    if args.pin:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("pinned sha256 for all sources")
    write_provenance(rows)
    print(f"wrote {PROVENANCE.relative_to(DATA_DIR.parent)} ({len(rows)} sources)")
    if unpinned:
        print("ERROR: unpinned sources: " + ", ".join(unpinned), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
