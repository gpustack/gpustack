"""Publish the validated tree on its frozen parent, without rebasing changes."""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional

from hack.catalog_sync.models import REPO_ID, require
from hack.catalog_sync.run import apply, read_json


class GitHub:
    """Call the fixed GitHub API using the publication job's short-lived token."""

    def __init__(self, repository: str):
        require(bool(REPO_ID.fullmatch(repository)), "Invalid GitHub repository")
        self.repository = repository

    def call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        argv = [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            method,
            f"repos/{self.repository}/{path}",
        ]
        if body is not None:
            argv.extend(["--input", "-"])
        result = subprocess.run(
            argv,
            input=json.dumps(body) if body is not None else None,
            text=True,
            capture_output=True,
            timeout=120,
        )
        require(
            result.returncode == 0,
            f"GitHub {method} {path} failed; inspect repository permissions and state",
        )
        return json.loads(result.stdout)


def check_base(api: GitHub, base: str) -> None:
    """Stop publication if main has advanced since research."""
    current = api.call("GET", "git/ref/heads/main")["object"]["sha"]
    require(current == base, "main changed during research; rerun on the new base")


def publish(root: Path, bundle: dict, checksum: str, branch: str, api: GitHub) -> dict:
    """Create an exact-parent commit and draft PR from validated catalog files."""
    require(
        bool(re.fullmatch(r"automation/model-catalog-[0-9]+-[0-9]+", branch)),
        "Invalid automation branch",
    )
    # Verify hashes, allowed paths and unchanged checkout before any API mutation.
    apply(root, copy.deepcopy(bundle), checksum, write=False)
    require(bool(bundle["files"]), "No catalog changes to publish")
    base = bundle["request"]["base_sha"]
    check_base(api, base)
    base_tree = api.call("GET", f"git/commits/{base}")["tree"]["sha"]
    entries = []
    for path, content in bundle["files"].items():
        blob = api.call("POST", "git/blobs", {"content": content, "encoding": "base64"})
        entries.append(
            {"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]}
        )
    tree = api.call("POST", "git/trees", {"base_tree": base_tree, "tree": entries})
    title = "feat(catalog): add verified model deployment recipes"
    commit = api.call(
        "POST",
        "git/commits",
        {"message": title, "tree": tree["sha"], "parents": [base]},
    )
    check_base(api, base)
    # Create-only refs preserve any existing branch and manual review edits.
    api.call("POST", "git/refs", {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
    check_base(api, base)
    return api.call(
        "POST",
        "pulls",
        {
            "title": title,
            "head": branch,
            "base": "main",
            "body": bundle["body"],
            "draft": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = publish(
            Path.cwd(),
            read_json(args.output / "validated.json"),
            os.environ["CATALOG_BUNDLE_SHA256"],
            os.environ["CATALOG_PR_BRANCH"],
            GitHub(os.environ["GITHUB_REPOSITORY"]),
        )
        print(f"Draft PR: {result['html_url']}")
    except (ValueError, subprocess.TimeoutExpired) as error:
        print(f"Catalog publication failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
