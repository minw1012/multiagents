import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


class SkillStore:
    """
    Stores downloaded skills and a small local manifest.
    Skills can be pulled from online git repositories and then exposed to agents as tools.
    """

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "skills_manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"skills": []}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("skills"), list):
                return data
        except Exception:
            pass
        return {"skills": []}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_skills(self) -> List[Dict[str, Any]]:
        return list(self._manifest.get("skills", []))

    def install_from_git(self, repo_url: str, ref: Optional[str] = None, alias: Optional[str] = None) -> Dict[str, Any]:
        if not repo_url or not isinstance(repo_url, str):
            return {"ok": False, "error": "repo_url is required"}

        safe_name = (alias or Path(repo_url.rstrip("/")).stem or f"skill_{now_ms()}").replace(" ", "_")
        target = self.root / safe_name

        if target.exists() and (target / ".git").exists():
            pull_cmd = ["git", "-C", str(target), "pull", "--ff-only"]
            pull = subprocess.run(pull_cmd, capture_output=True, text=True)
            if pull.returncode != 0:
                return {"ok": False, "error": pull.stderr.strip() or pull.stdout.strip()}
        elif target.exists() and not (target / ".git").exists():
            return {"ok": False, "error": f"target exists and is not a git repo: {target}"}
        else:
            clone_cmd = ["git", "clone", repo_url, str(target)]
            clone = subprocess.run(clone_cmd, capture_output=True, text=True)
            if clone.returncode != 0:
                return {"ok": False, "error": clone.stderr.strip() or clone.stdout.strip()}

        if ref:
            checkout_cmd = ["git", "-C", str(target), "checkout", ref]
            checkout = subprocess.run(checkout_cmd, capture_output=True, text=True)
            if checkout.returncode != 0:
                return {"ok": False, "error": checkout.stderr.strip() or checkout.stdout.strip()}

        rev_cmd = ["git", "-C", str(target), "rev-parse", "HEAD"]
        rev = subprocess.run(rev_cmd, capture_output=True, text=True)
        commit = rev.stdout.strip() if rev.returncode == 0 else ""

        record = {
            "name": safe_name,
            "repo_url": repo_url,
            "path": str(target),
            "ref": ref or "default",
            "commit": commit,
            "installed_at_ms": now_ms(),
        }
        skills = [s for s in self._manifest.get("skills", []) if s.get("name") != safe_name]
        skills.append(record)
        self._manifest["skills"] = skills
        self._save_manifest()
        return {"ok": True, "skill": record}
