import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import uuid
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = "dataset"


def _run(cmd, check=True, capture=False):
    if capture:
        proc = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.stdout.strip()
    else:
        return subprocess.run(cmd, check=check)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _timestamp():
    return _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _git_commit_short():
    try:
        return _run(["git", "rev-parse", "--short", "HEAD"], capture=True)
    except Exception:
        return None


def new_experiment_dir(scenario: str | None = None) -> str:
    stamp = _timestamp()
    name = f"exp-{stamp}" + (f"-{scenario}" if scenario else "")
    path = os.path.join(DATASET_DIR, name)
    _ensure_dir(path)
    return path


def harvest_rg(resource_group: str, out_dir: str) -> dict:
    _ensure_dir(out_dir)
    print(f"[harvest] Listing resources in RG '{resource_group}' ...")
    raw = _run([
        "az", "resource", "list", "--resource-group", resource_group, "-o", "json"
    ], capture=True)
    try:
        resources = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Failed to parse az resource list output:", e, file=sys.stderr)
        raise

    # Write aggregate
    _write_json(os.path.join(out_dir, "resources.json"), resources)

    # Write one file per resource for convenience
    seen = {}
    for r in resources:
        rid = r.get("id", "unknown-id")
        name = r.get("name") or rid.split("/")[-1]
        base = name.replace("/", "_")
        # de-dup if same name appears
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            base = f"{base}-{count}"
        _write_json(os.path.join(out_dir, f"{base}.json"), r)

    print(f"[harvest] Wrote {len(resources)} resources → {out_dir}")
    return {"count": len(resources)}


def cmd_deploy(args: argparse.Namespace) -> None:
    scenario = args.scenario
    exp_dir = args.out or new_experiment_dir(scenario)
    orig_dir = os.path.join(exp_dir, "original")
    _ensure_dir(orig_dir)

    meta = {
        "experiment_id": os.path.basename(exp_dir),
        "scenario": scenario,
        "resource_group": args.rg,
        "location": args.location,
        "template": args.template,
        "parameters": args.parameters,
        "commit_hash": _git_commit_short(),
        "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    print(f"[deploy] Ensuring resource group '{args.rg}' in {args.location} ...")
    _run(["az", "group", "create", "--name", args.rg, "--location", args.location, "-o", "none"])  # no output

    if args.template:
        deploy_name = f"crux-{uuid.uuid4().hex[:6]}"
        cmd = [
            "az", "deployment", "group", "create",
            "--resource-group", args.rg,
            "--name", deploy_name,
            "--template-file", args.template,
        ]
        if args.parameters:
            cmd += ["--parameters", f"@{args.parameters}"]
        print(f"[deploy] Deploying template: {args.template} ...")
        _run(cmd)
        meta["deployment_name"] = deploy_name
    else:
        print("[deploy] No template provided. Created resource group only.")

    # Harvest baseline
    harvest_rg(args.rg, orig_dir)
    _write_json(os.path.join(exp_dir, "metadata.json"), meta)
    # Initialize labels file
    labels_path = os.path.join(exp_dir, "labels.json")
    if not os.path.exists(labels_path):
        _write_json(labels_path, {})

    print(f"[deploy] Experiment directory: {exp_dir}")


def cmd_harvest(args: argparse.Namespace) -> None:
    out = args.out or os.path.join(DATASET_DIR, f"harvest-{_timestamp()}")
    harvest_rg(args.rg, out)
    print(f"[harvest] Output: {out}")


def cmd_mutate_raw(args: argparse.Namespace) -> None:
    # Minimal mutator: allow direct property updates on a given resource ID.
    if not args.resource_id:
        print("--resource-id is required for mutate-raw", file=sys.stderr)
        sys.exit(2)
    if not args.set:
        print("At least one --set key=value is required", file=sys.stderr)
        sys.exit(2)

    exp_dir = args.out or new_experiment_dir()
    mut_id = args.mutation_id or f"mutation-{uuid.uuid4().hex[:6]}"
    mut_dir = os.path.join(exp_dir, "mutated", mut_id)
    _ensure_dir(mut_dir)

    update_cmd = ["az", "resource", "update", "--ids", args.resource_id]
    for kv in args.set:
        update_cmd += ["--set", kv]

    print(f"[mutate] Applying update: {' '.join(update_cmd)}")
    _run(update_cmd)

    # Harvest after mutation
    harvest_rg(args.rg, mut_dir)

    mutation = {
        "mutation_id": mut_id,
        "resource_id": args.resource_id,
        "method": "az resource update",
        "args": {"set": args.set},
        "timestamp": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _write_json(os.path.join(mut_dir, "mutation.json"), mutation)
    print(f"[mutate] Wrote mutation metadata → {mut_dir}")


def cmd_cleanup(args: argparse.Namespace) -> None:
    print(f"[cleanup] Deleting resource group '{args.rg}' (no-wait) ...")
    _run(["az", "group", "delete", "--name", args.rg, "--yes", "--no-wait"])  # fire and forget
    print("[cleanup] Delete requested.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crux", description="CRUX harness: deploy, harvest, mutate, cleanup")
    sp = p.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("deploy", help="Create RG, deploy template (optional), harvest baseline")
    d.add_argument("--rg", required=True, help="Resource group name")
    d.add_argument("--location", required=True, help="Azure region (e.g., eastus)")
    d.add_argument("--template", help="Path to Bicep/ARM template file")
    d.add_argument("--parameters", help="Path to parameters JSON file")
    d.add_argument("--scenario", help="Scenario name (for experiment folder naming)")
    d.add_argument("--out", help="Experiment directory (default: dataset/exp-<ts>[-scenario])")
    d.set_defaults(func=cmd_deploy)

    h = sp.add_parser("harvest", help="Harvest all resources in an RG to a folder")
    h.add_argument("--rg", required=True, help="Resource group name")
    h.add_argument("--out", help="Output folder (default: dataset/harvest-<ts>)")
    h.set_defaults(func=cmd_harvest)

    m = sp.add_parser("mutate-raw", help="Apply direct property update to a resource and harvest")
    m.add_argument("--rg", required=True, help="Resource group name")
    m.add_argument("--resource-id", required=True, help="Target resource ID")
    m.add_argument("--set", action="append", help="Property update 'path=value'. Repeatable.")
    m.add_argument("--mutation-id", help="Optional mutation id (default: generated)")
    m.add_argument("--out", help="Experiment directory (default: new dataset/exp-<ts>)")
    m.set_defaults(func=cmd_mutate_raw)

    c = sp.add_parser("cleanup", help="Delete a resource group (no-wait)")
    c.add_argument("--rg", required=True, help="Resource group name")
    c.set_defaults(func=cmd_cleanup)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        sys.exit(e.returncode or 1)


if __name__ == "__main__":
    main()

