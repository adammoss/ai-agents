#!/usr/bin/env python3
"""Build a static catalogue for CMB anomaly-agent test runs."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "cmb_test_run_mimo"
DEFAULT_MODEL_LABEL = "Mimo V2.5 Pro"
SITE_DIR = ROOT / "docs"
CATALOGUE_VERSION = "v1.0"
PAPER_DIR = Path(
    "/Users/adammoss/Dropbox/Apps/Overleaf/"
    "AI Agents for Cosmological Anomaly Detection (Copy)"
)
CMB_ASSETS = {
    "planck_masked": "planckcmbwithmask256.png",
    "planck_unmasked": "planckcmbnomask256.png",
}


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    text = re.sub(r"^\*\*\s*", "", text)
    text = re.sub(r"\s*\*\*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text or default


def number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value.is_integer():
        return int(value)
    return value


def first_number(*values: Any) -> float | int | None:
    for value in values:
        converted = number(value)
        if converted is not None:
            return converted
    return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_number(folder: Path) -> int:
    match = re.match(r"Test_(\d+)", folder.name)
    return int(match.group(1)) if match else 9999


def slugify(folder_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", folder_name.lower())
    return slug.strip("-")


def normalize_verdict(raw: str) -> str:
    text = clean_text(raw, "Unknown").lower()
    if "no anomalies" in text:
        return "Consistent"
    if "borderline" in text:
        return "Borderline"
    if "anomalies found" in text or text == "anomaly":
        return "Anomaly"
    return clean_text(raw, "Unknown")


def normalize_novelty(raw: str) -> str:
    text = clean_text(raw, "Unknown")
    lower = text.lower()
    if lower.startswith("novel"):
        return "Novel"
    if lower.startswith("variation"):
        return "Variation"
    if lower.startswith("repeat"):
        return "Repeat"
    return text


def infer_tail(results: dict[str, Any]) -> str:
    tail = clean_text(results.get("tail"))
    if tail and tail.lower() != "unknown":
        return tail.lower()
    test_type = clean_text(results.get("test_type")).lower()
    if "upper" in test_type:
        return "upper"
    if "lower" in test_type:
        return "lower"
    if "two" in test_type:
        return "two-tailed"
    return "unspecified"


def infer_family(result_summary: dict[str, Any], title: str, description: str) -> str:
    family = clean_text((result_summary.get("test_signature") or {}).get("family"))
    if family:
        return family.replace("_", " ")
    text = f"{title} {description}".lower()
    if "alignment" in text or "axis" in text or "plane" in text:
        return "alignment"
    if "parity" in text:
        return "parity"
    if "hemispher" in text or "dipole" in text or "latitude" in text:
        return "direction"
    if "power" in text or "spectrum" in text:
        return "power"
    if "correlation" in text:
        return "cross correlation"
    if "variance" in text or "skewness" in text or "kurtosis" in text:
        return "moments"
    if "region" in text or "boundary" in text or "genus" in text or "lacunarity" in text:
        return "morphology"
    return "other"


def relative_to_root(path: Path) -> str:
    return path.relative_to(SITE_DIR).as_posix()


def copy_if_exists(src: Path, dest: Path) -> str | None:
    if not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return relative_to_root(dest)


def write_public_json(
    src: Path,
    dest: Path,
    source: dict[str, Any],
    folder_name: str,
    figure_png: str | None,
    figure_pdf: str | None,
) -> str | None:
    if not src.exists():
        return None

    def scrub(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {item_key: scrub(item_value, item_key) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if key == "plot_path" and figure_png:
            return figure_png
        if key == "plot_pdf_path" and figure_pdf:
            return figure_pdf
        if key == "output_dir":
            return f"{source['path_display']}/{folder_name}"
        if isinstance(value, str) and "/content/drive/" in value:
            return re.sub(r"/content/drive/[^\s'\",)]+", "[original-cloud-path]", value)
        return value

    dest.parent.mkdir(parents=True, exist_ok=True)
    data = read_json(src)
    dest.write_text(json.dumps(scrub(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return relative_to_root(dest)


def html_text(text: str) -> str:
    return html.escape(clean_text(text))


def paragraph(text: str) -> str:
    text = clean_text(text)
    if not text:
        return '<p class="muted">Not recorded.</p>'
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return "\n".join(f"<p>{html.escape(part)}</p>" for part in parts)


def fmt_metric(value: Any) -> str:
    value = number(value)
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.3g}"
    if abs(value) >= 1:
        return f"{value:.4g}"
    return f"{value:.3g}"


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def split_source_spec(spec: str) -> tuple[Path, str | None]:
    raw_path, sep, label = spec.partition("::")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path, clean_text(label) if sep else None


def unique_id(base: str, used: set[str]) -> str:
    source_id = slugify(base) or "source"
    candidate = source_id
    index = 2
    while candidate in used:
        candidate = f"{source_id}-{index}"
        index += 1
    used.add(candidate)
    return candidate


def make_source(path: Path, kind: str, used: set[str], label: str | None = None) -> dict[str, Any]:
    label = clean_text(label or path.stem or path.name)
    source_id = unique_id(label, used)
    return {
        "id": source_id,
        "label": label,
        "model": label,
        "kind": kind,
        "path": path,
        "path_display": display_path(path),
    }


def source_has_tests(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("Test_*/updated_results.json")) or any(path.glob("Test_*/result_summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Run directory to include. Use PATH::Model to set the LLM model label.",
    )
    parser.add_argument(
        "--json",
        dest="json_files",
        action="append",
        default=[],
        help="Single JSON file to include. Use PATH::Model to set the LLM model label.",
    )
    parser.add_argument(
        "--discover-runs",
        action="store_true",
        help="Include every runs/* directory that contains Test_*/updated_results.json.",
    )
    parser.add_argument("--out", default=str(SITE_DIR), help="Output directory for the static site.")
    parser.add_argument("--version", default=CATALOGUE_VERSION, help="Catalogue version label.")
    return parser.parse_args()


def build_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    used: set[str] = set()
    sources: list[dict[str, Any]] = []

    if args.discover_runs:
        runs_root = ROOT / "runs"
        for path in sorted(runs_root.iterdir()) if runs_root.exists() else []:
            if source_has_tests(path):
                sources.append(make_source(path, "run_directory", used))

    for spec in args.run_dir:
        path, label = split_source_spec(spec)
        sources.append(make_source(path, "run_directory", used, label))

    for spec in args.json_files:
        path, label = split_source_spec(spec)
        sources.append(make_source(path, "json_file", used, label))

    if not sources:
        sources.append(make_source(DEFAULT_RUN_DIR, "run_directory", used, DEFAULT_MODEL_LABEL))

    return sources


def prepare_site_dir() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    for rel in ("assets/figures", "assets/cmb", "data", "tests"):
        shutil.rmtree(SITE_DIR / rel, ignore_errors=True)


def copy_cmb_assets() -> dict[str, str | None]:
    copied: dict[str, str | None] = {}
    for key, filename in CMB_ASSETS.items():
        copied[key] = copy_if_exists(PAPER_DIR / filename, SITE_DIR / "assets" / "cmb" / filename)
    return copied


def normalized_metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(record.get("metrics") or {})
    if not metrics:
        return {}
    return {
        "planck_stat": number(metrics.get("planck_stat")),
        "mean_sim_stat": number(metrics.get("mean_sim_stat")),
        "std_sim_stat": number(metrics.get("std_sim_stat")),
        "median_sim_stat": number(metrics.get("median_sim_stat")),
        "sigma": number(metrics.get("sigma")),
        "p_value_one_tailed": number(metrics.get("p_value_one_tailed")),
        "p_value_two_tailed": number(metrics.get("p_value_two_tailed")),
        "primary_p_value": number(metrics.get("primary_p_value")),
        "n_sims": number(metrics.get("n_sims")),
        "n_valid_simulations": number(metrics.get("n_valid_simulations")),
        "sim_min": number(metrics.get("sim_min")),
        "sim_max": number(metrics.get("sim_max")),
        "thresholds": metrics.get("thresholds") if isinstance(metrics.get("thresholds"), list) else None,
    }


def resolve_asset_path(path_value: Any, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path_text = str(path_value)
    if "[original-cloud-path]" in path_text or "/content/drive/" in path_text:
        return None
    path = Path(path_text).expanduser()
    candidates = [path] if path.is_absolute() else [base_dir / path, ROOT / path, ROOT / "docs" / path, SITE_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def copy_asset_from_record(
    record: dict[str, Any],
    key: str,
    dest: Path,
    base_dir: Path,
) -> str | None:
    assets = record.get("assets") if isinstance(record.get("assets"), dict) else {}
    src = resolve_asset_path(assets.get(key), base_dir)
    if src is None:
        return None
    return copy_if_exists(src, dest)


def build_test_entry(
    source: dict[str, Any],
    slug: str,
    folder_name: str,
    updated: dict[str, Any],
    result_summary: dict[str, Any],
    assets: dict[str, str | None],
    test_index: int | None = None,
) -> dict[str, Any]:
    normalized = bool(updated.get("metrics") and updated.get("title"))
    results = dict(updated.get("results") or result_summary.get("results") or {})
    if not results and result_summary:
        results = {k: v for k, v in result_summary.items() if k != "custom_summary"}
        if "p_value" in results and "p_value_one_tailed" not in results and "p_value_two_tailed" not in results:
            tail_str = str(results.get("tail") or "").lower()
            if "two" in tail_str:
                results["p_value_two_tailed"] = results["p_value"]
            else:
                results["p_value_one_tailed"] = results["p_value"]
    summary = (
        updated.get("updated_summary")
        or updated.get("updated_test_summary")
        or result_summary.get("updated_test_summary")
        or result_summary.get("test_summary")
        or {}
    )
    custom = dict(updated.get("custom_summary") or result_summary.get("custom_summary") or {})

    if normalized:
        title = clean_text(updated.get("title") or folder_name.replace("_", " "))
        description = clean_text(updated.get("description"))
        tail = clean_text(updated.get("tail"), "unspecified")
        p_one = number((updated.get("metrics") or {}).get("p_value_one_tailed"))
        p_two = number((updated.get("metrics") or {}).get("p_value_two_tailed"))
        primary_p = number((updated.get("metrics") or {}).get("primary_p_value"))
        metrics = normalized_metrics(updated)
    else:
        title = clean_text(
            results.get("test_name")
            or result_summary.get("test_name")
            or summary.get("Test name")
            or folder_name.replace("_", " ")
        )
        description = clean_text(
            summary.get("Description")
            or results.get("test_description")
            or result_summary.get("test_description")
        )
        tail = infer_tail(results)
        p_one = first_number(results.get("p_value_one_tailed"), result_summary.get("p_value_one_tailed"))
        p_two = first_number(results.get("p_value_two_tailed"), result_summary.get("p_value_two_tailed"))
        primary_p = p_two if tail == "two-tailed" and p_two is not None else p_one
        if primary_p is None:
            primary_p = p_two
        metrics = {
            "planck_stat": number(results.get("planck_stat")),
            "mean_sim_stat": number(results.get("mean_sim_stat")),
            "std_sim_stat": number(results.get("std_sim_stat")),
            "median_sim_stat": number(results.get("median_sim_stat")),
            "sigma": number(results.get("sigma")),
            "p_value_one_tailed": p_one,
            "p_value_two_tailed": p_two,
            "primary_p_value": primary_p,
            "n_sims": number(results.get("n_sims") or result_summary.get("n_sims")),
            "n_valid_simulations": number(custom.get("n_valid_simulations")),
            "sim_min": number(custom.get("sim_min")),
            "sim_max": number(custom.get("sim_max")),
            "thresholds": custom.get("thresholds") if isinstance(custom.get("thresholds"), list) else None,
        }

    verdict_raw = clean_text(summary.get("Verdict") or updated.get("verdict_raw") or updated.get("verdict"), "Unknown")
    novelty_raw = clean_text(summary.get("Test novelty") or updated.get("novelty_raw") or updated.get("novelty"), "Unknown")
    source_test_id = f"{source['id']}-{slug}"
    return {
        "id": source_test_id,
        "local_id": slug,
        "folder": folder_name,
        "test_number": test_index if test_index is not None else test_number(Path(folder_name)),
        "source_id": source["id"],
        "source_label": source["label"],
        "source_run": source["label"],
        "source_kind": source["kind"],
        "source_path": source["path_display"],
        "model": source["model"],
        "title": title,
        "description": description,
        "hypothesis": clean_text(updated.get("hypothesis") or results.get("test_hypothesis")),
        "test_type": clean_text(updated.get("test_type") or results.get("test_type")),
        "tail": tail,
        "family": clean_text(updated.get("family") or infer_family(result_summary, title, description), "other"),
        "justification": clean_text(updated.get("justification") or results.get("justification")),
        "results_text": clean_text(updated.get("results_text") or summary.get("Results")),
        "interpretation": clean_text(updated.get("interpretation") or summary.get("Interpretation")),
        "literature": clean_text(updated.get("literature") or summary.get("Comparison to literature")),
        "meets_hypothesis": clean_text(updated.get("meets_hypothesis") or summary.get("Meets hypothesis?")),
        "novelty": normalize_novelty(novelty_raw),
        "novelty_raw": novelty_raw,
        "verdict": normalize_verdict(verdict_raw),
        "verdict_raw": verdict_raw,
        "metrics": metrics,
        "assets": assets,
        "page": f"tests/{source['id']}/{slug}/",
        "analysis_code": str(updated.get("analysis_code") or result_summary.get("analysis_code") or "").strip(),
    }


def load_run_directory(source: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = source["path"]
    if not source_has_tests(run_dir):
        raise SystemExit(f"No Test_*/updated_results.json files found in run directory: {run_dir}")

    tests: list[dict[str, Any]] = []
    folders = sorted(
        [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("Test_")],
        key=lambda p: (test_number(p), p.name),
    )

    for folder in folders:
        updated = read_json(folder / "updated_results.json")
        result_summary = read_json(folder / "result_summary.json")
        slug = slugify(folder.name)
        figure_png = copy_if_exists(
            folder / "statistic_figure.png",
            SITE_DIR / "assets" / "figures" / source["id"] / f"{slug}.png",
        )
        figure_pdf = copy_if_exists(
            folder / "statistic_figure.pdf",
            SITE_DIR / "assets" / "figures" / source["id"] / f"{slug}.pdf",
        )
        updated_json = write_public_json(
            folder / "updated_results.json",
            SITE_DIR / "data" / "raw" / source["id"] / f"{slug}.updated_results.json",
            source,
            folder.name,
            figure_png,
            figure_pdf,
        )
        result_json = write_public_json(
            folder / "result_summary.json",
            SITE_DIR / "data" / "raw" / source["id"] / f"{slug}.result_summary.json",
            source,
            folder.name,
            figure_png,
            figure_pdf,
        )
        planck_npy = copy_if_exists(
            folder / "planck_statistic.npy",
            SITE_DIR / "data" / "statistics" / source["id"] / f"{slug}.planck_statistic.npy",
        )
        sims_npy = copy_if_exists(
            folder / "simulation_statistics.npy",
            SITE_DIR / "data" / "statistics" / source["id"] / f"{slug}.simulation_statistics.npy",
        )
        assets = {
            "figure_png": figure_png,
            "figure_pdf": figure_pdf,
            "updated_results_json": updated_json,
            "result_summary_json": result_json,
            "planck_statistic_npy": planck_npy,
            "simulation_statistics_npy": sims_npy,
        }
        tests.append(build_test_entry(source, slug, folder.name, updated, result_summary, assets))

    return tests


def extract_json_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("tests"), list):
        return [item for item in data["tests"] if isinstance(item, dict)]
    if any(key in data for key in ("results", "updated_summary", "updated_test_summary", "metrics", "test_summary")):
        return [data]
    dict_values = [value for value in data.values() if isinstance(value, dict)]
    if dict_values:
        return dict_values
    return []


def load_json_file(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = source["path"]
    if not path.exists():
        raise SystemExit(f"JSON source not found: {path}")
    data = read_json_value(path)
    records = extract_json_records(data)
    if not records:
        raise SystemExit(f"No test records found in JSON source: {path}")

    tests: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        folder_name = clean_text(record.get("folder") or record.get("id") or f"Test_{index:03d}")
        title_hint = clean_text(record.get("title") or (record.get("results") or {}).get("test_name"))
        slug = slugify(folder_name if folder_name else title_hint or f"test-{index:03d}")
        if not slug.startswith("test-") and title_hint:
            slug = slugify(f"test-{index:03d}-{title_hint}")

        figure_png = copy_asset_from_record(
            record,
            "figure_png",
            SITE_DIR / "assets" / "figures" / source["id"] / f"{slug}.png",
            path.parent,
        )
        figure_pdf = copy_asset_from_record(
            record,
            "figure_pdf",
            SITE_DIR / "assets" / "figures" / source["id"] / f"{slug}.pdf",
            path.parent,
        )
        raw_json_dest = SITE_DIR / "data" / "raw" / source["id"] / f"{slug}.json"
        raw_json_dest.parent.mkdir(parents=True, exist_ok=True)
        raw_json_dest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        assets = {
            "figure_png": figure_png,
            "figure_pdf": figure_pdf,
            "updated_results_json": relative_to_root(raw_json_dest),
            "result_summary_json": None,
            "planck_statistic_npy": copy_asset_from_record(
                record,
                "planck_statistic_npy",
                SITE_DIR / "data" / "statistics" / source["id"] / f"{slug}.planck_statistic.npy",
                path.parent,
            ),
            "simulation_statistics_npy": copy_asset_from_record(
                record,
                "simulation_statistics_npy",
                SITE_DIR / "data" / "statistics" / source["id"] / f"{slug}.simulation_statistics.npy",
                path.parent,
            ),
        }
        raw_test_number = number(record.get("test_number"))
        tests.append(
            build_test_entry(
                source,
                slug,
                folder_name,
                record,
                record,
                assets,
                int(raw_test_number) if raw_test_number is not None else index,
            )
        )

    return tests


def load_tests(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for source in sources:
        if source["kind"] == "json_file":
            tests.extend(load_json_file(source))
        else:
            tests.extend(load_run_directory(source))
    return tests


def build_metadata(
    tests: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    version: str,
    visual_assets: dict[str, str | None],
) -> dict[str, Any]:
    verdict_counts: dict[str, int] = {}
    novelty_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for test in tests:
        verdict_counts[test["verdict"]] = verdict_counts.get(test["verdict"], 0) + 1
        novelty_counts[test["novelty"]] = novelty_counts.get(test["novelty"], 0) + 1
        family_counts[test["family"]] = family_counts.get(test["family"], 0) + 1
        source_counts[test["source_label"]] = source_counts.get(test["source_label"], 0) + 1
        model_counts[test["model"]] = model_counts.get(test["model"], 0) + 1

    p_values = [
        test["metrics"]["primary_p_value"]
        for test in tests
        if test["metrics"].get("primary_p_value") is not None
    ]

    return {
        "title": "AI Agents for Cosmological Anomaly Detection Test Catalogue",
        "version": version,
        "sources": [
            {
                "id": source["id"],
                "label": source["label"],
                "model": source["model"],
                "kind": source["kind"],
                "path": source["path_display"],
            }
            for source in sources
        ],
        "source_counts": dict(sorted(source_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "visual_assets": visual_assets,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "curation_note": (
            "This catalogue preserves agent-generated hypotheses, code, results, "
            "and summaries from each run. Literature comparison text is included "
            "as provenance and should be checked before manuscript use."
        ),
        "n_tests": len(tests),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "novelty_counts": dict(sorted(novelty_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "minimum_primary_p_value": min(p_values) if p_values else None,
        "registry_files": {
            "json": "data/tests.json",
            "csv": "data/tests.csv",
        },
    }


def write_registry(tests: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    data_dir = SITE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    registry_tests = []
    for test in tests:
        item = {k: v for k, v in test.items() if k != "analysis_code"}
        registry_tests.append(item)

    (data_dir / "tests.json").write_text(
        json.dumps({"metadata": metadata, "tests": registry_tests}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for test in tests:
        metrics = test["metrics"]
        rows.append(
            {
                "id": test["id"],
                "source_id": test["source_id"],
                "source_label": test["source_label"],
                "model": test["model"],
                "test_number": test["test_number"],
                "title": test["title"],
                "verdict": test["verdict"],
                "novelty": test["novelty"],
                "family": test["family"],
                "tail": test["tail"],
                "primary_p_value": metrics.get("primary_p_value"),
                "p_value_one_tailed": metrics.get("p_value_one_tailed"),
                "p_value_two_tailed": metrics.get("p_value_two_tailed"),
                "sigma": metrics.get("sigma"),
                "planck_stat": metrics.get("planck_stat"),
                "mean_sim_stat": metrics.get("mean_sim_stat"),
                "std_sim_stat": metrics.get("std_sim_stat"),
                "n_sims": metrics.get("n_sims"),
                "page": test["page"],
                "figure_png": test["assets"].get("figure_png"),
                "raw_json": test["assets"].get("updated_results_json"),
            }
        )

    with (data_dir / "tests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (data_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_index(metadata: dict[str, Any]) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")
    model_count = len(metadata.get("model_counts") or {})
    model_label = f"{model_count} model" if model_count == 1 else f"{model_count} models"
    (SITE_DIR / "catalogue.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(metadata["title"])}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <p class="eyebrow">Frozen catalogue {html.escape(metadata["version"])} · {html.escape(model_label)}</p>
      <h1>AI Agents for Cosmological Anomaly Detection Test Catalogue</h1>
    </div>
    <nav class="toplinks" aria-label="Catalogue navigation">
      <a href="index.html">← Paper</a>
      <a href="data/tests.json">JSON</a>
      <a href="data/tests.csv">CSV</a>
      <a href="data/manifest.json">Manifest</a>
    </nav>
  </header>

  <main>
    <section class="notice">
      <strong>Provenance:</strong> {html.escape(metadata["curation_note"])}
    </section>

    <section class="summary-grid" id="summary-grid" aria-label="Catalogue summary"></section>

    <section class="controls" aria-label="Catalogue filters">
      <label class="searchbox">
        <span>Search</span>
        <input id="search" type="search" placeholder="Name, hypothesis, family, interpretation">
      </label>
      <label>
        <span>Model</span>
        <select id="model-filter"></select>
      </label>
      <label>
        <span>Verdict</span>
        <select id="verdict-filter"></select>
      </label>
      <label>
        <span>Novelty</span>
        <select id="novelty-filter"></select>
      </label>
      <label>
        <span>Family</span>
        <select id="family-filter"></select>
      </label>
      <label>
        <span>Sort</span>
        <select id="sort-order">
          <option value="pvalue">Smallest p-value</option>
          <option value="number">Test number</option>
          <option value="sigma">Largest sigma</option>
          <option value="title">Title</option>
        </select>
      </label>
      <button id="reset" type="button">Reset</button>
    </section>

    <section class="table-shell" aria-label="Agent-generated tests">
      <div class="table-meta">
        <h2>Tests</h2>
        <p id="result-count"></p>
      </div>
      <div id="test-list" class="test-list"></div>
    </section>
  </main>

  <footer>
    <p>Generated from {html.escape(model_label)} on {html.escape(metadata["generated_at_utc"])}.</p>
  </footer>
  <script src="app.js"></script>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_styles() -> None:
    target = SITE_DIR / "styles.css"
    if target.exists():
        return
    target.write_text(
        """@font-face {
  font-family: system-ui;
  src: local(".SFNSText-Regular");
}

:root {
  --bg: #0b1118;
  --surface: #eef2f4;
  --panel: #fbfcfd;
  --panel-soft: #f3f6f7;
  --text: #1c2630;
  --muted: #667381;
  --line: #cbd4da;
  --teal: #168b91;
  --teal-dark: #0d5d65;
  --cyan: #43b7c2;
  --amber: #c47a1d;
  --red: #c83d32;
  --green: #23835f;
  --ink: #111923;
  --code: #101820;
  --shadow: 0 1px 2px rgba(9, 17, 25, 0.09), 0 10px 28px rgba(9, 17, 25, 0.08);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background:
    radial-gradient(circle at 16% -10%, rgba(67, 183, 194, 0.18), transparent 28rem),
    radial-gradient(circle at 90% 2%, rgba(196, 122, 29, 0.13), transparent 26rem),
    linear-gradient(180deg, #0b1118 0, #111923 210px, var(--surface) 211px);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}

a {
  color: var(--teal-dark);
  text-decoration-thickness: 0.08em;
  text-underline-offset: 0.18em;
}

.topbar {
  position: relative;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  min-height: 190px;
  padding: 34px clamp(16px, 4vw, 44px) 26px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.16);
  background:
    linear-gradient(90deg, rgba(7, 12, 19, 0.94), rgba(7, 12, 19, 0.70) 48%, rgba(7, 12, 19, 0.22)),
    url("assets/cmb/planckcmbwithmask256.png") center / cover no-repeat,
    #101820;
  color: #f7fbfc;
  overflow: hidden;
}

.topbar::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(rgba(255, 255, 255, 0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
  background-size: 100% 36px, 48px 100%;
  mix-blend-mode: screen;
}

.topbar > * {
  position: relative;
  z-index: 1;
}

.eyebrow {
  margin: 0 0 6px;
  color: #a7dbe1;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1,
h2,
h3 {
  margin: 0;
  color: var(--ink);
  letter-spacing: 0;
}

.topbar h1 {
  color: #f8fbfc;
  text-shadow: 0 2px 16px rgba(0, 0, 0, 0.34);
}

h1 {
  max-width: 920px;
  font-size: clamp(1.55rem, 3vw, 2.45rem);
  line-height: 1.08;
}

h2 {
  font-size: 1.1rem;
}

h3 {
  font-size: 0.98rem;
}

main {
  width: min(1480px, 100%);
  margin: 0 auto;
  padding: 18px clamp(14px, 3vw, 32px) 48px;
}

.toplinks {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.toplinks a,
button,
.link-button {
  min-height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(203, 212, 218, 0.88);
  border-radius: 6px;
  background: rgba(251, 252, 253, 0.92);
  color: var(--ink);
  padding: 7px 12px;
  font-weight: 700;
  font-size: 0.88rem;
  text-decoration: none;
  box-shadow: var(--shadow);
}

.topbar .toplinks a {
  background: rgba(7, 12, 19, 0.54);
  border-color: rgba(255, 255, 255, 0.28);
  color: #f8fbfc;
  backdrop-filter: blur(8px);
}

button {
  cursor: pointer;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(130px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.notice {
  position: relative;
  margin-bottom: 14px;
  border: 1px solid #d3dde2;
  border-left: 5px solid var(--cyan);
  border-radius: 8px;
  background: rgba(251, 252, 253, 0.94);
  color: #2b3540;
  padding: 11px 13px;
  font-size: 0.92rem;
  box-shadow: var(--shadow);
}

.notice::before {
  content: "";
  position: absolute;
  left: -1px;
  right: -1px;
  top: -1px;
  height: 3px;
  border-radius: 8px 8px 0 0;
  background: linear-gradient(90deg, var(--teal-dark), var(--cyan), var(--amber), var(--red));
}

.notice strong {
  color: #101820;
}

.metric-card {
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(242, 247, 248, 0.96));
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  box-shadow: var(--shadow);
}

.metric-card::before {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--teal), var(--cyan), var(--amber), var(--red));
}

.metric-card span {
  display: block;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
}

.metric-card strong {
  display: block;
  margin-top: 3px;
  font-size: 1.5rem;
  color: var(--ink);
}

.controls {
  display: grid;
  grid-template-columns: minmax(240px, 2fr) repeat(5, minmax(130px, 1fr)) auto;
  gap: 10px;
  align-items: end;
  margin: 12px 0;
  padding: 12px;
  background:
    linear-gradient(180deg, rgba(247, 250, 251, 0.98), rgba(232, 239, 242, 0.98));
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

label span {
  display: block;
  margin-bottom: 4px;
  color: #4b5563;
  font-size: 0.76rem;
  font-weight: 800;
  text-transform: uppercase;
}

input,
select {
  width: 100%;
  min-height: 38px;
  border: 1px solid #b7c3ca;
  border-radius: 6px;
  background: #ffffff;
  color: var(--ink);
  padding: 7px 10px;
  font: inherit;
}

.table-shell {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  overflow: hidden;
}

.table-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 14px;
  border-bottom: 1px solid var(--line);
}

.table-meta p {
  margin: 0;
  color: var(--muted);
}

.test-list {
  display: grid;
}

.test-row {
  display: grid;
  grid-template-columns: 92px minmax(280px, 2.1fr) minmax(120px, 0.7fr) minmax(120px, 0.7fr) minmax(90px, 0.55fr) minmax(90px, 0.55fr) minmax(105px, 0.65fr) minmax(120px, 0.7fr);
  gap: 14px;
  align-items: center;
  padding: 13px 14px;
  border-bottom: 1px solid #ece7df;
}

.test-row:hover {
  background: #f5f9fa;
}

.test-thumb {
  width: 92px;
  aspect-ratio: 16 / 10;
  border: 1px solid #d4dde2;
  border-radius: 6px;
  background:
    linear-gradient(135deg, rgba(22, 139, 145, 0.20), rgba(196, 122, 29, 0.18)),
    #e9eff2;
  overflow: hidden;
}

.test-thumb img {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
}

.test-row:last-child {
  border-bottom: 0;
}

.test-title {
  min-width: 0;
}

.test-title a {
  display: inline;
  color: var(--ink);
  font-weight: 800;
  text-decoration: none;
}

.test-title a:hover {
  text-decoration: underline;
}

.test-title p {
  margin: 3px 0 0;
  color: var(--muted);
  font-size: 0.9rem;
  overflow-wrap: anywhere;
}

.label {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  max-width: 100%;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 0.78rem;
  font-weight: 800;
  color: #ffffff;
  line-height: 1.15;
}

.label.anomaly {
  background: var(--red);
}

.label.borderline {
  background: var(--amber);
}

.label.consistent {
  background: var(--teal-dark);
}

.label.unknown {
  background: #52525b;
}

.cell-label {
  display: none;
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
}

.number {
  font-variant-numeric: tabular-nums;
  color: var(--ink);
  font-weight: 750;
}

.muted {
  color: var(--muted);
}

.detail-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 0.55fr);
  gap: 18px;
}

.panel {
  background: rgba(251, 252, 253, 0.97);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 16px;
  margin-bottom: 14px;
}

.detail-header {
  display: grid;
  gap: 12px;
  margin-bottom: 14px;
}

.detail-header h1 {
  max-width: 980px;
}

.status-line {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.figure-panel img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid #bfccd3;
  border-radius: 6px;
  background: #ffffff;
}

.figure-panel h2,
.panel h2 {
  margin-bottom: 10px;
}

.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}

.stats-table th,
.stats-table td {
  padding: 8px 0;
  border-bottom: 1px solid #ece7df;
  text-align: left;
  vertical-align: top;
}

.stats-table th {
  width: 52%;
  color: var(--muted);
  font-weight: 800;
}

.asset-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-top: 12px;
}

.asset-list a {
  min-height: 36px;
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fffdf9;
  padding: 7px 10px;
  font-weight: 750;
  text-decoration: none;
}

details {
  margin-top: 8px;
}

summary {
  cursor: pointer;
  font-weight: 800;
}

pre {
  max-height: 520px;
  overflow: auto;
  border-radius: 8px;
  background:
    linear-gradient(180deg, rgba(67, 183, 194, 0.08), transparent 90px),
    var(--code);
  color: #f8fafc;
  padding: 14px;
  font-size: 0.82rem;
  line-height: 1.45;
}

code {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}

footer {
  width: min(1480px, 100%);
  margin: 0 auto;
  padding: 0 clamp(14px, 3vw, 32px) 28px;
  color: var(--muted);
  font-size: 0.86rem;
}

@media (max-width: 1060px) {
  .summary-grid {
    grid-template-columns: repeat(3, minmax(140px, 1fr));
  }

  .controls {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .detail-layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 780px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .summary-grid,
  .controls {
    grid-template-columns: 1fr;
  }

  .test-row {
    grid-template-columns: 1fr 1fr;
    align-items: start;
  }

  .test-thumb {
    width: 100%;
    max-width: 180px;
  }

  .test-title {
    grid-column: 1 / -1;
  }

  .cell-label {
    display: block;
  }

  .asset-list {
    grid-template-columns: 1fr;
  }
}
""",
        encoding="utf-8",
    )


def write_app_js() -> None:
    (SITE_DIR / "app.js").write_text(
        """const state = {
  tests: [],
  filters: {
    q: "",
    model: "All",
    verdict: "All",
    novelty: "All",
    family: "All",
    sort: "pvalue",
  },
};

const el = (id) => document.getElementById(id);

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  const n = Number(value);
  if (Math.abs(n) >= 100) return n.toPrecision(4);
  if (Math.abs(n) >= 1) return n.toFixed(3).replace(/0+$/, "").replace(/\\.$/, "");
  return n.toPrecision(3);
}

function labelClass(verdict) {
  return String(verdict || "unknown").toLowerCase().replace(/[^a-z]+/g, "-");
}

function uniq(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function fillSelect(node, values) {
  node.innerHTML = "";
  ["All", ...values].forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    node.appendChild(opt);
  });
}

function renderSummary(metadata) {
  const counts = metadata.verdict_counts || {};
  const novelty = metadata.novelty_counts || {};
  const cards = [
    ["Tests", metadata.n_tests],
    ["Models", Object.keys(metadata.model_counts || {}).length],
    ["Anomaly", counts.Anomaly || 0],
    ["Borderline", counts.Borderline || 0],
    ["Novel", novelty.Novel || 0],
    ["Minimum p", fmt(metadata.minimum_primary_p_value)],
  ];
  el("summary-grid").innerHTML = cards
    .map(([label, value]) => `<article class="metric-card"><span>${label}</span><strong>${value}</strong></article>`)
    .join("");
}

function searchableText(test) {
  return [
    test.title,
    test.description,
    test.hypothesis,
    test.interpretation,
    test.family,
    test.model,
    test.verdict,
    test.novelty,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function filteredTests() {
  const q = state.filters.q.trim().toLowerCase();
  let tests = state.tests.filter((test) => {
    if (state.filters.model !== "All" && test.model !== state.filters.model) return false;
    if (state.filters.verdict !== "All" && test.verdict !== state.filters.verdict) return false;
    if (state.filters.novelty !== "All" && test.novelty !== state.filters.novelty) return false;
    if (state.filters.family !== "All" && test.family !== state.filters.family) return false;
    if (q && !searchableText(test).includes(q)) return false;
    return true;
  });

  tests.sort((a, b) => {
    if (state.filters.sort === "number") return a.model.localeCompare(b.model) || a.test_number - b.test_number || a.title.localeCompare(b.title);
    if (state.filters.sort === "sigma") return (Math.abs(b.metrics.sigma || 0) - Math.abs(a.metrics.sigma || 0)) || a.title.localeCompare(b.title);
    if (state.filters.sort === "title") return a.title.localeCompare(b.title);
    const ap = a.metrics.primary_p_value ?? 999;
    const bp = b.metrics.primary_p_value ?? 999;
    return ap - bp || a.title.localeCompare(b.title);
  });

  return tests;
}

function renderTests() {
  const tests = filteredTests();
  el("result-count").textContent = `${tests.length} shown`;
  el("test-list").innerHTML = tests
    .map((test) => {
      const m = test.metrics || {};
      const thumb = test.assets && test.assets.figure_png
        ? `<div class="test-thumb"><img src="${test.assets.figure_png}" alt=""></div>`
        : `<div class="test-thumb" aria-hidden="true"></div>`;
      return `<article class="test-row">
        ${thumb}
        <div class="test-title">
          <a href="${test.page}">${test.title}</a>
          <p>${test.description || "No description recorded."}</p>
        </div>
        <div><span class="cell-label">Verdict</span><span class="label ${labelClass(test.verdict)}">${test.verdict}</span></div>
        <div><span class="cell-label">Model</span>${test.model || "Unknown"}</div>
        <div><span class="cell-label">p-value</span><span class="number">${fmt(m.primary_p_value)}</span></div>
        <div><span class="cell-label">Sigma</span><span class="number">${fmt(m.sigma)}</span></div>
        <div><span class="cell-label">Novelty</span>${test.novelty || "Unknown"}</div>
        <div><span class="cell-label">Family</span>${test.family || "other"}</div>
      </article>`;
    })
    .join("");
}

function bindControls() {
  const search = el("search");
  search.addEventListener("input", (event) => {
    state.filters.q = event.target.value;
    renderTests();
  });

  [
    ["model-filter", "model"],
    ["verdict-filter", "verdict"],
    ["novelty-filter", "novelty"],
    ["family-filter", "family"],
    ["sort-order", "sort"],
  ].forEach(([id, key]) => {
    el(id).addEventListener("change", (event) => {
      state.filters[key] = event.target.value;
      renderTests();
    });
  });

  el("reset").addEventListener("click", () => {
    state.filters = { q: "", model: "All", verdict: "All", novelty: "All", family: "All", sort: "pvalue" };
    search.value = "";
    el("model-filter").value = "All";
    el("verdict-filter").value = "All";
    el("novelty-filter").value = "All";
    el("family-filter").value = "All";
    el("sort-order").value = "pvalue";
    renderTests();
  });
}

fetch("data/tests.json")
  .then((response) => response.json())
  .then((catalogue) => {
    state.tests = catalogue.tests;
    renderSummary(catalogue.metadata);
    fillSelect(el("model-filter"), uniq(state.tests.map((test) => test.model)));
    fillSelect(el("verdict-filter"), uniq(state.tests.map((test) => test.verdict)));
    fillSelect(el("novelty-filter"), uniq(state.tests.map((test) => test.novelty)));
    fillSelect(el("family-filter"), uniq(state.tests.map((test) => test.family)));
    bindControls();
    renderTests();
  })
  .catch((error) => {
    el("test-list").innerHTML = `<article class="test-row"><div class="test-title"><strong>Could not load catalogue data.</strong><p>${error}</p></div></article>`;
  });
""",
        encoding="utf-8",
    )


def status_label(verdict: str) -> str:
    klass = re.sub(r"[^a-z]+", "-", verdict.lower()).strip("-") or "unknown"
    return f'<span class="label {klass}">{html.escape(verdict)}</span>'


def metric_rows(test: dict[str, Any]) -> str:
    metrics = test["metrics"]
    rows = [
        ("Model", test["model"]),
        ("Run path", test["source_path"]),
        ("Primary p-value", fmt_metric(metrics.get("primary_p_value"))),
        ("One-tailed p-value", fmt_metric(metrics.get("p_value_one_tailed"))),
        ("Two-tailed p-value", fmt_metric(metrics.get("p_value_two_tailed"))),
        ("Sigma", fmt_metric(metrics.get("sigma"))),
        ("Planck statistic", fmt_metric(metrics.get("planck_stat"))),
        ("Simulation mean", fmt_metric(metrics.get("mean_sim_stat"))),
        ("Simulation std.", fmt_metric(metrics.get("std_sim_stat"))),
        ("Simulation median", fmt_metric(metrics.get("median_sim_stat"))),
        ("Simulations", fmt_metric(metrics.get("n_sims"))),
        ("Valid simulations", fmt_metric(metrics.get("n_valid_simulations"))),
        ("Tail", test["tail"]),
        ("Family", test["family"]),
        ("Novelty", test["novelty"]),
        ("Meets hypothesis", test["meets_hypothesis"] or "Not recorded"),
    ]
    return "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value or 'n/a'))}</td></tr>"
        for label, value in rows
    )


def asset_links(test: dict[str, Any], prefix: str = "../../") -> str:
    labels = {
        "figure_png": "Figure PNG",
        "figure_pdf": "Figure PDF",
        "updated_results_json": "Updated JSON",
        "result_summary_json": "Result summary JSON",
        "planck_statistic_npy": "Planck NPY",
        "simulation_statistics_npy": "Simulations NPY",
    }
    links = []
    for key, label in labels.items():
        path = test["assets"].get(key)
        if path:
            links.append(f'<a href="{prefix}{html.escape(path)}">{html.escape(label)}</a>')
    return "\n".join(links)


def write_test_page(test: dict[str, Any], metadata: dict[str, Any]) -> None:
    out_dir = SITE_DIR / test["page"].strip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    root_prefix = "../" * len(Path(test["page"].strip("/")).parts)
    tail_text = "two-tailed" if test["tail"] == "two-tailed" else f"{test['tail']} tail"
    figure = test["assets"].get("figure_png")
    figure_html = (
        f'<img src="{root_prefix}{html.escape(figure)}" alt="Statistic distribution for {html_text(test["title"])}">'
        if figure
        else '<p class="muted">No figure recorded.</p>'
    )
    code = html.escape(test.get("analysis_code") or "# Analysis code not recorded.")
    out_dir.joinpath("index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(test["title"])} · Test Catalogue</title>
  <link rel="stylesheet" href="{root_prefix}styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <p class="eyebrow">Test {test["test_number"]:02d} · {html.escape(test["model"])}</p>
      <h1>{html_text(test["title"])}</h1>
    </div>
    <nav class="toplinks" aria-label="Page navigation">
      <a href="{root_prefix}catalogue.html">← Catalogue</a>
      <a href="{root_prefix}">Paper</a>
      <a href="{root_prefix}data/tests.json">Registry JSON</a>
    </nav>
  </header>

  <main>
    <section class="notice">
      <strong>Provenance:</strong> {html.escape(metadata["curation_note"])}
    </section>

    <section class="detail-header">
      <div class="status-line">
        {status_label(test["verdict"])}
        <span class="muted">{html.escape(test["model"])} · {html.escape(test["novelty"])} · {html.escape(test["family"])} · {html.escape(tail_text)}</span>
      </div>
    </section>

    <div class="detail-layout">
      <div>
        <section class="panel">
          <h2>Hypothesis</h2>
          {paragraph(test["hypothesis"])}
        </section>

        <section class="panel">
          <h2>Method</h2>
          {paragraph(test["description"])}
        </section>

        <section class="panel">
          <h2>Result Summary</h2>
          {paragraph(test["results_text"])}
          {paragraph(test["interpretation"])}
        </section>

        <section class="panel">
          <h2>Agent Justification</h2>
          {paragraph(test["justification"])}
        </section>

        <section class="panel">
          <h2>Comparison To Literature</h2>
          {paragraph(test["literature"])}
        </section>

        <section class="panel">
          <h2>Analysis Code</h2>
          <details>
            <summary>Show generated Python statistic</summary>
            <pre><code>{code}</code></pre>
          </details>
        </section>
      </div>

      <aside>
        <section class="panel figure-panel">
          <h2>Distribution</h2>
          {figure_html}
        </section>

        <section class="panel">
          <h2>Metrics</h2>
          <table class="stats-table">
            <tbody>
              {metric_rows(test)}
            </tbody>
          </table>
        </section>

        <section class="panel">
          <h2>Artifacts</h2>
          <div class="asset-list">
            {asset_links(test, root_prefix)}
          </div>
        </section>
      </aside>
    </div>
  </main>

  <footer>
    <p>Catalogue {html.escape(metadata["version"])} generated from {len(metadata.get("model_counts") or {})} model(s).</p>
  </footer>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_test_pages(tests: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    for test in tests:
        write_test_page(test, metadata)


def write_readmes(metadata: dict[str, Any]) -> None:
    root_readme = ROOT / "README.md"
    model_lines = "\n".join(
        f"- `{source['label']}` ({source['kind']}): `{source['path']}`"
        for source in metadata.get("sources", [])
    )
    root_readme.write_text(
        f"""# AI Agents for Cosmological Anomaly Detection Test Catalogue

This workspace contains a frozen static catalogue for agent-generated CMB
anomaly tests.

The catalogue preserves agent-generated hypotheses, code, results, and
summaries. Literature comparison text is included as provenance and should be
checked before manuscript use.

## Models

{model_lines}

## Local preview

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Regenerate

```bash
python3 scripts/build_test_catalogue.py
```

The default builder reads `runs/cmb_test_run_mimo` and labels it `Mimo V2.5 Pro`.
To combine multiple model runs, pass each input with a model label:

```bash
python3 scripts/build_test_catalogue.py \\
  --run-dir 'runs/cmb_test_run_mimo::Mimo V2.5 Pro' \\
  --run-dir 'runs/another_model_run::Other Model'
```

Single JSON exports are also supported:

```bash
python3 scripts/build_test_catalogue.py \\
  --run-dir 'runs/cmb_test_run_mimo::Mimo V2.5 Pro' \\
  --json 'runs/model_b_results.json::Other Model'
```

The builder copies each test's figures and raw artifacts into `docs/`, and writes:

- `docs/index.html`: browsable static catalogue.
- `docs/tests/<test-id>/`: permanent detail page for each test.
- `docs/data/tests.json`: machine-readable registry.
- `docs/data/tests.csv`: compact table for audit and downstream analysis.
- `docs/data/raw/`: sanitized per-test JSON records preserving generated fields.
- `docs/data/statistics/`: Planck and simulation statistic arrays.

## Publishing

The `docs/` directory is ready for GitHub Pages, Netlify, Cloudflare Pages, or
any static file host. For GitHub Pages, publish the repository from the `docs/`
folder on the default branch.

## Paper citation snippet

```tex
The full catalogue of agent-proposed tests, including generated hypotheses,
analysis code, numerical results, and diagnostic plots, is provided as an
online resource in the frozen Test Catalogue {metadata["version"]}.
```
""",
        encoding="utf-8",
    )

    (SITE_DIR / "README.md").write_text(
        f"""# Static Catalogue

This directory is the publishable website for Test Catalogue {metadata["version"]}.

- `index.html` is the catalogue entry point.
- `data/tests.json` is the machine-readable registry.
- `tests/` contains stable per-test pages.

Generated at `{metadata["generated_at_utc"]}` from {len(metadata.get("model_counts") or {})} model(s).
""",
        encoding="utf-8",
    )

    (SITE_DIR / "paper_citation_snippet.tex").write_text(
        f"""The full catalogue of agent-proposed tests, including generated hypotheses,
analysis code, numerical results, and diagnostic plots, is provided as an
online resource in the frozen Test Catalogue {metadata["version"]}.
""",
        encoding="utf-8",
    )


def main() -> None:
    global SITE_DIR
    args = parse_args()
    SITE_DIR = Path(args.out).expanduser()
    if not SITE_DIR.is_absolute():
        SITE_DIR = ROOT / SITE_DIR

    sources = build_sources(args)
    prepare_site_dir()
    visual_assets = copy_cmb_assets()
    tests = load_tests(sources)
    if not tests:
        raise SystemExit("No tests found in the selected model inputs")
    metadata = build_metadata(tests, sources, args.version, visual_assets)
    write_registry(tests, metadata)
    write_index(metadata)
    write_styles()
    write_app_js()
    write_test_pages(tests, metadata)
    write_readmes(metadata)

    print(f"Built {metadata['n_tests']} tests into {SITE_DIR}")
    print(f"Registry: {SITE_DIR / 'data' / 'tests.json'}")
    print(f"Catalogue: {SITE_DIR / 'catalogue.html'}")


if __name__ == "__main__":
    main()
