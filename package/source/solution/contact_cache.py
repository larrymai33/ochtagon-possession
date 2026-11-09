"""Combined SkillCorner base/contact feature cache with strict provenance."""

from __future__ import annotations

import os

import numpy as np

from solution.contact_features import CONTACT_FEATURE_SCHEMA_VERSION, extract_contact_features
from solution.external_labels import FINAL_TEST_MATCH_IDS, SKILLCORNER_SOURCE_COMMIT
from solution.skillcorner_adapter import ADAPTER_VERSION, CachedMatch, load_skillcorner_match
from solution.tracking_features import FEATURE_SCHEMA_VERSION, extract_features


BASE_FEATURE_WIDTH = 55
CONTACT_FEATURE_WIDTH = 72
COMBINED_FEATURE_WIDTH = BASE_FEATURE_WIDTH + CONTACT_FEATURE_WIDTH
COMBINED_FEATURE_SCHEMA_VERSION = (
    FEATURE_SCHEMA_VERSION + "+" + CONTACT_FEATURE_SCHEMA_VERSION
)
SOURCE_ADAPTER_VERSION = ADAPTER_VERSION
COMBINED_CACHE_ADAPTER_VERSION = "skillcorner-contact-cache-v1"


def _development_match_id(match_id: int) -> int:
    value = int(match_id)
    if value in FINAL_TEST_MATCH_IDS:
        raise ValueError("match {} is quarantined and cannot enter development caches".format(
            value
        ))
    return value


def _directory_match_id(match_dir: str) -> int:
    name = os.path.basename(os.path.normpath(match_dir))
    if not name.isdigit():
        raise ValueError("SkillCorner match directory must have a numeric name")
    return _development_match_id(int(name))


def build_combined_feature_cache(match_dir: str, path: str) -> CachedMatch:
    """Build base features followed by provider-neutral contact features."""
    directory_match_id = _directory_match_id(match_dir)
    match = load_skillcorner_match(match_dir)
    if match.match_id != directory_match_id:
        raise ValueError("match metadata id {} differs from directory {}".format(
            match.match_id, directory_match_id
        ))
    base = extract_features(match.sequence)
    contact = extract_contact_features(match.sequence)
    if base.valid_runs != contact.valid_runs:
        raise ValueError("base/contact valid runs differ")
    if base.values.shape[1] != BASE_FEATURE_WIDTH:
        raise ValueError("base feature width is not {}".format(BASE_FEATURE_WIDTH))
    if contact.values.shape[1] != CONTACT_FEATURE_WIDTH:
        raise ValueError("contact feature width is not {}".format(CONTACT_FEATURE_WIDTH))
    names = base.feature_names + contact.feature_names
    if len(set(names)) != len(names):
        raise ValueError("base/contact feature names overlap")
    values = np.concatenate((base.values, contact.values), axis=1).astype(
        np.float32, copy=False
    )
    cached = CachedMatch(
        match_id=match.match_id,
        values=values,
        labels=match.labels,
        feature_names=names,
        valid_runs=base.valid_runs,
        source_commit=SKILLCORNER_SOURCE_COMMIT,
        schema_version=COMBINED_FEATURE_SCHEMA_VERSION,
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(
        path,
        match_id=np.array(cached.match_id, dtype=np.int64),
        values=cached.values,
        labels=cached.labels,
        feature_names=np.array(cached.feature_names, dtype="<U96"),
        valid_runs=np.array(cached.valid_runs, dtype=np.int64),
        source_commit=np.array(cached.source_commit),
        schema_version=np.array(cached.schema_version),
        base_schema_version=np.array(FEATURE_SCHEMA_VERSION),
        contact_schema_version=np.array(CONTACT_FEATURE_SCHEMA_VERSION),
        source_adapter_version=np.array(SOURCE_ADAPTER_VERSION),
        cache_adapter_version=np.array(COMBINED_CACHE_ADAPTER_VERSION),
    )
    return cached


def load_combined_feature_cache(
        path: str,
        expected_source_commit: str = SKILLCORNER_SOURCE_COMMIT,
        expected_schema: str = COMBINED_FEATURE_SCHEMA_VERSION,
        expected_base_schema: str = FEATURE_SCHEMA_VERSION,
        expected_contact_schema: str = CONTACT_FEATURE_SCHEMA_VERSION,
        expected_source_adapter: str = SOURCE_ADAPTER_VERSION,
        expected_cache_adapter: str = COMBINED_CACHE_ADAPTER_VERSION) -> CachedMatch:
    """Load a cache only when every source, schema, and adapter field matches."""
    with np.load(path, allow_pickle=False) as archive:
        if "match_id" not in archive.files:
            raise ValueError("combined cache manifest is missing match_id")
        match_id = _development_match_id(int(archive["match_id"].item()))
        required = {
            "source_commit", "schema_version", "base_schema_version",
            "contact_schema_version", "source_adapter_version",
            "cache_adapter_version", "match_id", "values", "labels",
            "feature_names", "valid_runs",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError("combined cache manifest is missing fields: {}".format(missing))
        source_commit = str(archive["source_commit"].item())
        schema_version = str(archive["schema_version"].item())
        base_schema = str(archive["base_schema_version"].item())
        contact_schema = str(archive["contact_schema_version"].item())
        source_adapter = str(archive["source_adapter_version"].item())
        cache_adapter = str(archive["cache_adapter_version"].item())
        if source_commit != expected_source_commit:
            raise ValueError("source commit mismatch: cache {} != expected {}".format(
                source_commit, expected_source_commit
            ))
        if schema_version != expected_schema:
            raise ValueError("composite schema mismatch: cache {} != expected {}".format(
                schema_version, expected_schema
            ))
        if base_schema != expected_base_schema:
            raise ValueError("base schema mismatch: cache {} != expected {}".format(
                base_schema, expected_base_schema
            ))
        if contact_schema != expected_contact_schema:
            raise ValueError("contact schema mismatch: cache {} != expected {}".format(
                contact_schema, expected_contact_schema
            ))
        if source_adapter != expected_source_adapter:
            raise ValueError("source adapter mismatch: cache {} != expected {}".format(
                source_adapter, expected_source_adapter
            ))
        if cache_adapter != expected_cache_adapter:
            raise ValueError("cache adapter mismatch: cache {} != expected {}".format(
                cache_adapter, expected_cache_adapter
            ))
        names = tuple(str(name) for name in archive["feature_names"].tolist())
        values = archive["values"].copy()
        if values.ndim != 2 or values.shape[1] != COMBINED_FEATURE_WIDTH:
            raise ValueError("combined feature width must be {}".format(
                COMBINED_FEATURE_WIDTH
            ))
        if len(names) != COMBINED_FEATURE_WIDTH or len(set(names)) != len(names):
            raise ValueError("combined feature names are invalid")
        runs = tuple(
            (int(start), int(stop))
            for start, stop in archive["valid_runs"].tolist()
        )
        return CachedMatch(
            match_id=match_id,
            values=values,
            labels=archive["labels"].copy(),
            feature_names=names,
            valid_runs=runs,
            source_commit=source_commit,
            schema_version=schema_version,
        )
