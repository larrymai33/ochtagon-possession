"""Fit/export the externally selected observation-gap ensemble separately."""

from __future__ import annotations

import torch  # Pinned Windows native-library import order.

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

from solution.ensemble_artifact import (
    ALLOWED_TRAINING_IDS, TRAINING_SEEDS, FrozenEnsembleArtifact,
    build_ensemble_manifest, ensemble_artifact_sha256,
    load_ensemble_artifact, save_ensemble_artifact,
)
from solution.final_tracking_ensemble import (
    SOURCE_PATHS, _write_new_submission, hgb_config, infer_tracking_sequence,
    source_hashes, target_input_hashes, tcn_config,
)
from solution.target_adapter import load_bar_ath_sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'data/cache/external_model/final_observation_ensemble_v1.pkl'
DEFAULT_REPORT = ROOT / 'reports/observation_robustness_final_v1/fit.json'
TRAINING_METHOD = 'observation-gap-paired-tcn-v1'
NORMALIZER_POLICY = 'original external training matches only'


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_digest(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('an explicit lowercase SHA-256 digest is required')


def write_new_json(path: Path, value: dict) -> str:
    """Publish complete JSON bytes atomically without replacing any existing path."""
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    raw = _canonical(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.observation-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return _digest(raw)


def inference_source_hashes() -> dict[str, str]:
    paths = set(SOURCE_PATHS) | {
        'solution/__init__.py', 'solution/final_observation_ensemble.py',
    }
    return {name: _digest((ROOT / name).read_bytes()) for name in sorted(paths)}


def validate_candidate_manifest(manifest: dict) -> None:
    if (not isinstance(manifest, dict)
            or manifest.get('training_method') != TRAINING_METHOD
            or manifest.get('target_labels_used') is not False
            or manifest.get('provided_video_used') is not False
            or manifest.get('normalizer_policy') != NORMALIZER_POLICY
            or manifest.get('view_probability') != .5
            or 'public_success_requirement' in manifest):
        raise ValueError('artifact is not the selected external observation candidate')
    if manifest.get('source_sha256') != source_hashes():
        raise ValueError('candidate core inference source differs')
    if manifest.get('inference_source_sha256') != inference_source_hashes():
        raise ValueError('candidate inference wrapper source differs')


def load_candidate_artifact(path: Path, expected_sha256: str):
    _require_digest(expected_sha256)
    artifact = load_ensemble_artifact(path, expected_sha256=expected_sha256)
    validate_candidate_manifest(artifact.manifest)
    return artifact


def validate_selection_result(contract: dict, result: dict, source: dict) -> dict:
    """Recompute the similarity decision from bound external per-match metrics."""
    from solution.observation_experiment_contract import similarity_gate

    if (result.get('schema') != 'observation-robustness-result-v1'
            or result.get('contract_payload_sha256') != _digest(_canonical(contract))
            or result.get('baseline_verified') is not True
            or result.get('target_data_or_public_score_accessed') is not False):
        raise ValueError('external result identity or baseline verification differs')
    rows = result.get('per_match')
    if (not isinstance(rows, list) or len(rows) != len(ALLOWED_TRAINING_IDS)
            or any(type(row.get('match_id')) is not int for row in rows)
            or tuple(row['match_id'] for row in rows) != ALLOWED_TRAINING_IDS):
        raise ValueError('selection result must contain the exact ordered external cohort')
    for row in rows:
        for name, expected in source['baseline_metrics_by_match'][str(row['match_id'])].items():
            actual = row['baseline'][name]
            if not np.isfinite(actual) or abs(actual - expected) > 1e-12:
                raise ValueError('selection baseline differs from accepted evidence')
    baseline = [row['baseline']['spell_f1'] for row in rows]
    candidate = [row['candidate']['spell_f1'] for row in rows]
    decision = similarity_gate(baseline, candidate, baseline_verified=True)
    recorded = result.get('similarity', {})
    if (recorded.get('similar') is not True or recorded.get('baseline_verified') is not True
            or decision != recorded or decision['similar'] is not True):
        raise ValueError('external candidate did not pass the frozen similarity criterion')
    for view in ('baseline', 'candidate', 'baseline_masked', 'candidate_masked'):
        scores = np.asarray([row[view]['spell_f1'] for row in rows])
        recorded_mean = result['mean_spell_f1'][view]
        if (scores.dtype.kind not in 'iuf' or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1)) or not np.isfinite(recorded_mean)
                or abs(float(scores.mean()) - recorded_mean) > 1e-12):
            raise ValueError('selection aggregate differs from per-match metrics')
    return decision


def load_selection_evidence(contract_path: Path, expected_contract_sha256: str,
                            expected_result_sha256: str):
    """Verify source/input bindings, result and every sealed prediction before fit."""
    from solution.observation_experiment_contract import load_contract, read_bound
    from solution.evaluate_observation_robustness import _read_predictions
    from solution.evaluate_short_excursion_reranker import load_current_source

    _require_digest(expected_contract_sha256)
    _require_digest(expected_result_sha256)
    contract = load_contract(Path(contract_path), expected_contract_sha256)
    result_path = Path(contract['output_dir']) / 'result.json'
    result = json.loads(read_bound(result_path, expected_result_sha256))
    source = load_current_source(verify_files=False)
    validate_selection_result(contract, result, source)
    _require_digest(result['freeze_sha256'])
    predictions = _read_predictions(contract, result['freeze_sha256'])
    if tuple(sorted(predictions)) != ALLOWED_TRAINING_IDS:
        raise ValueError('prediction freeze cohort differs')
    del predictions
    claim_path = Path(contract['output_dir']) / 'score_claim.json'
    claim_raw = claim_path.read_bytes()
    if json.loads(claim_raw) != {
            'freeze_sha256': result['freeze_sha256'],
            'all_predictions_verified_before_label_access': True}:
        raise ValueError('external score claim does not bind the selected prediction freeze')
    evidence = {'contract_sha256': expected_contract_sha256,
                'contract_payload_sha256': _digest(_canonical(contract)),
                'result_sha256': expected_result_sha256, 'freeze_sha256': result['freeze_sha256'],
                'score_claim_sha256': _digest(claim_raw), 'similarity': result['similarity']}
    return contract, result, source, evidence


def fit_final_candidate(contract_path: Path, expected_contract_sha256: str,
                        expected_result_sha256: str) -> dict:
    """Fit the new family on all external training matches after its fixed gate passes."""
    model_path, report_path = Path(DEFAULT_MODEL), Path(DEFAULT_REPORT)
    manifest_path = model_path.with_suffix(model_path.suffix + '.manifest.json')
    attempt_path = report_path.with_name('fit-attempt.json')
    for path in (model_path, manifest_path, report_path, attempt_path):
        if os.path.lexists(path):
            raise FileExistsError(path)
    contract, result, source, evidence = load_selection_evidence(
        contract_path, expected_contract_sha256, expected_result_sha256)
    if not torch.cuda.is_available():
        raise RuntimeError('frozen final fitting requires CUDA')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8') != ':4096:8':
        raise ValueError('CUBLAS_WORKSPACE_CONFIG differs from frozen configuration')

    from solution.observation_experiment_contract import verify_contract, VIEW_CACHE_DIR, runtime_identity
    from solution.evaluate_observation_robustness import load_original, load_view, prepared_original
    from solution.train_external import _training_arrays, fit_models
    from solution.train_tcn_external import fit_fold_normalizer, prepare_cached_match
    from solution.train_observation_robustness import train_with_observation_views

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    before = inference_source_hashes()
    training_sources = {name: value for name, value in contract['bindings'].items() if name.endswith('.py')}
    if training_sources.get('solution/final_observation_ensemble.py') != before['solution/final_observation_ensemble.py']:
        raise ValueError('final fitting wrapper was not included in the frozen experiment')
    attempt_hash = write_new_json(attempt_path, {'schema': 'observation-final-fit-attempt-v1',
                                                'selection': evidence})
    start = time.monotonic()
    caches = tuple(load_original(source, mid, training=True) for mid in ALLOWED_TRAINING_IDS)
    originals = tuple(prepared_original(item) for item in caches)
    masked = tuple(prepare_cached_match(load_view(contract, item)) for item in caches)
    normalizer = fit_fold_normalizer(originals)
    with threadpool_limits(limits=4):
        values, labels, weights, names = _training_arrays(caches, stride=5)
        if tuple(names) != normalizer.feature_names:
            raise ValueError('HGB and TCN feature names differ')
        hgb = fit_models(values, labels, names, hgb_config(), sample_weights=weights)
        del values, labels, weights, caches
        histories, models = [], []
        for seed in TRAINING_SEEDS:
            print(f'fitting final observation TCN seed {seed}', flush=True)
            model, history = train_with_observation_views(
                originals, masked, normalizer, tcn_config(seed), 'cuda', view_probability=.5)
            models.append(model.cpu().eval())
            histories.append({'seed': seed, 'epochs': history})
            del model
            gc.collect()
            torch.cuda.empty_cache()
    verify_contract(contract)
    _, _, _, checked_evidence = load_selection_evidence(
        contract_path, expected_contract_sha256, expected_result_sha256)
    if checked_evidence != evidence or inference_source_hashes() != before:
        raise ValueError('selection evidence or inference source changed during fitting')
    if _digest(attempt_path.read_bytes()) != attempt_hash:
        raise ValueError('final fit attempt receipt changed during fitting')
    masked_hashes = {str(mid): contract['bindings'][
        (VIEW_CACHE_DIR / f'{mid}.npz').relative_to(ROOT).as_posix()] for mid in ALLOWED_TRAINING_IDS}
    manifest = build_ensemble_manifest(source['contact_cache_sha256'], extra={
        'training_method': TRAINING_METHOD, 'source_sha256': source_hashes(),
        'inference_source_sha256': before, 'training_source_sha256': training_sources,
        'masked_input_sha256': masked_hashes, 'external_selection': evidence,
        'mask_config': contract['spec']['mask_config'], 'mask_schedule': contract['spec']['mask_schedule'],
        'view_probability': .5, 'view_rng_seed': 'training seed + 2',
        'normalizer_policy': NORMALIZER_POLICY, 'hgb_config': asdict(hgb_config()),
        'hgb_sample_stride': 5, 'training_config': [asdict(tcn_config(seed)) for seed in TRAINING_SEEDS],
        'training_history': histories, 'runtime': runtime_identity(),
        'runtime_settings': contract['spec']['runtime_settings'],
        'target_labels_used': False, 'provided_video_used': False,
        'resampling_contract': 'adjacent-native-linear-per-run-10hz-v1',
        'elapsed_seconds': time.monotonic() - start,
    })
    validate_candidate_manifest(manifest)
    artifact = FrozenEnsembleArtifact(hgb, tuple(models), normalizer, manifest)
    save_ensemble_artifact(model_path, artifact)
    model_hash = ensemble_artifact_sha256(model_path)
    checked = load_candidate_artifact(model_path, model_hash)
    if checked.manifest != manifest:
        raise ValueError('saved candidate manifest differs after round trip')
    manifest_hash = write_new_json(manifest_path, {'artifact_sha256': model_hash, 'manifest': manifest})
    report = {'schema': 'observation-final-fit-v1', 'training_method': TRAINING_METHOD,
              'artifact': str(model_path), 'artifact_sha256': model_hash,
              'manifest': str(manifest_path), 'manifest_sha256': manifest_hash,
              'attempt_sha256': attempt_hash, 'external_selection': evidence,
              'training_ids': list(ALLOWED_TRAINING_IDS), 'roundtrip_verified': True,
              'target_inference_performed': False, 'elapsed_seconds': time.monotonic() - start}
    write_new_json(report_path, report)
    return report


def run_candidate_inference(model_path: Path, output_path: Path,
                            data_dir: str, expected_sha256: str) -> dict:
    """Explicit invocation only: validate the new artifact, then perform automatic inference."""
    output = Path(output_path)
    receipt = output.with_suffix(output.suffix + '.manifest.json')
    if os.path.lexists(output) or os.path.lexists(receipt):
        raise FileExistsError('candidate output or receipt already exists')
    artifact = load_candidate_artifact(model_path, expected_sha256)
    frozen_sources = inference_source_hashes()
    inputs = target_input_hashes(data_dir)
    sequence = load_bar_ath_sequence(data_dir)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        labels = infer_tracking_sequence(sequence, artifact)
        repeated = infer_tracking_sequence(sequence, artifact)
    if not np.array_equal(labels, repeated):
        raise ValueError('candidate inference did not reproduce exactly')
    if (inference_source_hashes() != frozen_sources or target_input_hashes(data_dir) != inputs
            or ensemble_artifact_sha256(model_path) != expected_sha256):
        raise ValueError('candidate source, model or tracking inputs changed during inference')
    output_hash = _write_new_submission(output, labels)
    record = {'schema': 'observation-inference-v1', 'training_method': TRAINING_METHOD,
              'artifact_sha256': expected_sha256, 'submission_sha256': output_hash,
              'inference_source_sha256': frozen_sources, 'input_sha256': inputs,
              'frames': len(labels), 'deterministic_label_reproduction': True,
              'target_labels_used': False, 'provided_video_used': False}
    write_new_json(receipt, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--contract', type=Path, required=True)
    fit.add_argument('--expected-contract-sha256', required=True)
    fit.add_argument('--expected-result-sha256', required=True)
    infer = commands.add_parser('infer')
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--expected-sha256', required=True)
    infer.add_argument('--data-dir', required=True)
    infer.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'fit':
        result = fit_final_candidate(args.contract, args.expected_contract_sha256, args.expected_result_sha256)
    else:
        result = run_candidate_inference(args.model, args.out, args.data_dir, args.expected_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
