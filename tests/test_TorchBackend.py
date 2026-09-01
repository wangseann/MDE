'''Focused tests for the optional, array-level Torch MDE backend.'''

import ast
import builtins
from hashlib import sha256
import importlib.util
import json
import sys
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path( __file__ ).parents[1] / 'dimx' / 'TorchBackend.py'
TORCH_AVAILABLE = importlib.util.find_spec( 'torch' ) is not None

# Load the standalone module exactly as the existing driving-MDE consumer
# loads its backend script.  This keeps these focused tests independent of
# unrelated imports performed by dimx.__init__.
_SPECIFICATION = importlib.util.spec_from_file_location(
    '_torch_backend_under_test', MODULE_PATH )
_MODULE = importlib.util.module_from_spec( _SPECIFICATION )
_SPECIFICATION.loader.exec_module( _MODULE )
candidate_sweep_torch = _MODULE.candidate_sweep_torch
candidate_sweep_cpu_reference = _MODULE.candidate_sweep_cpu_reference
compare_rows = _MODULE.compare_rows
eval_cols = _MODULE.eval_cols
simplex_1d_predict = _MODULE.simplex_1d_predict
simplex_predict = _MODULE.simplex_predict


def _SyntheticData():
    rng = np.random.default_rng( 371 )
    xlib = rng.normal( size = (48, 6) ).astype( np.float32 )
    xpred = rng.normal( size = (23, 6) ).astype( np.float32 )
    ylib = ( 1.7 * xlib[:, 2] - 0.4 * xlib[:, 0] +
             0.03 * rng.normal( size = len( xlib ) ) ).astype( np.float32 )
    ypred = ( 1.7 * xpred[:, 2] - 0.4 * xpred[:, 0] +
              0.03 * rng.normal( size = len( xpred ) ) ).astype( np.float32 )
    return xlib, ylib, xpred, ypred


@pytest.mark.skipif( sys.version_info[:2] != (3, 11),
                     reason = 'AST schema fingerprint is pinned to Python 3.11' )
def test_deployed_numeric_function_AST_fingerprint():
    '''Guard the exact deployed function bodies against incidental changes.'''
    names = ( 'rho', 'simplex_predict', 'simplex_1d_predict', 'eval_cols',
              'candidate_sweep_cpu_reference', 'candidate_sweep_torch',
              'compare_rows' )
    tree = ast.parse( MODULE_PATH.read_text() )
    functions = { node.name : ast.dump( node, include_attributes = False )
                  for node in tree.body
                  if isinstance( node, ast.FunctionDef ) }
    payload = '\n'.join( functions[name] for name in names ).encode()

    # Established by comparing both modules under Python 3.11 against
    # prototype SHA-256 eb1e0f82baf0... before this extraction was committed.
    assert sha256( payload ).hexdigest() == \
           '3de7fd2abc7719f5e93241ad12270a9a6d9acd6bdc6fd93b224aec87f9f4d787'


def test_direct_file_import_does_not_import_torch( monkeypatch ):
    '''CPU-only users can load this module without Torch being installed.'''
    realImport = builtins.__import__

    def GuardedImport( name, *args, **kwargs ):
        if name == 'torch' or name.startswith( 'torch.' ) :
            raise AssertionError( 'Torch was imported while loading the module' )
        return realImport( name, *args, **kwargs )

    monkeypatch.setattr( builtins, '__import__', GuardedImport )
    specification = importlib.util.spec_from_file_location(
        '_direct_torch_backend_import', MODULE_PATH )
    module = importlib.util.module_from_spec( specification )
    specification.loader.exec_module( module )

    assert callable( module.simplex_1d_predict )
    assert callable( module.simplex_predict )
    assert callable( module.eval_cols )
    assert callable( module.candidate_sweep_cpu_reference )
    assert callable( module.candidate_sweep_torch )
    assert callable( module.compare_rows )
    assert 'torch' not in module.__dict__
    assert module.__prototype_sha256__ == \
           'eb1e0f82baf0474fc4794d94898e9948db174c8acd8ca9c7cbcabc8bc7df63c0'

    assert list( signature( module.simplex_1d_predict ).parameters ) == \
           ['xl', 'yl', 'xq']
    assert list( signature( module.simplex_predict ).parameters ) == \
           ['xlib', 'ylib', 'xq', 'fixed_neighbors']
    sweepParameters = signature(
        module.candidate_sweep_torch ).parameters
    assert list( sweepParameters ) == [
        'xlib', 'ylib', 'xpred', 'ypred', 'selected', 'candidates',
        'batch_candidates', 'pred_chunk', 'device_name', 'fixed_neighbors' ]
    assert all( sweepParameters[name].kind is Parameter.KEYWORD_ONLY
                for name in list( sweepParameters )[4:] )
    assert sweepParameters['fixed_neighbors'].default is None


def test_requested_backend_has_actionable_missing_torch_error( monkeypatch ):
    xlib, ylib, xpred, ypred = _SyntheticData()
    monkeypatch.setitem( sys.modules, 'torch', None )

    with pytest.raises( RuntimeError, match = 'torch is not importable' ) :
        candidate_sweep_torch(
            xlib, ylib, xpred, ypred,
            selected = [], candidates = np.array( [0] ),
            batch_candidates = 1, pred_chunk = 8, device_name = 'cpu' )


def test_simplex_prediction_helpers_preserve_deployed_shapes_and_values():
    xlib = np.array( [[0.0], [1.0], [2.0], [4.0]] )
    ylib = np.array( [0.0, 1.0, 4.0, 16.0] )
    xquery = np.array( [[0.2], [1.8], [3.7]] )

    oneDimensional = simplex_1d_predict( xlib[:, 0], ylib, xquery[:, 0] )
    general = simplex_predict( xlib, ylib, xquery )

    assert oneDimensional.shape == (3,)
    assert oneDimensional.dtype == np.float32
    np.testing.assert_allclose( oneDimensional, general, rtol = 1e-6,
                                atol = 1e-6 )

    exact = simplex_1d_predict( xlib[:, 0], ylib,
                                np.array( [0.0, 2.0, 4.0] ) )
    # Preserve the deployed helper's equal weighting when the nearest
    # distance is exactly zero. This is a compatibility fixture, not a new
    # interpretation of the Simplex rule.
    np.testing.assert_allclose( exact, np.array( [0.5, 2.5, 10.0] ),
                                rtol = 0, atol = 1e-6 )


@pytest.mark.skipif( not TORCH_AVAILABLE, reason = 'optional Torch not installed' )
def test_cpu_torch_sweep_matches_scalar_reference_and_is_json_safe():
    xlib, ylib, xpred, ypred = _SyntheticData()
    selected = [0]
    candidates = np.array( [5, 2, 3, 1], dtype = np.int64 )

    best, bestRho, rows, timings = candidate_sweep_torch(
        xlib, ylib, xpred, ypred, selected = selected,
        candidates = candidates, batch_candidates = 3, pred_chunk = 7,
        device_name = 'cpu' )

    expectedBest, expectedBestRho, referenceRows, _ = \
        candidate_sweep_cpu_reference(
            xlib, ylib, xpred, ypred, selected = selected,
            candidates = candidates )
    expected = [row['rho'] for row in referenceRows]
    comparison = compare_rows( referenceRows, rows )

    assert [row['candidate'] for row in rows] == candidates.tolist()
    np.testing.assert_allclose( [row['rho'] for row in rows], expected,
                                rtol = 2e-5, atol = 2e-5 )
    assert best == expectedBest
    assert bestRho == pytest.approx( expectedBestRho, rel = 2e-5,
                                    abs = 2e-5 )
    assert comparison['top1_match']
    assert comparison['max_abs_rho_diff'] <= 2e-5
    assert set( timings ) == {
        'transfer_seconds', 'sweep_seconds', 'total_seconds' }
    assert all( isinstance( value, float ) and value >= 0
                for value in timings.values() )
    json.dumps( { 'best' : best, 'rho' : bestRho, 'rows' : rows,
                  'timings' : timings }, allow_nan = False )


@pytest.mark.skipif( not TORCH_AVAILABLE, reason = 'optional Torch not installed' )
def test_cpu_torch_sweep_is_invariant_to_batch_and_prediction_chunks():
    xlib, ylib, xpred, ypred = _SyntheticData()
    candidates = np.array( [4, 1, 5, 2], dtype = np.int64 )
    common = { 'selected' : [0, 3], 'candidates' : candidates,
               'device_name' : 'cpu' }

    small = candidate_sweep_torch(
        xlib, ylib, xpred, ypred, batch_candidates = 1, pred_chunk = 1,
        **common )
    large = candidate_sweep_torch(
        xlib, ylib, xpred, ypred, batch_candidates = len( candidates ),
        pred_chunk = len( xpred ), **common )

    assert small[0] == large[0]
    assert [row['candidate'] for row in small[2]] == candidates.tolist()
    assert [row['candidate'] for row in large[2]] == candidates.tolist()
    np.testing.assert_allclose( [row['rho'] for row in small[2]],
                                [row['rho'] for row in large[2]],
                                rtol = 1e-7, atol = 1e-7 )


@pytest.mark.skipif( not TORCH_AVAILABLE, reason = 'optional Torch not installed' )
def test_all_undefined_correlations_preserve_deployed_negative_infinity():
    xlib, ylib, xpred, _ = _SyntheticData()
    ypred = np.ones( len( xpred ), dtype = np.float32 )

    best, bestRho, rows, _ = candidate_sweep_torch(
        xlib, ylib, xpred, ypred, selected = [],
        candidates = np.array( [0, 1] ), batch_candidates = 2,
        pred_chunk = 8, device_name = 'cpu' )

    assert best is None
    assert np.isneginf( bestRho )
    assert all( np.isnan( row['rho'] ) for row in rows )


@pytest.mark.skipif( not TORCH_AVAILABLE, reason = 'optional Torch not installed' )
def test_equal_candidate_scores_keep_input_order_winner():
    xlib, ylib, xpred, ypred = _SyntheticData()
    xlib[:, 1] = xlib[:, 0]
    xpred[:, 1] = xpred[:, 0]

    best, _, rows, _ = candidate_sweep_torch(
        xlib, ylib, xpred, ypred, selected = [],
        candidates = np.array( [1, 0] ), batch_candidates = 2,
        pred_chunk = 8, device_name = 'cpu' )

    assert [ row['candidate'] for row in rows ] == [1, 0]
    assert rows[0]['rho'] == rows[1]['rho']
    assert best == 1
