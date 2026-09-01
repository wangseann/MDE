'''Tests for the optional Torch candidate cross-map backend.

The CPU pyEDM path remains the numerical reference.  Tests that exercise the
Torch kernel run it on CPU, so they require no GPU and are suitable for hosted
CI.  Resolver tests use dependency injection and therefore run even when the
optional ``torch`` package is absent.
'''

from importlib import import_module

import numpy as np
from pandas import DataFrame
import pytest
from pyEDM import ComputeError, Simplex

from dimx.MDE import MDE


torchCrossMap = import_module( 'dimx.TorchCrossMap' )


#----------------------------------------------------------------------------
def _Frame( N = 96 ):
    '''Finite deterministic data with no nearest-neighbor distance ties.'''
    rng    = np.random.RandomState( 124 )
    time   = np.arange( N, dtype = float )
    target = np.sin( 0.17 * time ) + 0.35 * np.cos( 0.071 * time )
    target = target + 0.04 * rng.normal( size = N )

    return DataFrame( {
        'target'      : target,
        'driver_a'    : np.roll( target, -1 ) +
                        0.02 * rng.normal( size = N ),
        'driver_bad'  : rng.normal( size = N ),
        'driver_weak' : 0.35 * np.roll( target, -1 ) +
                        0.65 * rng.normal( size = N ),
    } )


#----------------------------------------------------------------------------
def _Args( **overrides ):
    args = { 'target'          : 'target',
             'lib'             : [1, 40],
             'pred'            : [50, 80],
             'Tp'              : 1,
             'exclusionRadius' : 0 }
    args.update( overrides )
    return args


#----------------------------------------------------------------------------
def _CPUFactory( expectedDF, expectedArgs, sentinel, calls ):
    def factory( numericDF, argsD, **kwargs ):
        assert numericDF is expectedDF
        assert argsD is expectedArgs
        calls.append( kwargs )
        return sentinel
    return factory


#----------------------------------------------------------------------------
def _RequireTorch():
    try :
        import_module( 'torch' )
    except Exception as exc :
        pytest.skip( f'optional Torch dependency unavailable: {exc}' )


#----------------------------------------------------------------------------
def test_ranges_are_one_offset_inclusive_and_shift_targets_by_Tp():
    '''The adapter maps pyEDM ranges to predictor and target row arrays.'''
    frame = DataFrame( {
        'target' : np.arange( 20, dtype = float ),
        'driver' : np.arange( 100, 120, dtype = float ),
    } )
    args = _Args( lib = [2, 7], pred = [11, 16], Tp = 2 )

    crossMap = torchCrossMap.TorchCrossMap(
                   frame, args, deviceName = 'cpu' )

    # lib rows 2..5 predict target rows 4..7; the last Tp library rows cannot
    # supply an in-range target. Prediction rows retain their full range.
    np.testing.assert_array_equal( crossMap.xlib[:, 1],
                                   np.arange( 101, 105, dtype = np.float32 ) )
    np.testing.assert_array_equal( crossMap.ylib,
                                   np.arange( 3, 7, dtype = np.float32 ) )
    np.testing.assert_array_equal( crossMap.xpred[:, 1],
                                   np.arange( 110, 116, dtype = np.float32 ) )
    np.testing.assert_array_equal( crossMap.ypred,
                                   np.arange( 12, 18, dtype = np.float32 ) )


#----------------------------------------------------------------------------
def test_auto_uses_injected_CPU_factory_when_Torch_unavailable( monkeypatch ):
    frame    = _Frame()
    args     = _Args()
    sentinel = object()
    calls    = []
    messages = []

    monkeypatch.setattr(
        torchCrossMap, '_TorchStatus',
        lambda device: (False, 'optional dependency torch is not installed') )

    resolved = torchCrossMap.ResolveCrossMap(
                   frame, args, backend = 'auto', torchDevice = 'cuda',
                   cpuFactory = _CPUFactory( frame, args, sentinel, calls ),
                   logMsg = messages.append )

    assert resolved is sentinel
    assert len( calls ) == 1
    assert 'using CPU' in messages[0]


#----------------------------------------------------------------------------
def test_auto_uses_injected_CPU_factory_for_unsupported_settings( monkeypatch ):
    frame    = _Frame()
    # These windows overlap, so row-wise exclusion would be required.  The
    # first Torch kernel deliberately does not approximate that EDM setting.
    args     = _Args( lib = [1, 40], pred = [35, 70] )
    sentinel = object()
    calls    = []
    messages = []

    monkeypatch.setattr( torchCrossMap, '_TorchStatus',
                         lambda device: (True, None) )

    resolved = torchCrossMap.ResolveCrossMap(
                   frame, args, backend = 'auto', torchDevice = 'cpu',
                   cpuFactory = _CPUFactory( frame, args, sentinel, calls ),
                   logMsg = messages.append )

    assert resolved is sentinel
    assert len( calls ) == 1
    assert 'unsupported' in messages[0]
    assert 'disjoint lib/pred' in messages[0]


#----------------------------------------------------------------------------
def test_auto_uses_CPU_for_nonfinite_data( monkeypatch ):
    frame = _Frame()
    frame.loc[ 5, 'driver_a' ] = np.nan
    args     = _Args()
    sentinel = object()
    calls    = []

    monkeypatch.setattr( torchCrossMap, '_TorchStatus',
                         lambda device: (True, None) )

    resolved = torchCrossMap.ResolveCrossMap(
                   frame, args, backend = 'auto', torchDevice = 'cpu',
                   cpuFactory = _CPUFactory( frame, args, sentinel, calls ) )

    assert resolved is sentinel
    assert len( calls ) == 1


#----------------------------------------------------------------------------
def test_removed_nonfinite_column_is_not_part_of_accelerator_frame():
    frame = _Frame()
    frame['ignored'] = np.nan
    args = _Args( candidateColumns = ['driver_a', 'driver_bad'] )

    crossMap = torchCrossMap.TorchCrossMap(
                   frame, args, deviceName = 'cpu' )

    assert crossMap.columns == ['target', 'driver_a', 'driver_bad']
    assert np.isfinite( crossMap.xlib ).all()
    assert np.isfinite( crossMap.xpred ).all()


#----------------------------------------------------------------------------
def test_explicit_Torch_reports_unavailable_dependency( monkeypatch ):
    monkeypatch.setattr(
        torchCrossMap, '_TorchStatus',
        lambda device: (False, 'optional dependency torch is not installed') )

    with pytest.raises( RuntimeError,
                        match = 'requested, but optional dependency torch' ) :
        torchCrossMap.ResolveCrossMap( _Frame(), _Args(),
                                       backend = 'torch',
                                       torchDevice = 'cuda' )


#----------------------------------------------------------------------------
def test_explicit_Torch_reports_unsupported_EDM_settings( monkeypatch ):
    monkeypatch.setattr( torchCrossMap, '_TorchStatus',
                         lambda device: (True, None) )

    with pytest.raises( RuntimeError, match = 'requires disjoint lib/pred' ) :
        torchCrossMap.ResolveCrossMap(
            _Frame(), _Args( lib = [1, 40], pred = [35, 70] ),
            backend = 'torch', torchDevice = 'cpu' )


#----------------------------------------------------------------------------
@pytest.mark.parametrize( 'Tp, lib', [
    pytest.param(  0, [1, 40], id = 'contemporaneous' ),
    pytest.param(  1, [1, 40], id = 'forecast' ),
    pytest.param( -1, [2, 40], id = 'negative-Tp' ),
] )
def test_Torch_CPU_cross_map_rho_matches_pyEDM_Simplex( Tp, lib ):
    '''Torch and pyEDM agree for one- and two-coordinate manifolds.'''
    _RequireTorch()
    frame = _Frame()
    args  = _Args( Tp = Tp, lib = lib )
    crossMap = torchCrossMap.TorchCrossMap(
                   frame, args, deviceName = 'cpu',
                   batchCandidates = 2, predChunk = 7 )

    candidateGroups = [
        [ ['driver_a'], ['driver_bad'], ['driver_weak'] ],
        [ ['driver_bad',  'driver_a'],
          ['driver_weak', 'driver_a'] ],
    ]

    for dimension, candidates in enumerate( candidateGroups, start = 1 ) :
        observed = crossMap.CrossMap( candidates, dimension = dimension )

        for columns in candidates :
            simplex = Simplex( dataFrame       = frame,
                               columns         = columns,
                               target          = args['target'],
                               lib             = args['lib'],
                               pred            = args['pred'],
                               E               = 0,
                               embedded        = True,
                               exclusionRadius = args['exclusionRadius'],
                               Tp              = args['Tp'],
                               noTime          = True )
            expected = ComputeError( simplex['Observations'],
                                     simplex['Predictions'] )['rho']
            key = f'{",".join(columns)}:{args["target"]}'
            rho, returnedColumns = observed[ key ]

            assert returnedColumns == columns
            assert rho == pytest.approx( expected, abs = 2e-6 )


#----------------------------------------------------------------------------
def test_MDE_Torch_CPU_matches_CPU_selected_columns_and_rhos( tmp_path ):
    '''The optional backend preserves the user-visible greedy MDE result.'''
    _RequireTorch()
    frame = _Frame()
    common = dict( target          = 'target',
                   removeColumns  = ['target'],
                   noTime         = True,
                   lib            = [1, 40],
                   pred           = [50, 80],
                   Tp             = 1,
                   D              = 2,
                   noCCM          = True,
                   crossMapRhoMin = -1,
                   crossMapCores  = 1,
                   mpMethod       = 'spawn',
                   sharedMem      = 0,
                   consoleOut     = False,
                   outDir         = str( tmp_path ) )

    cpu = MDE( frame.copy(), crossMapBackend = 'cpu', **common )
    cpu.Run()
    accelerated = MDE( frame.copy(), crossMapBackend = 'torch',
                       torchDevice = 'cpu',
                       torchBatchCandidates = 2,
                       torchPredChunk = 7, **common )
    accelerated.Run()

    assert accelerated.MDEOut['variables'].tolist() == \
           cpu.MDEOut['variables'].tolist()
    np.testing.assert_allclose( accelerated.MDEOut['rho'],
                                cpu.MDEOut['rho'],
                                rtol = 0, atol = 2e-6 )
