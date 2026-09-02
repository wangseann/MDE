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
def _SingleDriverFrame( N = 64 ):
    '''Causal synthetic process: driver at t predicts target at t + 1.'''
    rng    = np.random.RandomState( 20260902 )
    time   = np.arange( N, dtype = float )
    driver = np.sin( 0.19 * time ) + 0.2 * np.cos( 0.047 * time )
    target = np.empty( N, dtype = float )
    target[0]  = 0.0
    target[1:] = driver[:-1] + 0.02 * rng.normal( size = N - 1 )
    return DataFrame( { 'target' : target, 'driver' : driver } )


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
def _AssertMDEParity( cpu, accelerated ):
    '''Compare the user result and every retained candidate cross-map rho.'''
    assert accelerated.MDEOut['variables'].tolist() == \
           cpu.MDEOut['variables'].tolist()
    np.testing.assert_allclose( accelerated.MDEOut['rho'],
                                cpu.MDEOut['rho'],
                                rtol = 0, atol = 2e-6 )

    assert accelerated.rhoD.keys() == cpu.rhoD.keys()
    for dimension, cpuRho in cpu.rhoD.items() :
        acceleratedRho = accelerated.rhoD[ dimension ]
        assert acceleratedRho.index.tolist() == cpuRho.index.tolist()
        np.testing.assert_allclose( acceleratedRho['rho'], cpuRho['rho'],
                                    rtol = 0, atol = 2e-6 )


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

    _AssertMDEParity( cpu, accelerated )


#----------------------------------------------------------------------------
def test_MDE_Torch_CPU_matches_CPU_with_precomputed_CCM_slopes( monkeypatch ):
    '''Both sweep backends apply the same supplied CCM slopes and ranking.'''
    _RequireTorch()
    runModule = import_module( 'dimx.Run' )

    def UnexpectedLiveCCM( *args, **kwargs ):
        raise AssertionError( 'precomputed slopes must bypass live EDM calls' )

    monkeypatch.setattr( runModule, 'EmbedDimension', UnexpectedLiveCCM )
    monkeypatch.setattr( runModule, 'CCM', UnexpectedLiveCCM )

    frame  = _Frame()
    labels = frame.columns.tolist()
    slopeMatrix = DataFrame( 0.0, index = labels, columns = labels )
    slopeMatrix.loc['driver_a',    'target'] = 0.01
    slopeMatrix.loc['driver_weak', 'target'] = 0.4
    slopeMatrix.loc['driver_bad',  'target'] = 0.1
    slopeMatrixBaseline = slopeMatrix.copy( deep = True )

    common = dict( target          = 'target',
                   removeColumns  = ['target'],
                   noTime         = True,
                   lib            = [1, 40],
                   pred           = [50, 80],
                   Tp             = 1,
                   D              = 2,
                   ccmSlope       = 0.05,
                   crossMapRhoMin = -1,
                   crossMapCores  = 1,
                   mpMethod       = 'spawn',
                   sharedMem      = 0,
                   consoleOut     = False )

    cpuMatrix = slopeMatrix.copy( deep = True )
    acceleratedMatrix = slopeMatrix.copy( deep = True )
    cpu = MDE( frame.copy(), slopeMatrix = cpuMatrix,
               crossMapBackend = 'cpu', **common )
    cpu.Run()
    accelerated = MDE( frame.copy(), slopeMatrix = acceleratedMatrix,
                       crossMapBackend = 'torch', torchDevice = 'cpu',
                       torchBatchCandidates = 2, torchPredChunk = 7,
                       **common )
    accelerated.Run()

    _AssertMDEParity( cpu, accelerated )
    assert cpu.rhoD[1].index[0] == 'driver_a'
    assert 'driver_a' not in cpu.rhoD_CCM[1].index
    assert cpu.MDEOut['variables'].iloc[0] == cpu.rhoD_CCM[1].index[0]
    assert cpu.MDEOut['variables'].iloc[0] != 'driver_a'
    assert cpu.slopeMatrix.equals( slopeMatrixBaseline )
    assert accelerated.slopeMatrix.equals( slopeMatrixBaseline )
    assert cpu.EDim == accelerated.EDim == {}
    assert cpu._edimCache == accelerated._edimCache == {}
    assert cpu._ccmCache == accelerated._ccmCache == {}
    assert accelerated.rhoD_CCM.keys() == cpu.rhoD_CCM.keys() == {1, 2}

    for dimension, cpuCCM in cpu.rhoD_CCM.items() :
        acceleratedCCM = accelerated.rhoD_CCM[ dimension ]
        assert acceleratedCCM.index.tolist() == cpuCCM.index.tolist()
        np.testing.assert_allclose( acceleratedCCM['rho'], cpuCCM['rho'],
                                    rtol = 0, atol = 2e-6 )
        np.testing.assert_array_equal( acceleratedCCM['slope'],
                                       cpuCCM['slope'] )
        for candidate, slope in cpuCCM['slope'].items() :
            assert slope == slopeMatrix.loc[ candidate, 'target' ]


#----------------------------------------------------------------------------
def test_MDE_Torch_CPU_matches_CPU_with_internal_CCM( monkeypatch ):
    '''Both sweep backends obtain identical slopes from MDE's live pyEDM CCM.'''
    _RequireTorch()
    runModule = import_module( 'dimx.Run' )
    realEmbedDimension = runModule.EmbedDimension
    realCCM = runModule.CCM
    calls = { 'EmbedDimension' : 0, 'CCM' : 0 }

    def CountEmbedDimension( *args, **kwargs ):
        calls['EmbedDimension'] += 1
        return realEmbedDimension( *args, **kwargs )

    def CountCCM( *args, **kwargs ):
        calls['CCM'] += 1
        # Exercise the real seeded CCM algorithm without starting its optional
        # second process layer; scheduling is not part of backend parity.
        return realCCM( *args, parallel = False, **kwargs )

    monkeypatch.setattr( runModule, 'EmbedDimension', CountEmbedDimension )
    monkeypatch.setattr( runModule, 'CCM', CountCCM )

    frame = _SingleDriverFrame()
    common = dict( target          = 'target',
                   removeColumns  = ['target'],
                   noTime         = True,
                   lib            = [1, 24],
                   pred           = [35, 52],
                   Tp             = 1,
                   D              = 1,
                   E              = 0,
                   maxE           = 2,
                   embedDimRhoMin = 0.9,
                   libSizes       = [8, 12, 16],
                   sample         = 1,
                   ccmSeed        = 371,
                   ccmSlope       = 0.1,
                   crossMapRhoMin = 0.9,
                   crossMapCores  = 1,
                   mpMethod       = 'spawn',
                   sharedMem      = 0,
                   consoleOut     = False )

    cpu = MDE( frame.copy(), crossMapBackend = 'cpu', **common )
    cpu.Run()
    assert calls == { 'EmbedDimension' : 1, 'CCM' : 1 }
    accelerated = MDE( frame.copy(), crossMapBackend = 'torch',
                       torchDevice = 'cpu', torchBatchCandidates = 2,
                       torchPredChunk = 7, **common )
    accelerated.Run()

    _AssertMDEParity( cpu, accelerated )
    assert calls == { 'EmbedDimension' : 2, 'CCM' : 2 }
    assert cpu.MDEOut['variables'].tolist() == ['driver']
    assert cpu.MDEOut['rho'].iloc[0] == pytest.approx( 0.996660,
                                                       abs = 2e-6 )
    assert len( cpu._ccmCache ) == len( accelerated._ccmCache ) == 1
    assert all( np.isfinite( slope ) for slope in cpu._ccmCache.values() )
    assert accelerated._ccmCache == cpu._ccmCache
    assert cpu._ccmCache['driver'] == pytest.approx( 0.15737, abs = 1e-5 )
    assert cpu._ccmCache['driver'] > common['ccmSlope']
    assert accelerated._edimCache == cpu._edimCache
    assert accelerated.EDim == cpu.EDim
    assert cpu.EDim == { 'driver:target' : 1 }
    assert cpu._edimCache['driver'][1] > common['embedDimRhoMin']
    assert accelerated.rhoD_CCM == cpu.rhoD_CCM == {}
