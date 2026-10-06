'''Optional Torch cross-map adapter for the MDE candidate sweep.

The public Torch kernel lives in :mod:`dimx.TorchBackend` so the driving and
fMRI pipelines can load that single file directly.  This module only adapts
MDE's DataFrame, 1-offset lib/pred ranges, and named candidate columns to that
array contract.

The established pyEDM path remains the reference implementation.  Unsupported
scientific settings never silently change meaning: ``auto`` falls back to the
CPU pool and explicit ``torch`` raises an actionable error.
'''

import importlib.util

import numpy as np


_BACKENDS = ('cpu', 'auto', 'torch')


#----------------------------------------------------------------------------
def _Ranges( ranges, name, N, Tp = None ):
    '''Expand inclusive, 1-offset pyEDM range pairs into row indices.

    MDE always calls Simplex with ``embedded=True``.  For a non-negative Tp,
    pyEDM shortens the final library segment so every neighbor has a target at
    neighbor + Tp.  Prediction rows are not shortened; rows without an
    in-record observation are removed later because they cannot contribute to
    rho.
    '''
    if not isinstance( ranges, list ) or not ranges or len( ranges ) % 2 :
        raise ValueError( f'{name} must contain start, stop pairs.' )

    out = []
    lastPair = len( ranges ) - 2
    for i in range( 0, len( ranges ), 2 ) :
        start, stop = ranges[ i ], ranges[ i + 1 ]
        if not isinstance( start, (int, np.integer) ) or \
           not isinstance( stop, (int, np.integer) ) :
            raise ValueError( f'{name} indices must be integers.' )
        if start < 1 or stop > N or start >= stop :
            raise ValueError( f'{name} pair [{start}, {stop}] is invalid for '
                              f'{N} rows.' )

        if Tp is not None and Tp >= 0 and i == lastPair :
            stop = stop - Tp
        if stop < start :
            raise ValueError( f'{name} has no usable rows after Tp={Tp}.' )

        out.extend( range( start - 1, stop ) )

    return np.asarray( out, dtype = np.int64 )


#----------------------------------------------------------------------------
def _TorchStatus( deviceName ):
    '''Return (available, reason) without importing Torch at package import.'''
    if importlib.util.find_spec( 'torch' ) is None :
        return False, 'optional dependency torch is not installed'

    try :
        import torch
        device = torch.device( deviceName )
    except Exception as exc :
        return False, f'invalid Torch device {deviceName!r}: {exc}'

    if device.type == 'cuda' and not torch.cuda.is_available() :
        return False, f'Torch device {deviceName!r} is unavailable'
    if device.type == 'mps' and not torch.backends.mps.is_available() :
        return False, f'Torch device {deviceName!r} is unavailable'

    try :
        torch.empty( 0, dtype = torch.float32, device = device )
    except Exception as exc :
        return False, f'Torch device {deviceName!r} is unavailable: {exc}'

    return True, None


#----------------------------------------------------------------------------
class TorchCrossMap :
    '''Run-scoped in-process Torch implementation of candidate cross mapping.'''

    def __init__( self, numericDF, argsD, deviceName = 'cuda',
                  batchCandidates = 16, predChunk = 128, logMsg = None ):
        self.target   = argsD['target']
        self.device   = deviceName
        self.batch    = batchCandidates
        self.chunk    = predChunk
        self.logMsg   = logMsg
        self._closed  = False

        frameColumns = list( numericDF.columns )
        if self.target not in frameColumns :
            raise RuntimeError( f'Torch cross map target {self.target!r} is '
                                'not in the numeric data.' )

        # Removed columns are not part of any candidate manifold. Excluding
        # them here avoids both needless host/device memory and an unnecessary
        # finite-data fallback caused by a value the user explicitly ignored.
        candidates = argsD.get( 'candidateColumns', frameColumns )
        columns = [ self.target ] + [ c for c in candidates
                                      if c != self.target ]
        missing = [ c for c in columns if c not in frameColumns ]
        if missing :
            raise RuntimeError( f'Torch cross map candidate columns not in '
                                f'the numeric data: {missing}.' )
        self.columns = columns
        self.columnIndex = { name : i for i, name in enumerate( columns ) }

        data = np.ascontiguousarray(
                   numericDF.loc[ :, columns ].to_numpy( dtype = np.float32 ),
                   dtype = np.float32 )
        N = data.shape[0]
        Tp = argsD['Tp']

        lib_i  = _Ranges( argsD['lib'],  'lib',  N, Tp = Tp )
        pred_i = _Ranges( argsD['pred'], 'pred', N )

        libTarget_i  = lib_i + Tp
        predTarget_i = pred_i + Tp
        if np.any( libTarget_i < 0 ) or np.any( libTarget_i >= N ) :
            raise RuntimeError( 'Torch cross map does not support a lib/Tp '
                                'combination whose target is outside the data.' )

        predValid = ( predTarget_i >= 0 ) & ( predTarget_i < N )
        pred_i = pred_i[ predValid ]
        predTarget_i = predTarget_i[ predValid ]
        if len( pred_i ) < 5 :
            raise RuntimeError( 'Torch cross map needs at least five prediction '
                                'rows with an observed target.' )

        # The standalone production kernel has no temporal row identifiers.
        # It is equivalent to pyEDM only when no library row must be excluded.
        exclusionRadius = argsD['exclusionRadius']
        # Search the sorted library rather than allocating an N_pred x N_lib
        # separation matrix (the latter is prohibitive for real fMRI runs).
        libSorted = np.sort( lib_i )
        excluded = False
        for predRow in pred_i :
            left = np.searchsorted( libSorted, predRow - exclusionRadius,
                                    side = 'left' )
            if left < len( libSorted ) and \
               libSorted[left] <= predRow + exclusionRadius :
                excluded = True
                break
        if excluded :
            raise RuntimeError(
                'Torch cross map currently requires disjoint lib/pred rows '
                'outside exclusionRadius; use crossMapBackend="auto" for '
                'scientifically exact CPU fallback.' )

        self.xlib  = data[ lib_i, : ]
        self.xpred = data[ pred_i, : ]
        target_i   = self.columnIndex[ self.target ]
        self.ylib  = data[ libTarget_i, target_i ]
        self.ypred = data[ predTarget_i, target_i ]

        arrays = ( self.xlib, self.xpred, self.ylib, self.ypred )
        if not all( np.isfinite( a ).all() for a in arrays ) :
            raise RuntimeError( 'Torch cross map currently requires finite '
                                'library, prediction, and target values; use '
                                'crossMapBackend="auto" for CPU fallback.' )

        if self.logMsg is not None :
            self.logMsg( f'\tTorchCrossMap device={self.device} '
                         f'batchCandidates={self.batch} predChunk={self.chunk}' )

    #------------------------------------------------------------------------
    def CrossMap( self, candidateColumns, dimension = 1,
                  logPct = 0, verbose = False ):
        '''Return the same named rho dictionary as ``CrossMapPool.CrossMap``.'''
        if not candidateColumns :
            return {}

        if len( self.ylib ) < dimension + 1 :
            raise RuntimeError( f'Torch cross map {dimension}-D needs at least '
                                f'{dimension + 1} library rows.' )

        selectedNames = list( candidateColumns[0][1:] )
        for cols in candidateColumns :
            if list( cols[1:] ) != selectedNames :
                raise RuntimeError( 'Torch cross map candidate state columns '
                                    'must be identical within a dimension.' )

        try :
            selected  = [ self.columnIndex[c] for c in selectedNames ]
            candidates = np.asarray(
                [ self.columnIndex[cols[0]] for cols in candidateColumns ],
                dtype = np.int64 )
        except KeyError as exc :
            raise RuntimeError( f'Torch cross map unknown column {exc.args[0]!r}.' ) \
                from exc

        from .TorchBackend import candidate_sweep_torch

        _, _, rows, _ = candidate_sweep_torch(
            self.xlib, self.ylib, self.xpred, self.ypred,
            selected         = selected,
            candidates       = candidates,
            batch_candidates = self.batch,
            pred_chunk       = self.chunk,
            device_name      = self.device )

        result = {}
        for cols, row in zip( candidateColumns, rows ) :
            rho = row['rho']
            if np.isfinite( rho ) :
                # pyEDM ComputeError reports rho at six decimal places.
                rho = round( float( rho ), 6 )
            # CrossMapPool returns a NumPy scalar. Preserve that internal
            # contract so Run() remains identical for CPU and Torch pools.
            rho = np.float64( rho )
            key = ','.join( cols )
            result[ f'{key}:{self.target}' ] = ( rho, list( cols ) )

        if verbose and self.logMsg is not None and logPct and logPct >= 100 :
            total = len( candidateColumns )
            self.logMsg( f'\t{dimension}-D cross map {total}/{total} (100%)' )

        return result

    #------------------------------------------------------------------------
    def close( self ):
        self._closed = True


#----------------------------------------------------------------------------
def ResolveCrossMap( numericDF, argsD, backend = 'cpu', torchDevice = 'cuda',
                     batchCandidates = 16, predChunk = 128,
                     crossMapCores = None, mpMethod = None, sharedMem = 0.1,
                     maxTasks = None, logMsg = None, cpuFactory = None ):
    '''Resolve the requested sweep backend and return its run-scoped object.'''
    if backend not in _BACKENDS :
        raise RuntimeError( f'Unknown crossMapBackend {backend!r}; expected '
                            'cpu, auto, or torch.' )
    if cpuFactory is None :
        from .Parallel import CrossMapPool
        cpuFactory = CrossMapPool

    def CPU():
        return cpuFactory( numericDF, argsD,
                           crossMapCores = crossMapCores,
                           mpMethod  = mpMethod,
                           sharedMem = sharedMem,
                           maxTasks  = maxTasks,
                           logMsg    = logMsg )

    if backend == 'cpu' :
        return CPU()

    available, reason = _TorchStatus( torchDevice )
    if not available :
        if backend == 'torch' :
            raise RuntimeError( f'Torch cross map requested, but {reason}.' )
        if logMsg is not None :
            logMsg( f'\tTorch cross map unavailable ({reason}); using CPU.' )
        return CPU()

    try :
        return TorchCrossMap( numericDF, argsD,
                              deviceName = torchDevice,
                              batchCandidates = batchCandidates,
                              predChunk = predChunk,
                              logMsg = logMsg )
    except ( RuntimeError, ValueError ) as exc :
        if backend == 'torch' :
            raise
        if logMsg is not None :
            logMsg( f'\tTorch cross map unsupported ({exc}); using CPU.' )
        return CPU()
