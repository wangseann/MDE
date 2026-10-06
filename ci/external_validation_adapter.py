'''Compatibility hook for the pinned EDM_MDE_validation tests.

The independent validation repository predates MDEConfig. Its two MDE tests
still provide the former ``cores``, ``title``, and ``removeTime`` arguments.
The test files and golden outputs remain unmodified. Shared keywords are
translated before each external MDE test; a scoped wrapper translates
``removeTime`` at the constructor call, after test-specific overrides.

The external ``test_simplex7`` also inserts NaNs directly into pyEDM's shared
Lorenz sample instead of a copy. An automatic fixture restores that sample
after every test so later validation files always receive pristine data.
'''
from types import SimpleNamespace

import pytest


#------------------------------------------------------------
def _LegacyMDE( *args, **kwargs ):
    '''Map the external suite's explicit first-column removal to noTime.'''
    from dimx import MDE

    if kwargs.pop( 'removeTime', False ) :
        kwargs['noTime'] = False
    return MDE( *args, **kwargs )


#------------------------------------------------------------
@pytest.fixture( autouse = True )
def _AdaptLegacyMDE( request, monkeypatch ):
    '''Replace only the pinned test module's MDE entry point for this test.'''
    if request.node.path.name == 'test_MDE.py' :
        monkeypatch.setattr( request.module, 'dx',
                             SimpleNamespace( MDE = _LegacyMDE ) )


#------------------------------------------------------------
@pytest.fixture( autouse = True )
def _IsolateLorenzSampleData():
    '''Restore shared pyEDM Lorenz data after every external test.'''
    from pyEDM import sampleData

    snapshot = sampleData['Lorenz5D'].copy( deep = True )
    try :
        yield
    finally :
        sampleData['Lorenz5D'] = snapshot


#------------------------------------------------------------
def pytest_runtest_setup( item ):
    '''Translate retired harness-only keywords without changing dimx.'''
    if item.path.name != 'test_MDE.py' :
        return

    MDEArgs = getattr( item.module, 'MDEArgs', None )
    if MDEArgs is None :
        return

    if 'cores' in MDEArgs :
        MDEArgs['crossMapCores'] = MDEArgs.pop( 'cores' )

    # title controlled the legacy automatic plot. Both external tests set
    # plot=False, and current MDE exposes title through MDE.Plot( title=... ).
    MDEArgs.pop( 'title', None )
