'''Small numerical checks for consistent Evaluate split handling.'''
import numpy as np
from numpy.testing import assert_array_equal
from pandas import DataFrame

from dimx import Evaluate


def test_forecast_training_targets_stay_inside_library():
    rows = np.arange( 1., 51. )
    frame = DataFrame( {
        'driver_x': np.sin(rows),
        'driver_y': np.cos(rows / 3.),
        'driver_z': np.sin(rows / 7.),
        'target': rows,
    } )
    evaluation = Evaluate(
        frame, mde_columns = ['driver_x', 'driver_y'],
        columnMatch = ['driver'], predictVar = 'target',
        library = [1, 30], prediction = [35, 50], Tp = 2,
        noTime = True, components = 2, dmap_k = 10 )

    evaluation.Run()

    assert evaluation.lib_i == list( range(28) )
    assert_array_equal( evaluation.predictVar_lib, np.arange(3., 31.) )
    assert_array_equal( evaluation.predictVar_pred, np.arange(37., 51.) )
    assert_array_equal( evaluation.mdeEval['Observations'].values,
                        evaluation.predictVar_pred )
