'''Regression coverage for the upstream graph application.'''
import json

import networkx as nx
from pandas import DataFrame

from dimx.apps.GraphMDE import GraphMDE


def test_graph_rejects_self_loops_and_longer_cycles():
    frames = {
        'root': DataFrame( { 'variables': ['root', 'a'],
                            'rho': [1., 0.8] } ),
        'a': DataFrame( { 'variables': ['root', 'b'],
                         'rho': [0.7, 0.6] } ),
    }

    graph = GraphMDE( frames, N = 2 ).BuildGraph()

    assert set( graph.edges ) == { ('a', 'root'), ('b', 'a') }
    assert nx.is_directed_acyclic_graph( graph )


def test_graph_exports_unmeasured_rho_as_json_null( tmp_path ):
    output = tmp_path / 'graph.json'
    frames = { 'root': DataFrame( { 'variables': ['a', 'b'] } ) }

    graph = GraphMDE( frames, N = 1, outFile = str(output) ).BuildGraph()

    assert set( graph.edges ) == { ('a', 'root') }
    restored = nx.node_link_graph( json.loads( output.read_text() ) )
    assert set( restored.edges ) == { ('a', 'root') }
    assert restored['a']['root']['rho'] is None
