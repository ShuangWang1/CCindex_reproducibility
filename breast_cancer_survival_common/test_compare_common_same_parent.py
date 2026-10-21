"""Small scientific regression cases for shared-parent comparisons."""
import importlib.util
from pathlib import Path
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SPEC = importlib.util.spec_from_file_location('common_same_parent', Path(__file__).with_name('compare_common_same_parent.py'))
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_reverse_growth_includes_multiple_parents_and_edge_operations():
    graph = analysis.decode('0&2&4&6', '0-1;0-2;1-2;0-3;1-3')
    candidates = list(analysis.direct_parents(graph))
    assert {change for op, change, _ in candidates if op == 'add_vertex'} == {'2', '3'}
    assert any(op == 'add_edge' for op, _, _ in candidates)
    assert all(sum(nx.triangles(p).values()) // 3 == 1 and nx.is_connected(p)
               for _, _, p in candidates)


def test_classification_is_cohort_specific_and_requires_estimable_fits():
    statistics = pd.DataFrame({'p_value_rank_binary': [.01, .05, np.nan, .01, 0, 1.1],
                               'rank_binary_fit_status': ['estimated'] * 3 + ['failed', 'estimated', 'estimated']})
    assert analysis.classify(statistics).tolist() == ['significant', 'nonsignificant'] + ['undefined'] * 4


def test_vectorized_tests_match_scalar_and_bh_with_missing_values():
    values = np.array([[1, 1, 0, 1], [2, 1, 0, 2], [3, np.nan, 0, 3],
                       [5, 4, 0, 5], [6, 5, 0, 6], [7, 6, 0, 7]], dtype=float)
    sig = np.array([False] * 3 + [True] * 3)
    result = analysis.compare_values(values, sig, ~sig)
    assert result.metric_index.tolist() == [0, 3]  # constant / insufficient criteria excluded
    expected = mannwhitneyu([5, 6, 7], [1, 2, 3], alternative='two-sided', method='asymptotic')
    np.testing.assert_allclose(result.location_p_value, expected.pvalue)
    np.testing.assert_allclose(result.location_q_bh, expected.pvalue)
    np.testing.assert_allclose(result.cliffs_delta, 1)
    np.testing.assert_allclose(analysis.bh([.01, .04, .03]), [.03, .04, .04])


def test_shared_order_removes_jointly_gray_parents_and_retains_one_sided_values():
    first = pd.DataFrame(np.nan, index=['a', 'b'], columns=['p1', 'p2', 'p3'])
    second = first.copy()
    second.loc['a', 'p2'] = .01
    rows, columns = analysis.common_order([first, second])
    assert set(rows) == {'a', 'b'}
    assert columns == ['p2']
    assert first.loc[rows, columns].isna().all().all()
    assert analysis.visible_parents([first, second]) == ['p2']
    first.loc['b', 'p3'] = 0.0
    assert analysis.visible_parents([first, second]) == ['p2', 'p3']
    _, columns = analysis.common_order([first, second])
    assert set(columns) == {'p2', 'p3'}
    empty = pd.DataFrame(np.nan, index=['a'], columns=['p1'])
    assert analysis.visible_parents([empty, empty]) == []


def test_exact_matching_preserves_labels_even_with_identical_topology():
    graph = analysis.decode('0&2&4', '0-1;0-2;1-2')
    identical = analysis.decode('4&0&2', '0-1;0-2;1-2')
    different = analysis.decode('0&2&6', '0-1;0-2;1-2')
    assert analysis.same_graph(graph, identical)
    assert not analysis.same_graph(graph, different)


def test_matches_established_same_parent_implementation():
    path = analysis.REPO / 'breast_cancer_survival/motif_results/sibling_topology_analysis_new/run_same_parent_comparison.py'
    spec = importlib.util.spec_from_file_location('established_same_parent', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    frame = pd.DataFrame({'degree_mean': [1., 2, 2, 5, 6, 7],
                          'local_egophily_mean': [0., 1, 3, 2, 1, 0],
                          'significance_group': ['nonsignificant'] * 3 + ['significant'] * 3})
    metrics = ['degree_mean', 'local_egophily_mean']
    _, expected = reference.compare_parent(frame, metrics, 3)
    sig = frame.significance_group.eq('significant').to_numpy()
    actual = analysis.compare_values(frame[metrics].to_numpy(), sig, ~sig)
    actual['metric'] = np.asarray(metrics)[actual.metric_index]
    actual = actual.set_index('metric').sort_index()
    expected = expected.set_index('metric').sort_index()
    for field in ['location_p_value', 'location_q_bh', 'cliffs_delta',
                  'median_difference', 'standardized_median_difference']:
        np.testing.assert_allclose(actual[field], expected[field])


def test_feature_cache_preserves_float_values_and_ties():
    values = [np.nextafter(1/3, 0), 1/3, np.nextafter(1/3, 1), 1/3]
    original = pd.DataFrame({'common_motif_id': ['a', 'b', 'c', 'd'], 'metric': values})
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'features.csv.gz'
        analysis.write_csv(original, path)
        recovered = analysis.read_features(path)
    np.testing.assert_array_equal(recovered.metric, original.metric)
    assert recovered.metric.nunique() == original.metric.nunique()


if __name__ == '__main__':
    tests = [(name, value) for name, value in list(globals().items()) if name.startswith('test_')]
    for name, test in tests:
        test()
        print(f'PASS {name}')
    print(f'{len(tests)} checks passed')
