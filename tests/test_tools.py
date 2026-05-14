import matplotlib.figure

import tools


# --- list_cancers ---


def test_list_cancers_returns_sorted_list():
    result = tools.list_cancers()
    assert isinstance(result, list)
    assert result == sorted(result)
    assert len(result) == 10
    assert "lung" in result


# --- get_targets ---


def test_get_targets_known_cancer_returns_gene_list():
    result = tools.get_targets("lung")
    assert isinstance(result, list)
    assert "ALK" in result
    assert "KRAS" in result


def test_get_targets_known_cancer_is_case_insensitive():
    assert tools.get_targets("LUNG") == tools.get_targets("lung")


def test_get_targets_unknown_cancer_returns_error_with_suggestions():
    result = tools.get_targets("breastt")
    assert isinstance(result, dict)
    assert "error" in result
    assert "unknown cancer" in result["error"].lower()
    assert "breast" in result["did_you_mean"]


def test_get_targets_unknown_cancer_includes_available_list():
    result = tools.get_targets("totally-made-up-cancer")
    assert isinstance(result, dict)
    assert result["did_you_mean"] == []
    assert sorted(result["available"]) == sorted(tools.list_cancers())


# --- get_expressions ---


def test_get_expressions_returns_dict_of_floats():
    genes = tools.get_targets("lung")
    result = tools.get_expressions(genes)
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, float) for v in result.values())
    assert set(result.keys()) == set(genes)


def test_get_expressions_empty_list_returns_empty_dict():
    assert tools.get_expressions([]) == {}


def test_get_expressions_unknown_gene_is_omitted():
    result = tools.get_expressions(["NOT_A_REAL_GENE", "KRAS"])
    assert "NOT_A_REAL_GENE" not in result
    assert "KRAS" in result


# --- top_genes ---


def test_top_genes_returns_sorted_descending():
    result = tools.top_genes("lung", 3)
    assert isinstance(result, list)
    assert len(result) == 3
    values = [row["median_value"] for row in result]
    assert values == sorted(values, reverse=True)


def test_top_genes_default_n_is_5():
    result = tools.top_genes("breast")
    assert len(result) == 5


def test_top_genes_caps_at_available_count():
    result = tools.top_genes("lung", 100)
    # Lung has fewer than 100 rows; result should equal the actual count
    assert len(result) == len(tools.get_targets("lung"))


def test_top_genes_unknown_cancer_returns_error():
    result = tools.top_genes("nonexistent", 3)
    assert isinstance(result, dict)
    assert "error" in result


# --- compare_cancers ---


def test_compare_cancers_returns_three_buckets():
    result = tools.compare_cancers("breast", "gastric")
    assert set(result.keys()) == {"shared", "only_a", "only_b"}
    assert "PIK3CA" in result["shared"]
    assert "CDH1" in result["shared"]


def test_compare_cancers_disjoint_sets():
    result = tools.compare_cancers("lung", "prostate")
    lung_genes = set(tools.get_targets("lung"))
    prostate_genes = set(tools.get_targets("prostate"))
    assert set(result["shared"]) == lung_genes & prostate_genes
    assert set(result["only_a"]) == lung_genes - prostate_genes
    assert set(result["only_b"]) == prostate_genes - lung_genes


def test_compare_cancers_unknown_first_returns_error():
    result = tools.compare_cancers("nope", "lung")
    assert "error" in result


def test_compare_cancers_unknown_second_returns_error():
    result = tools.compare_cancers("lung", "nope")
    assert "error" in result


# --- plot_expressions ---


def test_plot_expressions_returns_chart_id_and_figure():
    tools.reset_figures()
    result = tools.plot_expressions({"KRAS": 0.359, "ALK": 0.215}, "Lung sample")
    assert "chart_id" in result
    fig = tools.pop_last_figure()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_expressions_empty_returns_error():
    result = tools.plot_expressions({}, "empty")
    assert isinstance(result, dict)
    assert "error" in result


def test_plot_expressions_bars_match_input_order():
    tools.reset_figures()
    tools.plot_expressions({"A": 0.1, "B": 0.5, "C": 0.3}, "test")
    fig = tools.pop_last_figure()
    ax = fig.axes[0]
    bar_heights = [p.get_height() for p in ax.patches]
    assert bar_heights == [0.1, 0.5, 0.3]


# --- TOOL_SPEC and dispatch ---


def test_dispatch_routes_known_tool():
    result, fig = tools.dispatch("list_cancers", {})
    assert isinstance(result, list)
    assert fig is None


def test_dispatch_routes_unknown_tool():
    result, fig = tools.dispatch("not_a_tool", {})
    assert "error" in result
    assert "available" in result
    assert fig is None


def test_dispatch_returns_figure_when_tool_emits_one():
    result, fig = tools.dispatch(
        "plot_expressions", {"expressions": {"X": 0.5}, "title": "t"}
    )
    assert "chart_id" in result
    assert isinstance(fig, matplotlib.figure.Figure)


def test_dispatch_handles_bad_arguments_gracefully():
    result, fig = tools.dispatch("get_targets", {})
    assert "error" in result


def test_tool_spec_has_all_six_tools():
    names = [t["function"]["name"] for t in tools.TOOL_SPEC]
    assert set(names) == {
        "list_cancers",
        "get_targets",
        "get_expressions",
        "top_genes",
        "compare_cancers",
        "plot_expressions",
    }
