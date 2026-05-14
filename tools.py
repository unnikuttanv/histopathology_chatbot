"""Agent tools over the take-home dataset.

Every tool is a pure function. Tools never raise to the agent —
they return either the requested data or a structured {"error": ...}
result. The agent treats both as normal tool output.
"""
from __future__ import annotations

import difflib
import uuid

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import data


def list_cancers() -> list[str]:
    """Return the cancer indications present in the dataset."""
    return data.cancers()


def get_targets(cancer_name: str) -> list[str] | dict:
    """Return the list of genes associated with a cancer indication.

    Returns either a list[str] of gene names, or a structured error dict
    when the cancer name is not present in the dataset.
    """
    if not isinstance(cancer_name, str) or not cancer_name.strip():
        return {
            "error": "cancer_name must be a non-empty string",
            "did_you_mean": [],
            "available": list_cancers(),
        }
    name = cancer_name.strip().lower()
    available = list_cancers()
    if name not in available:
        suggestions = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        return {
            "error": f"unknown cancer {cancer_name!r}",
            "did_you_mean": suggestions,
            "available": available,
        }
    return data.df.loc[data.df["cancer_indication"] == name, "gene"].tolist()


def get_expressions(genes: list[str]) -> dict[str, float]:
    """Return median expression values for the given genes.

    Unknown genes are silently omitted (consistent with the take-home spec).
    """
    if not genes:
        return {}
    subset = data.df[data.df["gene"].isin(genes)]
    return dict(zip(subset["gene"], subset["median_value"].astype(float)))


def top_genes(cancer_name: str, n: int = 5) -> list[dict] | dict:
    """Return the top-N genes by median expression for a cancer."""
    targets = get_targets(cancer_name)
    if isinstance(targets, dict):
        return targets  # error dict
    name = cancer_name.strip().lower()
    subset = (
        data.df[data.df["cancer_indication"] == name]
        .sort_values("median_value", ascending=False)
        .head(n)
    )
    return [
        {"gene": row.gene, "median_value": float(row.median_value)}
        for row in subset.itertuples(index=False)
    ]


def compare_cancers(cancer_a: str, cancer_b: str) -> dict:
    """Compare two cancers by gene set."""
    a_genes = get_targets(cancer_a)
    if isinstance(a_genes, dict):
        return a_genes
    b_genes = get_targets(cancer_b)
    if isinstance(b_genes, dict):
        return b_genes
    a_set, b_set = set(a_genes), set(b_genes)
    return {
        "shared": sorted(a_set & b_set),
        "only_a": sorted(a_set - b_set),
        "only_b": sorted(b_set - a_set),
    }


# Side-channel: tools that produce a figure append it here. The agent
# pops figures off this queue after dispatching a tool call so the UI
# can render them. Keeps figure objects out of the LLM's context.
_figure_queue: list[Figure] = []


def reset_figures() -> None:
    _figure_queue.clear()


def pop_last_figure() -> Figure | None:
    return _figure_queue.pop() if _figure_queue else None


def plot_expressions(expressions: dict[str, float], title: str = "") -> dict:
    """Render a bar chart of expression values. Returns a chart_id;
    the figure is queued on the side-channel for the UI."""
    if not expressions:
        return {"error": "no expressions provided"}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    genes = list(expressions.keys())
    values = list(expressions.values())
    ax.bar(genes, values)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Median expression")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _figure_queue.append(fig)
    return {"chart_id": str(uuid.uuid4()), "n_bars": len(genes), "title": title}


TOOL_SPEC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_cancers",
            "description": "List the cancer indications present in the dataset.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_targets",
            "description": (
                "Return the list of genes associated with a cancer indication. "
                "Returns a structured error with did_you_mean if the cancer is unknown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cancer_name": {
                        "type": "string",
                        "description": "Cancer indication, e.g. 'lung'. Case-insensitive.",
                    }
                },
                "required": ["cancer_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_expressions",
            "description": "Return median expression values for the given genes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene symbols.",
                    }
                },
                "required": ["genes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_genes",
            "description": "Return the top-N genes by median expression for a cancer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cancer_name": {"type": "string"},
                    "n": {"type": "integer", "default": 5},
                },
                "required": ["cancer_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cancers",
            "description": (
                "Compare two cancers by gene set. Returns shared, only_a, only_b."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cancer_a": {"type": "string"},
                    "cancer_b": {"type": "string"},
                },
                "required": ["cancer_a", "cancer_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_expressions",
            "description": (
                "Render a bar chart of expression values. Call get_expressions first; "
                "pass its dict as the 'expressions' argument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expressions": {
                        "type": "object",
                        "description": "Mapping from gene symbol to median expression value.",
                    },
                    "title": {"type": "string", "default": ""},
                },
                "required": ["expressions"],
            },
        },
    },
]


_DISPATCH = {
    "list_cancers": list_cancers,
    "get_targets": get_targets,
    "get_expressions": get_expressions,
    "top_genes": top_genes,
    "compare_cancers": compare_cancers,
    "plot_expressions": plot_expressions,
}


def dispatch(name: str, args: dict) -> tuple[dict | list, Figure | None]:
    """Execute a tool by name. Returns (result, optional_figure).

    Unknown tools and bad arguments are returned as structured errors,
    never raised to the agent.
    """
    fn = _DISPATCH.get(name)
    if fn is None:
        return (
            {"error": f"unknown tool {name!r}", "available": list(_DISPATCH)},
            None,
        )
    try:
        result = fn(**(args or {}))
    except TypeError as e:
        return ({"error": f"bad arguments for {name}: {e}", "received": args}, None)
    except Exception as e:
        return ({"error": f"{type(e).__name__}: {e}"}, None)
    fig = pop_last_figure() if name == "plot_expressions" else None
    return result, fig
