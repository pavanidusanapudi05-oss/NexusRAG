import plotly.graph_objects as go
import networkx as nx
from typing import Dict, Any, Optional
from .graph_store import LocalKnowledgeGraph

class GraphVisualizer:
    @staticmethod
    def create_plotly_figure(kg: LocalKnowledgeGraph, filter_doc: Optional[str] = None) -> go.Figure:
        g = kg.graph
        if g.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(title="No Graph Entities Available", template="plotly_dark")
            return fig

        # Spring layout positions
        pos = nx.spring_layout(g, seed=42, k=0.8)

        edge_x = []
        edge_y = []
        for edge in g.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color="#64748b"),
            hoverinfo="none",
            mode="lines"
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []

        type_colors = {
            "Policy": "#818cf8",
            "Regulation": "#ef4444",
            "Department": "#34d399",
            "Requirement": "#f59e0b",
            "Document": "#60a5fa"
        }

        for node in g.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            attr = g.nodes[node]
            name = attr.get("name", node)
            etype = attr.get("type", "Entity")
            doc = attr.get("document_name", "N/A")
            node_text.append(f"<b>{name}</b><br>Type: {etype}<br>Doc: {doc}")
            node_color.append(type_colors.get(etype, "#94a3b8"))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[g.nodes[n].get("name", n)[:18] for n in g.nodes()],
            textposition="bottom center",
            hovertext=node_text,
            marker=dict(
                color=node_color,
                size=22,
                line=dict(width=2, color="#0f172a")
            ),
            textfont=dict(size=10, color="#f8fafc")
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        return fig
