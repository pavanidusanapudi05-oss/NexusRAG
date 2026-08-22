import os
from pathlib import Path
from typing import Optional, Dict, Any
import networkx as nx
import plotly.graph_objects as go
from pyvis.network import Network
from .graph_store import KnowledgeGraphStore, ENTITY_COLORS

class KnowledgeGraphVisualizer:
    @staticmethod
    def generate_pyvis_html(kg_store: KnowledgeGraphStore, output_path: str = 'nexusrag/data/processed/kg_interactive.html', height: str = '650px', width: str = '100%') -> str:
        net = Network(height=height, width=width, bgcolor='#0f172a', font_color='#f8fafc', directed=True)
        
        for node_id, data in kg_store.graph.nodes(data=True):
            name = data.get('name', node_id)
            e_type = data.get('type', 'Entity')
            color = data.get('color', ENTITY_COLORS.get(e_type, '#94A3B8'))
            
            props = data.get('properties', {})
            prop_lines = [f'{k}: {v}' for k, v in props.items()]
            title = f'{e_type}: {name}\n' + '\n'.join(prop_lines)
            
            net.add_node(
                node_id,
                label=name,
                title=title,
                color=color,
                size=22 if e_type in ['Policy', 'Regulation'] else 16,
                borderWidth=2,
                font={'color': '#f8fafc', 'size': 12, 'face': 'Inter, system-ui, sans-serif'}
            )
            
        for u, v, data in kg_store.graph.edges(data=True):
            rel_type = data.get('type', 'RELATES_TO')
            label = rel_type.replace('_', ' ').title()
            net.add_edge(
                u,
                v,
                label=label,
                title=rel_type,
                color='#64748b',
                arrows='to',
                font={'color': '#94a3b8', 'size': 10, 'align': 'middle'}
            )
            
        net.toggle_physics(True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(output_path)
        return output_path

    @staticmethod
    def generate_plotly_figure(kg_store: KnowledgeGraphStore) -> go.Figure:
        g = kg_store.graph
        if g.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(title='Knowledge Graph is Empty', paper_bgcolor='#0f172a', plot_bgcolor='#0f172a')
            return fig

        pos = nx.spring_layout(g, seed=42, k=0.6)

        edge_x, edge_y = [], []
        edge_annotations = []
        for edge in g.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_annotations.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    text=edge[2].get('type', '').replace('_', ' '),
                    showarrow=False,
                    font=dict(color='#94a3b8', size=9),
                    bgcolor='rgba(15, 23, 42, 0.6)'
                )
            )

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#475569'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_colors, node_text, node_hover = [], [], [], [], []
        for node, data in g.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            e_type = data.get('type', 'Entity')
            color = ENTITY_COLORS.get(e_type, '#94A3B8')
            node_colors.append(color)
            name = data.get('name', node)
            node_text.append(name)
            node_hover.append(f'<b>{name}</b><br>Type: {e_type}<br>ID: {node}')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_hover,
            text=node_text,
            textposition='top center',
            marker=dict(
                color=node_colors,
                size=20,
                line=dict(width=2, color='#ffffff')
            ),
            textfont=dict(color='#f8fafc', size=11)
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=40),
                            annotations=edge_annotations,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            paper_bgcolor='#0f172a',
                            plot_bgcolor='#0f172a'
                        ))
        return fig
