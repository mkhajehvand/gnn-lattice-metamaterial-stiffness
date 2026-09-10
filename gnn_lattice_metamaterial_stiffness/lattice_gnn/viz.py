# render catalog, periodic, and graph representations for selected structures

import numpy as np
import networkx as nx
from .plot_style import configure_matplotlib
configure_matplotlib()
import matplotlib.pyplot as plt

# import the module so plots use data loaded by the current process
from . import data as data

STRUT_COLOR = [0.00, 0.49, 0.52]
LEFT_NODE_SIZE = 170
LEFT_CUT_NODE_SIZE = 210
LEFT_LABEL_SIZE = 8
LEFT_TITLE_SIZE = 10
LEFT_3D_ZOOM = 1.18
LEFT_3D_ELEVATION = 10
LEFT_3D_AZIMUTH_DEFAULT = -15
# use structure specific viewing angles to improve visual clarity
LEFT_3D_AZIMUTH_BY_STRUCTURE = {
    "cub_Z06.0_E1": -15,
    "cub_Z08.0_E3": -15,
    "cub_Z14.0_E38": -15,
    "cub_Z12.0_E19": -15,
    "cub_Z04.0_E970": -15,
    "ort_Z04.3_E319": -30,
}
LEFT_3D_STRUT_WIDTH = 2.4
BIPARTITE_NODE_LINEAR_SCALE = 2.5
BIPARTITE_STRUT_NODE_SIZE = 115 * BIPARTITE_NODE_LINEAR_SCALE**2
BIPARTITE_JOINT_NODE_SIZE = 145 * BIPARTITE_NODE_LINEAR_SCALE**2
BIPARTITE_CUT_NODE_SIZE = 170 * BIPARTITE_NODE_LINEAR_SCALE**2
BIPARTITE_LABEL_SIZE = 7 * BIPARTITE_NODE_LINEAR_SCALE


# restore the original joint palette and color assignment
def _joint_color_map(key_order):
    cmap_name = "tab20" if len(key_order) <= 20 else "hsv"
    cmap = plt.get_cmap(cmap_name, max(1, len(key_order)))
    return {key: cmap(idx % cmap.N) for idx, key in enumerate(key_order)}

# use a stable fallback for graphs too small for networkx's Kamada-Kawai layout
def _kk_or_spring_layout(G, scale=1.0, seed=0):
    if G.number_of_nodes() < 3:
        return nx.spring_layout(G, scale=scale, seed=seed)
    return nx.kamada_kawai_layout(G, scale=scale)

# map catalog nodes to their fundamental periodic structure joint IDs
def _catalog_to_fundamental_joint_ids(
    pos_frac: np.ndarray,
    key_order,
    window_shift: np.ndarray,
    tolerance: float = 5e-6,
):
    shifted = np.mod(
        np.asarray(pos_frac, dtype=np.float64)
        + np.asarray(window_shift, dtype=np.float64).reshape(1, 3),
        1.0,
    )
    fundamental_positions = np.mod(
        np.asarray(key_order, dtype=np.float64).reshape(-1, 3),
        1.0,
    )
    raw_to_fundamental = {}
    for raw_jid, shifted_position in enumerate(shifted):
        difference = np.abs(fundamental_positions - shifted_position)
        periodic_difference = np.minimum(difference, 1.0 - difference)
        distances = np.max(periodic_difference, axis=1)
        fundamental_jid = int(np.argmin(distances))
        raw_to_fundamental[int(raw_jid)] = (
            fundamental_jid
            if float(distances[fundamental_jid]) <= float(tolerance)
            else None
        )
    return raw_to_fundamental

# build the multiplicity-weighted periodic strut graph
def _build_strut_graph_nx(graph):
    G = nx.Graph()
    G.add_nodes_from(range(int(graph["num_struts"])))
    records = graph.get("strut_line_adjacency")
    if records is None:
        records = data._periodic_strut_adjacency_records(graph["strut_endpoints"])
    for a, b, multiplicity in records:
        G.add_edge(int(a), int(b), multiplicity=int(multiplicity))

    for n in G.nodes():
        G.nodes[n]['label'] = f"s{n}"
        G.nodes[n]['color'] = STRUT_COLOR
    return G

# construct the displayed periodic joint multigraph
def _build_joint_graph_nx(graph, key2col):
    key_order = graph["key_order"]
    boundary_cut_joint_ids = set(int(x) for x in graph.get("boundary_cut_joint_ids", set()))
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(key_order)))

    for idx, key in enumerate(key_order):
        is_cut = idx in boundary_cut_joint_ids
        G.nodes[idx]['label'] = f"j{idx}" + ("*" if is_cut else "")
        G.nodes[idx]['color'] = 'k' if is_cut else key2col[key]
        G.nodes[idx]['is_cut'] = is_cut

    G.add_edges_from(
        (int(u), int(v))
        for u, v in np.asarray(graph["strut_endpoints"], dtype=np.int64)
    )
    return G

# construct the displayed joint-strut incidence multigraph
def _build_bipartite_graph_nx(graph, key2col):
    key_order = graph["key_order"]
    boundary_cut_joint_ids = set(int(x) for x in graph.get("boundary_cut_joint_ids", set()))
    G = nx.MultiGraph()
    for s in range(int(graph["num_struts"])):
        G.add_node(f"s{s}", bipartite=0, color=STRUT_COLOR)

    for idx, key in enumerate(key_order):
        is_cut = idx in boundary_cut_joint_ids
        G.add_node(
            f"j{idx}",
            bipartite=1,
            color='k' if is_cut else key2col[key],
            is_cut=is_cut,
        )

    for s_idx, (u, v) in enumerate(np.asarray(graph["strut_endpoints"], dtype=np.int64)):
        for jid in (int(u), int(v)):
            G.add_edge(f"s{s_idx}", f"j{jid}")
    return G

# draw a centered joint label above every 3D marker and strut
def _draw_3d_joint_label(ax, xyz, label: str, fontsize: float):
    return ax.text(
        *np.asarray(xyz, dtype=np.float64),
        label,
        fontsize=fontsize,
        ha="center",
        va="center",
        color="k",
        zorder=100,
        clip_on=False,
    )


# renders one multi panel figure per structure name
# figures are saved by the runner's plt.show() monkeypatch
def plot_structures(struct_names):
    for name in struct_names:
        if name not in data.structures:
            print(
                f"[viz] warning: {name} not found in the "
                "Lumpe and Stankovic (2021) catalog"
            )
            continue

        try:
            graph = data.build_periodic_structure_graph(name, data.structures)
        except ValueError as e:
            print(f"[viz] warning: {name} skipped ({e})")
            continue
        entry = data.structures[name]
        raw_pos_frac = np.asarray(entry["pos"], dtype=np.float64)
        raw_pos_cart, raw_pos_frac, raw_struts, _ = data.strut_features(
            name, data.structures
        )
        key_order = list(graph["key_order"])
        endpoints = np.asarray(graph["strut_endpoints"], dtype=np.int64)
        shifts = np.asarray(graph["strut_shifts"], dtype=np.int64)
        cell_matrix = np.asarray(graph["cell_matrix_unscaled"], dtype=np.float64)
        key2col = _joint_color_map(key_order)
        raw_to_fundamental = _catalog_to_fundamental_joint_ids(
            raw_pos_frac,
            key_order,
            graph["grega_window_shift_vector"],
        )
        left_3d_azimuth = LEFT_3D_AZIMUTH_BY_STRUCTURE.get(
            name, LEFT_3D_AZIMUTH_DEFAULT
        )

        fig = plt.figure(figsize=(8.8, 11.9))
        gs = fig.add_gridspec(
            4,
            2,
            width_ratios=[2, 1],
            height_ratios=[1, 1, 1, 1],
            wspace=0.02,
            hspace=0.10,
        )

        # (a) catalog structure before scaling
        ax3d = fig.add_subplot(
            gs[0, 0], projection='3d', computed_zorder=False
        )
        ax3d.view_init(LEFT_3D_ELEVATION, left_3d_azimuth)
        ax3d.set_aspect('equal')
        try:
            ax3d.set_box_aspect((1, 1, 1), zoom=LEFT_3D_ZOOM)
        except TypeError:
            ax3d.set_box_aspect((1, 1, 1))
        ax3d.axis('off')
        raw_plot_points = [np.asarray(x, dtype=np.float64) for x in raw_pos_cart]
        raw_coords = np.asarray(raw_plot_points, dtype=np.float64)
        raw_lo = raw_coords.min(axis=0) - 0.05
        raw_hi = raw_coords.max(axis=0) + 0.05
        raw_span = max(float(np.max(raw_hi - raw_lo)), 1e-6)
        raw_center = 0.5 * (raw_lo + raw_hi)
        for i, j in raw_struts:
            start = np.asarray(raw_pos_cart[int(i)], dtype=np.float64)
            end = np.asarray(raw_pos_cart[int(j)], dtype=np.float64)
            ax3d.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                c=STRUT_COLOR,
                lw=LEFT_3D_STRUT_WIDTH,
                alpha=1.0,
                zorder=1,
            )
        for raw_jid, xyz in enumerate(raw_pos_cart):
            fundamental_jid = raw_to_fundamental[raw_jid]
            if fundamental_jid is None:
                ax3d.scatter(
                    *xyz, s=LEFT_CUT_NODE_SIZE, marker="x", c="k",
                    linewidths=1.8,
                    depthshade=False, zorder=5,
                )
                continue
            fundamental_key = key_order[fundamental_jid]
            ax3d.scatter(
                *xyz,
                s=LEFT_NODE_SIZE,
                marker="o",
                c=[key2col[fundamental_key]],
                depthshade=False,
                zorder=5,
            )
            _draw_3d_joint_label(
                ax3d,
                xyz,
                f"j{fundamental_jid}",
                LEFT_LABEL_SIZE,
            )
        ax3d.set_xlim(raw_center[0] - raw_span / 2, raw_center[0] + raw_span / 2)
        ax3d.set_ylim(raw_center[1] - raw_span / 2, raw_center[1] + raw_span / 2)
        ax3d.set_zlim(raw_center[2] - raw_span / 2, raw_center[2] + raw_span / 2)
        ax3d.set_title(
            f'"{name}" Structure',
            fontsize=LEFT_TITLE_SIZE,
        )

        # (b) fundamental periodic structure before density scaling
        ax_fund = fig.add_subplot(
            gs[1, 0], projection='3d', computed_zorder=False
        )
        ax_fund.view_init(LEFT_3D_ELEVATION, left_3d_azimuth)
        ax_fund.set_aspect('equal')
        try:
            ax_fund.set_box_aspect((1, 1, 1), zoom=LEFT_3D_ZOOM)
        except TypeError:
            ax_fund.set_box_aspect((1, 1, 1))
        ax_fund.axis('off')
        plot_points = []
        fundamental_points = []
        for jid, key in enumerate(key_order):
            xyz = cell_matrix @ np.asarray(key, dtype=np.float64)
            plot_points.append(xyz)
            fundamental_points.append(xyz)
            col = key2col[key]
            ax_fund.scatter(
                *xyz, s=LEFT_NODE_SIZE, marker="o", c=[col], alpha=1.0,
                depthshade=False, zorder=5,
            )
        periodic_images = {}
        for sid, ((u, v), shift) in enumerate(zip(endpoints, shifts)):
            start_frac = np.asarray(key_order[int(u)], dtype=np.float64)
            end_frac = np.asarray(key_order[int(v)], dtype=np.float64) + shift
            start = cell_matrix @ start_frac
            end = cell_matrix @ end_frac
            plot_points.extend([start, end])
            ax_fund.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                c=STRUT_COLOR,
                lw=LEFT_3D_STRUT_WIDTH,
                alpha=1.0,
                zorder=1,
            )
            if np.any(shift != 0):
                image_key = (int(v), tuple(int(x) for x in shift))
                if image_key not in periodic_images:
                    periodic_images[image_key] = end
                    end_color = key2col[key_order[int(v)]]
                    ax_fund.scatter(
                        *end,
                        s=LEFT_NODE_SIZE,
                        marker="o",
                        facecolors="none",
                        edgecolors=[end_color],
                        linewidths=1.3,
                        depthshade=False,
                        zorder=5,
                    )
            mid = 0.5 * (start + end)
            ax_fund.text(
                *mid, f"s{sid}", fontsize=LEFT_LABEL_SIZE,
                ha="center", va="center",
                zorder=10,
            )
        coords = np.asarray(plot_points, dtype=np.float64)
        lo = coords.min(axis=0) - 0.05
        hi = coords.max(axis=0) + 0.05
        span = max(float(np.max(hi - lo)), 1.0)
        center = 0.5 * (lo + hi)
        ax_fund.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax_fund.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax_fund.set_zlim(center[2] - span / 2, center[2] + span / 2)
        for jid, xyz in enumerate(fundamental_points):
            _draw_3d_joint_label(
                ax_fund,
                xyz,
                f"j{jid}",
                LEFT_LABEL_SIZE,
            )
        ax_fund.set_title(
            "Fundamental Periodic Structure", fontsize=LEFT_TITLE_SIZE
        )

        # (c) joint graph
        ax_jg = fig.add_subplot(gs[2, 0])
        G_jg = _build_joint_graph_nx(graph, key2col)
        pos_jg = _kk_or_spring_layout(G_jg, scale=1.0)
        nx.draw_networkx_edges(G_jg, pos_jg, ax=ax_jg, edge_color='k', alpha=0.7)
        phys_nodes = [n for n in G_jg if not G_jg.nodes[n].get('is_cut', False)]
        cut_nodes = [n for n in G_jg if G_jg.nodes[n].get('is_cut', False)]
        nx.draw_networkx_nodes(
            G_jg, pos_jg, nodelist=phys_nodes,
            node_color=[G_jg.nodes[n]['color'] for n in phys_nodes],
            node_size=LEFT_NODE_SIZE,
        )
        if cut_nodes:
            nx.draw_networkx_nodes(
                G_jg, pos_jg, nodelist=cut_nodes,
                node_color='k', node_shape='x',
                node_size=LEFT_CUT_NODE_SIZE, linewidths=1.8,
            )
        nx.draw_networkx_labels(
            G_jg, pos_jg,
            labels={n: G_jg.nodes[n]['label'] for n in G_jg},
            font_size=LEFT_LABEL_SIZE,
        )
        ax_jg.axis('off'); ax_jg.set_aspect('equal')
        ax_jg.set_title("Joint Graph", fontsize=LEFT_TITLE_SIZE)

        # (d) strut graph
        ax_sg = fig.add_subplot(gs[3, 0])
        G_sg = _build_strut_graph_nx(graph)
        pos_sg = _kk_or_spring_layout(G_sg, scale=1.0)
        nx.draw_networkx_edges(G_sg, pos_sg, ax=ax_sg, edge_color='k', alpha=0.7)
        nx.draw_networkx_nodes(
            G_sg, pos_sg,
            node_color=[STRUT_COLOR] * G_sg.number_of_nodes(),
            node_shape="s",
            node_size=LEFT_NODE_SIZE,
        )
        nx.draw_networkx_labels(
            G_sg, pos_sg,
            labels={n: G_sg.nodes[n]['label'] for n in G_sg},
            font_size=LEFT_LABEL_SIZE,
            font_color="black",
        )
        multiplicity_labels = {
            (u, v): f"x{int(attrs.get('multiplicity', 1))}"
            for u, v, attrs in G_sg.edges(data=True)
            if int(attrs.get('multiplicity', 1)) > 1
        }
        if multiplicity_labels:
            nx.draw_networkx_edge_labels(
                G_sg, pos_sg, edge_labels=multiplicity_labels,
                ax=ax_sg,
                font_size=7,
                bbox={
                    "facecolor": "none",
                    "edgecolor": "none",
                    "alpha": 0.0,
                    "pad": 0.0,
                },
            )
        ax_sg.axis('off'); ax_sg.set_aspect('equal')
        ax_sg.set_title("Strut Graph", fontsize=LEFT_TITLE_SIZE)

        # (e) bipartite graph
        ax_bg = fig.add_subplot(gs[:, 1])
        G_bg = _build_bipartite_graph_nx(graph, key2col)
        s_nodes = [n for n, d in G_bg.nodes(data=True) if d['bipartite'] == 0]
        j_nodes = [n for n, d in G_bg.nodes(data=True) if d['bipartite'] == 1]
        j_phys_nodes = [n for n in j_nodes if not G_bg.nodes[n].get('is_cut', False)]
        j_cut_nodes = [n for n in j_nodes if G_bg.nodes[n].get('is_cut', False)]
        pos_bg = {}
        ys = np.linspace(0, 1, len(s_nodes) + 2)[1:-1]
        for i, n in enumerate(s_nodes): pos_bg[n] = (0, ys[i])
        ys = np.linspace(0, 1, len(j_nodes) + 2)[1:-1]
        for i, n in enumerate(j_nodes): pos_bg[n] = (1, ys[i])
        edge_colors = [
            (G_bg.nodes[u]['color'] if G_bg.nodes[u]['bipartite'] == 1 else G_bg.nodes[v]['color'])
            for u, v in G_bg.edges()
        ]
        nx.draw_networkx_edges(
            G_bg,
            pos_bg,
            ax=ax_bg,
            edge_color=edge_colors,
            width=3.0,
            alpha=0.7,
            # edge key 1 marks the second incidence of a periodic self-loop
            # opposite curvature keeps both parallel incidences visible
            connectionstyle=["arc3,rad=0.06", "arc3,rad=-0.06"],
        )
        nx.draw_networkx_nodes(
            G_bg, pos_bg, nodelist=s_nodes,
            node_color=[STRUT_COLOR] * len(s_nodes),
            node_shape='s',
            node_size=BIPARTITE_STRUT_NODE_SIZE,
        )
        nx.draw_networkx_nodes(
            G_bg, pos_bg, nodelist=j_phys_nodes,
            node_color=[G_bg.nodes[n]['color'] for n in j_phys_nodes],
            node_shape='o',
            node_size=BIPARTITE_JOINT_NODE_SIZE,
        )
        if j_cut_nodes:
            nx.draw_networkx_nodes(
                G_bg, pos_bg, nodelist=j_cut_nodes,
                node_color='k',
                node_shape='x',
                node_size=BIPARTITE_CUT_NODE_SIZE,
                linewidths=1.8 * BIPARTITE_NODE_LINEAR_SCALE,
            )
        nx.draw_networkx_labels(
            G_bg,
            pos_bg,
            labels={node: node for node in s_nodes},
            font_size=BIPARTITE_LABEL_SIZE,
            font_color="black",
        )
        nx.draw_networkx_labels(
            G_bg,
            pos_bg,
            labels={node: node for node in j_nodes},
            font_size=BIPARTITE_LABEL_SIZE,
            font_color="black",
        )
        ax_bg.axis('off')
        ax_bg.set_title("Bipartite Graph", fontsize=10.5)

        fig.subplots_adjust(left=0.015, right=0.99, bottom=0.015, top=0.975)
        plt.show()  # script.py replaces this call to save the figure as a pdf
