import json
import streamlit.components.v1 as components


def extract_fraud_topology_data(target_file: str, xai_links: list, raw_relations: list = None) -> dict:
    """
    Extracts and structures node and edge data from resolved human-readable IDs
    without rendering any Streamlit UI canvas elements.
    """
    nodes = []
    edges = []
    added_nodes = set()

    combined_links = []
    seen_pairs = set()

    # Safely cast target_file to string to ensure exact matching below
    target_file = str(target_file)

    # Prioritize XAI attribution links
    for link in xai_links:
        src = str(link.get('source_id', 'Unknown'))
        tgt = str(link.get('target_id', 'Unknown'))
        relation = link.get('relation', 'linked_to')
        weight = link.get('importance_score', 50.0)

        pair_key = (src, tgt, relation)
        seen_pairs.add(pair_key)
        combined_links.append({
            "source_id": src,
            "target_id": tgt,
            "relation": relation,
            "importance_score": weight,
            "source_type": link.get('source_type', 'user'),
            "target_type": link.get('target_type', 'company')
        })

    # Merge neighbor relation logs
    if raw_relations:
        for rel in raw_relations:
            src = str(rel.get('source_id', 'Unknown'))
            tgt = str(rel.get('target_id', 'Unknown'))
            relation = rel.get('relation', 'linked_to')

            pair_key = (src, tgt, relation)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                combined_links.append({
                    "source_id": src,
                    "target_id": tgt,
                    "relation": relation,
                    "importance_score": rel.get('importance_score', 20.0),
                    "source_type": rel.get('source_type', 'user'),
                    "target_type": rel.get('target_type', 'company')
                })

    for link in combined_links:
        src = link["source_id"]
        tgt = link["target_id"]
        relation = link["relation"]
        weight = link["importance_score"]

        def add_node_if_missing(node_id, node_type):
            if node_id not in added_nodes:
                # Assign dynamic structural metadata styles based on resolved identity class
                if node_type == "file" or node_id == target_file:
                    label = f"File {node_id}" if "File" not in node_id else node_id
                    color = "#ef4444"  # Red
                    shape = "square"
                elif node_type == "company":
                    label = f"Company: {node_id}"
                    color = "#059669"  # Emerald Green
                    shape = "triangle"
                else:
                    label = f"User: {node_id}"
                    color = "#1e3a8a"  # Navy Blue
                    shape = "dot"

                nodes.append({
                    "id": node_id,
                    "label": label,
                    "type": node_type,
                    "color": color,
                    "shape": shape
                })
                added_nodes.add(node_id)

        add_node_if_missing(src, link["source_type"])
        add_node_if_missing(tgt, link["target_type"])

        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation,
            "importance_score": weight,
            "label": f"{relation} ({weight}%)",
            "width": round(1 + (weight / 40), 2)
        })

    return {
        "nodes": nodes,
        "edges": edges
    }


def render_fraud_topology(target_file: str, xai_links: list, raw_relations: list = None, height: int = 460):
    """
    Renders the extracted node/edge topology as an interactive vis-network
    graph embedded via components.html -- same CDN-embed pattern the rest
    of the dashboard already uses for Chart.js panels.
    """
    data = extract_fraud_topology_data(target_file, xai_links, raw_relations)

    if not data["nodes"]:
        components.html(
            "<div style='padding:2.5rem; text-align:center; color:#94a3b8; font-size:0.9rem;'>"
            "No graph relations available to render for this transaction.</div>",
            height=120,
        )
        return

    vis_nodes = [
        {
            "id": n["id"],
            "label": n["label"],
            "color": n["color"],
            "shape": n["shape"],
            "font": {"color": "#0f172a", "size": 13},
        }
        for n in data["nodes"]
    ]
    vis_edges = [
        {
            "from": e["source"],
            "to": e["target"],
            "label": e["label"],
            "width": e["width"],
            "arrows": "to",
            "color": {"color": "#94a3b8", "highlight": "#16a34a"},
            "font": {"size": 10, "align": "middle", "color": "#475569"},
        }
        for e in data["edges"]
    ]

    html = f"""
    <div id="fraud-graph" style="width:100%; height:{height}px; border:1px solid #e2e8f0; border-radius:8px; background:#ffffff;"></div>
    <div id="fraud-graph-error" style="display:none; padding:1rem; color:#991b1b; font-size:0.85rem; font-family:monospace; white-space:pre-wrap;"></div>
    <!-- NOTE: must be the "standalone" UMD build -- the plain vis-network.min.js
         (e.g. the old cdnjs path) does NOT bundle vis-data, so vis.DataSet is
         undefined there and this whole script fails silently, leaving a blank div. -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
    <script>
      try {{
        const container = document.getElementById('fraud-graph');
        const data = {{
          nodes: new vis.DataSet({json.dumps(vis_nodes)}),
          edges: new vis.DataSet({json.dumps(vis_edges)})
        }};
        const options = {{
          physics: {{ stabilization: true, barnesHut: {{ gravitationalConstant: -3000, springLength: 140 }} }},
          interaction: {{ hover: true, zoomView: true, dragView: true }},
          layout: {{ improvedLayout: true }}
        }};
        new vis.Network(container, data, options);
      }} catch (err) {{
        const errBox = document.getElementById('fraud-graph-error');
        errBox.style.display = 'block';
        errBox.textContent = 'Graph render failed: ' + err.message;
        console.error('fraud-graph render error:', err);
      }}
    </script>
    """
    components.html(html, height=height + 20)