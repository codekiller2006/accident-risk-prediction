from pathlib import Path
import osmnx as ox
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "ml_final_features.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "road_coordinates.csv"

OVERPASS_MIRRORS = [
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.osm.jp/api",
    "https://overpass-api.de/api",
]


def extract_name(edge_data: dict) -> str:
    raw = edge_data.get("name")
    if isinstance(raw, list):
        return " / ".join(str(x) for x in raw if x)
    if pd.notna(raw) and str(raw).strip():
        return str(raw).strip()
    return ""


def midpoint_from_edge(G, u: int, v: int, edge_data: dict) -> tuple[float, float]:
    geom = edge_data.get("geometry")
    if geom is not None:
        pt = geom.interpolate(0.5, normalized=True)
        return round(float(pt.y), 6), round(float(pt.x), 6)
    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2.0
    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2.0
    return round(float(lat), 6), round(float(lon), 6)


def download_austin_graph():
    ox.settings.use_cache = False
    ox.settings.overpass_rate_limit = False
    ox.settings.requests_timeout = 180

    last_err = None
    for mirror in OVERPASS_MIRRORS:
        print(f"Trying Overpass mirror: {mirror} ...")
        ox.settings.overpass_url = mirror
        try:
            return ox.graph_from_place("Austin, Texas, USA", network_type="drive")
        except Exception as exc:
            print(f"  Mirror failed ({type(exc).__name__}): {exc}")
            last_err = exc

    raise RuntimeError(
        "All Overpass mirrors failed. Try switching to a mobile hotspot for 30 seconds and re-running."
    ) from last_err


def main() -> None:
    print("Loading unique road_segment_id values from ml_final_features.csv...")
    df = pd.read_csv(DATA_PATH, usecols=["road_segment_id"])
    seg_ids = df["road_segment_id"].astype(str).dropna().unique().tolist()
    print(f"Found {len(seg_ids):,} unique road segments.")

    print("Downloading Austin, Texas drivable road network from OpenStreetMap...")
    G = download_austin_graph()
    print(f"Downloaded graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges.")

    rows = []
    exact_edge_hits = 0
    node_fallback_hits = 0
    misses = 0

    for seg_id in seg_ids:
        parts = seg_id.split("_")
        if len(parts) < 3:
            misses += 1
            continue

        try:
            u, v, k = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            misses += 1
            continue

        lat = lon = None
        road_name = ""

        # 1. Exact (u, v, k) edge match
        if G.has_edge(u, v, k):
            edge_data = G.get_edge_data(u, v, k)
            lat, lon = midpoint_from_edge(G, u, v, edge_data)
            road_name = extract_name(edge_data)
            exact_edge_hits += 1
        # 2. Same (u, v) pair with a different parallel key
        elif G.has_edge(u, v):
            first_key = next(iter(G.get_edge_data(u, v)))
            edge_data = G.get_edge_data(u, v, first_key)
            lat, lon = midpoint_from_edge(G, u, v, edge_data)
            road_name = extract_name(edge_data)
            exact_edge_hits += 1
        # 3. Both intersection nodes exist in current OSM graph
        elif u in G.nodes and v in G.nodes:
            lat = round((float(G.nodes[u]["y"]) + float(G.nodes[v]["y"])) / 2.0, 6)
            lon = round((float(G.nodes[u]["x"]) + float(G.nodes[v]["x"])) / 2.0, 6)
            node_fallback_hits += 1
        # 4. At least one endpoint node exists
        elif u in G.nodes:
            lat = round(float(G.nodes[u]["y"]), 6)
            lon = round(float(G.nodes[u]["x"]), 6)
            node_fallback_hits += 1
        elif v in G.nodes:
            lat = round(float(G.nodes[v]["y"]), 6)
            lon = round(float(G.nodes[v]["x"]), 6)
            node_fallback_hits += 1
        else:
            misses += 1
            continue

        rows.append(
            {
                "road_segment_id": seg_id,
                "lat": lat,
                "lon": lon,
                "road_name": road_name,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    matched = len(out_df)
    pct = (matched / len(seg_ids) * 100.0) if seg_ids else 0.0
    named = int((out_df["road_name"].astype(str).str.len() > 0).sum()) if matched else 0

    print("\n" + "=" * 55)
    print(f"Saved: {OUT_PATH}")
    print(f"Matched coordinates : {matched:,} / {len(seg_ids):,} ({pct:.1f}%)")
    print(f"  - Exact edge hits : {exact_edge_hits:,}")
    print(f"  - Node fallbacks  : {node_fallback_hits:,}")
    print(f"  - Unmatched       : {misses:,}")
    print(f"Segments with names : {named:,}")
    print("=" * 55)


if __name__ == "__main__":
    main()