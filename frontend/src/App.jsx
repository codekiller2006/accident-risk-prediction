import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";

const API_BASE = "http://localhost:8000";
const AUSTIN_CENTER = [30.2672, -97.7431];
const DAYS = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
];
const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

// Fallback placement only if a segment is missing from road_coordinates.csv
function getApproxCoords(segmentId) {
  const idStr = String(segmentId);
  let h = 2166136261;
  for (let i = 0; i < idStr.length; i++) {
    h ^= idStr.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const latOffset = (((h >>> 0) % 10000) / 10000 - 0.5) * 0.14;
  const lngOffset = ((((h >>> 12) & 0xffff) % 10000) / 10000 - 0.5) * 0.16;
  return [AUSTIN_CENTER[0] + latOffset, AUSTIN_CENTER[1] + lngOffset];
}

function resolveCoords(road) {
  if (
    road.has_exact_coords &&
    typeof road.lat === "number" &&
    typeof road.lon === "number"
  ) {
    return [road.lat, road.lon];
  }
  return getApproxCoords(road.road_segment_id);
}

function getRoadLabel(road) {
  if (!road) return "";
  if (road.road_name && road.road_name.trim()) {
    return road.road_name.trim();
  }
  return `Segment ${road.road_segment_id}`;
}

function getTierColor(tier) {
  switch (tier) {
    case "VERY HIGH":
      return "#ef4444";
    case "HIGH":
      return "#f97316";
    case "MEDIUM":
      return "#eab308";
    default:
      return "#22c55e";
  }
}

function formatEnglishList(items) {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function describeShapFactor(featureName, road, hour, dayOfWeek, month) {
  if (featureName.startsWith("highway_")) {
    const rawType = featureName.replace("highway_", "");
    const cleanType = rawType.replace(/_/g, " ");
    return road.highway === rawType
      ? `its classification as a ${cleanType}`
      : `not being a ${cleanType}`;
  }

  const isRush =
    dayOfWeek < 5 && [7, 8, 9, 16, 17, 18].includes(Number(hour));

  switch (featureName) {
    case "length":
      return `its segment length (${road.length} m)`;
    case "maxspeed":
      return `its ${road.maxspeed} mph speed limit`;
    case "hour":
      return `the ${String(hour).padStart(2, "0")}:00 time window`;
    case "is_rush_hour":
      return isRush ? "weekday rush-hour traffic" : "off-peak traffic hours";
    case "is_weekend":
      return dayOfWeek >= 5
        ? "weekend traffic patterns"
        : "weekday commuter traffic";
    case "day_of_week":
      return `${DAYS[dayOfWeek]} traffic patterns`;
    case "month":
      return `seasonal trends in ${MONTHS[month - 1]}`;
    case "lanes":
      return `having ${road.lanes} lane${Number(road.lanes) === 1 ? "" : "s"}`;
    case "oneway":
      return road.oneway ? "being a one-way corridor" : "two-way traffic flow";
    case "start_node_centrality":
      return "intersection network connectivity";
    default:
      return featureName.replace(/_/g, " ");
  }
}

function buildShapNarrative(explanation, road, hour, dayOfWeek, month) {
  if (!explanation || !road || !explanation.top_factors?.length) return null;

  const subject =
    road.road_name && road.road_name.trim()
      ? `${road.road_name.trim()} (segment ${road.road_segment_id})`
      : `Segment ${road.road_segment_id}`;

  const raisers = explanation.top_factors
    .filter((f) => f.shap_value > 0.02)
    .slice(0, 3)
    .map((f) => describeShapFactor(f.feature, road, hour, dayOfWeek, month));

  const reducers = explanation.top_factors
    .filter((f) => f.shap_value < -0.02)
    .slice(0, 2)
    .map((f) => describeShapFactor(f.feature, road, hour, dayOfWeek, month));

  const mainLine =
    raisers.length > 0
      ? `${subject} has a ${explanation.risk_tier} relative risk score (${explanation.risk_score}%), pushed up primarily by ${formatEnglishList(
          raisers
        )}.`
      : `${subject} has a ${explanation.risk_tier} relative risk score (${explanation.risk_score}%), with no single feature strongly elevating risk.`;

  const offsetLine =
    reducers.length > 0
      ? `Risk is partially offset by ${formatEnglishList(reducers)}.`
      : `None of the top model factors meaningfully reduce risk during this time slot.`;

  return `${mainLine} ${offsetLine}`;
}

function MapBoundsController({ hotspots }) {
  const map = useMap();

  useEffect(() => {
    if (!hotspots || hotspots.length === 0) return;
    const points = hotspots.map((r) => r.coords);
    if (points.length === 1) {
      map.setView(points[0], 13);
    } else {
      map.fitBounds(points, { padding: [32, 32], maxZoom: 13 });
    }
  }, [hotspots, map]);

  return null;
}

export default function App() {
  // Global filters
  const [hour, setHour] = useState(17);
  const [dayOfWeek, setDayOfWeek] = useState(4);
  const [month, setMonth] = useState(10);
  const [highwayFilter, setHighwayFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  // API metadata & data
  const [highwayTypes, setHighwayTypes] = useState([]);
  const [coordinatesLoaded, setCoordinatesLoaded] = useState(false);
  const [mappedRoadsCount, setMappedRoadsCount] = useState(0);
  const [summary, setSummary] = useState(null);
  const [hotspots, setHotspots] = useState([]);
  const [selectedRoad, setSelectedRoad] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);

  // What-If Simulation state
  const [simSpeed, setSimSpeed] = useState(45);
  const [simRushHour, setSimRushHour] = useState(1);
  const [simLanes, setSimLanes] = useState(2);
  const [simHour, setSimHour] = useState(17);
  const [simResult, setSimResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  // 1. Load metadata once on mount
  useEffect(() => {
    axios
      .get(`${API_BASE}/`)
      .then((res) => {
        setHighwayTypes(res.data.highway_types || []);
        setCoordinatesLoaded(Boolean(res.data.coordinates_loaded));
        setMappedRoadsCount(Number(res.data.mapped_roads_count || 0));
        setApiError("");
      })
      .catch(() => {
        setApiError(
          "Cannot reach FastAPI backend on http://localhost:8000. Ensure Uvicorn is running and localhost is bypassed in proxy settings."
        );
      });
  }, []);

  const buildPayload = (road, h, dow, m, overrides = {}) => {
    const targetHour =
      overrides.hour !== undefined ? Number(overrides.hour) : Number(h);
    const defaultRush =
      dow < 5 && [7, 8, 9, 16, 17, 18].includes(targetHour) ? 1 : 0;
    return {
      hour: targetHour,
      day_of_week: Number(dow),
      month: Number(m),
      is_weekend: dow >= 5 ? 1 : 0,
      is_rush_hour:
        overrides.is_rush_hour !== undefined
          ? Number(overrides.is_rush_hour)
          : defaultRush,
      length: Number(road.length),
      highway: road.highway,
      lanes:
        overrides.lanes !== undefined
          ? Number(overrides.lanes)
          : Number(road.lanes),
      maxspeed:
        overrides.maxspeed !== undefined
          ? Number(overrides.maxspeed)
          : Number(road.maxspeed),
      oneway: Number(road.oneway),
      start_node_centrality: Number(road.start_node_centrality || 0),
    };
  };

  // 2. Inspect selected road: fetch SHAP explanation + 24-hour risk curve concurrently
  const inspectRoad = async (road, h = hour, dow = dayOfWeek, m = month) => {
    setSelectedRoad(road);
    const rush = dow < 5 && [7, 8, 9, 16, 17, 18].includes(h) ? 1 : 0;
    setSimSpeed(road.maxspeed);
    setSimRushHour(rush);
    setSimLanes(Math.max(1, Math.min(6, Math.round(road.lanes || 2))));
    setSimHour(h);
    setSimResult(null);

    const payload = buildPayload(road, h, dow, m);
    try {
      const [explainRes, timelineRes] = await Promise.all([
        axios.post(`${API_BASE}/explain`, payload),
        axios.get(
          `${API_BASE}/timeline/${encodeURIComponent(road.road_segment_id)}`,
          {
            params: { day_of_week: dow, month: m },
          }
        ),
      ]);
      setExplanation(explainRes.data);
      setTimeline(timelineRes.data.timeline || []);
    } catch (err) {
      console.error("Inspection error:", err);
    }
  };

  // 3. Fetch city hotspots when time or road type changes
  useEffect(() => {
    let active = true;
    const fetchHotspots = async () => {
      setLoading(true);
      try {
        const res = await axios.get(`${API_BASE}/hotspots`, {
          params: {
            hour,
            day_of_week: dayOfWeek,
            month,
            top_n: 30,
            highway: highwayFilter,
          },
        });
        if (!active) return;
        setApiError("");
        setSummary(res.data.summary);
        const enriched = res.data.hotspots.map((r) => ({
          ...r,
          coords: resolveCoords(r),
        }));
        setHotspots(enriched);
        if (enriched.length > 0) {
          inspectRoad(enriched[0], hour, dayOfWeek, month);
        } else {
          setSelectedRoad(null);
          setExplanation(null);
          setTimeline([]);
        }
      } catch {
        if (active) {
          setApiError("Failed to fetch hotspots from FastAPI. Check Terminal 1.");
        }
      } finally {
        if (active) setLoading(false);
      }
    };
    fetchHotspots();
    return () => {
      active = false;
    };
  }, [hour, dayOfWeek, month, highwayFilter]);

  // 4. Run What-If Simulation
  const runSimulation = async () => {
    if (!selectedRoad) return;
    setSimLoading(true);
    const baseline = buildPayload(selectedRoad, hour, dayOfWeek, month);
    try {
      const res = await axios.post(`${API_BASE}/simulate`, {
        baseline,
        overrides: {
          maxspeed: Number(simSpeed),
          is_rush_hour: Number(simRushHour),
          lanes: Number(simLanes),
          hour: Number(simHour),
        },
      });
      setSimResult(res.data);
    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      setSimLoading(false);
    }
  };

  const filteredHotspots = useMemo(() => {
    if (!searchQuery.trim()) return hotspots;
    const q = searchQuery.toLowerCase();
    return hotspots.filter(
      (r) =>
        r.road_segment_id.toLowerCase().includes(q) ||
        r.highway.toLowerCase().includes(q) ||
        (r.road_name && r.road_name.toLowerCase().includes(q))
    );
  }, [hotspots, searchQuery]);

  // SVG path generator for 24-hour curve
  const timelinePoints = useMemo(() => {
    if (!timeline.length) return "";
    return timeline
      .map((pt, idx) => {
        const x = (idx / 23) * 460 + 20;
        const y = 130 - (pt.risk_score / 100) * 110;
        return `${x},${y}`;
      })
      .join(" ");
  }, [timeline]);

  const shapNarrative = useMemo(
    () => buildShapNarrative(explanation, selectedRoad, hour, dayOfWeek, month),
    [explanation, selectedRoad, hour, dayOfWeek, month]
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0b1120",
        padding: "20px",
        color: "#e2e8f0",
      }}
    >
      <div style={{ maxWidth: "1480px", margin: "0 auto" }}>
        {/* Header Bar */}
        <header style={styles.header}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: apiError ? "#ef4444" : "#22c55e",
                  display: "inline-block",
                }}
              />
              <h1
                style={{
                  margin: 0,
                  fontSize: "20px",
                  fontWeight: 700,
                  letterSpacing: "-0.02em",
                }}
              >
                Urban Accident Risk Intelligence
              </h1>
            </div>
            <p style={{ margin: "4px 0 0", fontSize: "12px", color: "#94a3b8" }}>
              Austin Road Network • Spatio-Temporal XGBoost • SHAP Attribution •
              Scenario Simulation
            </p>
          </div>

          {/* Global Time & Road Controls */}
          <div
            style={{
              display: "flex",
              gap: "12px",
              flexWrap: "wrap",
              alignItems: "end",
            }}
          >
            <div style={styles.controlGroup}>
              <label style={styles.label}>
                HOUR ({String(hour).padStart(2, "0")}:00)
              </label>
              <input
                type="range"
                min="0"
                max="23"
                value={hour}
                onChange={(e) => setHour(Number(e.target.value))}
                style={{ width: "110px", accentColor: "#38bdf8" }}
              />
            </div>

            <div style={styles.controlGroup}>
              <label style={styles.label}>DAY</label>
              <select
                value={dayOfWeek}
                onChange={(e) => setDayOfWeek(Number(e.target.value))}
                style={styles.select}
              >
                {DAYS.map((d, idx) => (
                  <option key={d} value={idx}>
                    {d}
                  </option>
                ))}
              </select>
            </div>

            <div style={styles.controlGroup}>
              <label style={styles.label}>MONTH</label>
              <select
                value={month}
                onChange={(e) => setMonth(Number(e.target.value))}
                style={styles.select}
              >
                {MONTHS.map((m, idx) => (
                  <option key={m} value={idx + 1}>
                    {m}
                  </option>
                ))}
              </select>
            </div>

            <div style={styles.controlGroup}>
              <label style={styles.label}>ROAD CLASS</label>
              <select
                value={highwayFilter}
                onChange={(e) => setHighwayFilter(e.target.value)}
                style={styles.select}
              >
                <option value="all">All Roads</option>
                {highwayTypes.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </header>

        {apiError && (
          <div
            style={{
              background: "#450a0a",
              border: "1px solid #dc2626",
              color: "#fecaca",
              padding: "12px 16px",
              borderRadius: "8px",
              marginBottom: "16px",
              fontSize: "13px",
            }}
          >
            {apiError}
          </div>
        )}

        {/* KPI Strip */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: "14px",
            marginBottom: "18px",
          }}
        >
          <KpiCard
            title="EVALUATED SEGMENTS"
            value={summary ? summary.evaluated_roads.toLocaleString() : "—"}
            accent="#38bdf8"
          />
          <KpiCard
            title="CITYWIDE MEAN RISK"
            value={summary ? `${summary.mean_risk_score}%` : "—"}
            accent="#eab308"
          />
          <KpiCard
            title="CRITICAL HOTSPOTS (≥80%)"
            value={summary ? summary.critical_count.toLocaleString() : "—"}
            accent="#ef4444"
          />
          <KpiCard
            title="HIGH RISK SEGMENTS (60–79%)"
            value={summary ? summary.high_count.toLocaleString() : "—"}
            accent="#f97316"
          />
        </div>

        {/* Main Row: Dark Map + Ranked Hotspots */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.65fr 1fr",
            gap: "18px",
            marginBottom: "18px",
          }}
        >
          {/* Interactive Dark Map */}
          <div style={styles.card}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "10px",
              }}
            >
              <div>
                <h2 style={styles.cardTitle}>Austin Risk Map</h2>
                <span style={{ fontSize: "11px", color: "#94a3b8" }}>
                  {coordinatesLoaded
                    ? `Exact OSM street midpoints (${mappedRoadsCount.toLocaleString()} roads mapped) • Click any pin to inspect`
                    : "Click any hotspot marker to inspect SHAP drivers & 24-hour profile (approx. coordinates)"}
                </span>
              </div>
              <div
                style={{
                  display: "flex",
                  gap: "10px",
                  fontSize: "11px",
                  fontWeight: 600,
                }}
              >
                <span style={{ color: "#ef4444" }}>● Very High</span>
                <span style={{ color: "#f97316" }}>● High</span>
                <span style={{ color: "#eab308" }}>● Medium</span>
                <span style={{ color: "#22c55e" }}>● Low</span>
              </div>
            </div>
            <div
              style={{
                height: "420px",
                borderRadius: "8px",
                overflow: "hidden",
                border: "1px solid #1e293b",
              }}
            >
              <MapContainer
                center={AUSTIN_CENTER}
                zoom={11}
                scrollWheelZoom={false}
                style={{ height: "100%", width: "100%" }}
              >
                <TileLayer
                  attribution="© OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <MapBoundsController hotspots={filteredHotspots} />
                {filteredHotspots.map((road) => {
                  const isSelected =
                    selectedRoad?.road_segment_id === road.road_segment_id;
                  const color = getTierColor(road.risk_tier);
                  return (
                    <CircleMarker
                      key={road.road_segment_id}
                      center={road.coords}
                      radius={isSelected ? 12 : 8}
                      pathOptions={{
                        color: isSelected ? "#ffffff" : color,
                        weight: isSelected ? 2.5 : 1.5,
                        fillColor: color,
                        fillOpacity: 0.85,
                      }}
                      eventHandlers={{ click: () => inspectRoad(road) }}
                    >
                      <Popup>
                        <div style={{ fontSize: "12px", lineHeight: 1.5 }}>
                          <strong>{getRoadLabel(road)}</strong>
                          {road.road_name && (
                            <>
                              <br />
                              <span style={{ color: "#94a3b8", fontSize: "11px" }}>
                                ID: {road.road_segment_id}
                              </span>
                            </>
                          )}
                          <br />
                          Risk:{" "}
                          <strong style={{ color }}>
                            {road.risk_score}% ({road.risk_tier})
                          </strong>
                          <br />
                          Class: {road.highway} • {road.maxspeed} mph •{" "}
                          {road.lanes} lanes
                          <br />
                          Historical Crashes: {road.historical_accidents}
                          <br />
                          <span style={{ color: "#94a3b8", fontSize: "10px" }}>
                            {road.has_exact_coords
                              ? `OSM Midpoint: ${road.coords[0].toFixed(4)}, ${road.coords[1].toFixed(4)}`
                              : "Approx. fallback coordinate"}
                          </span>
                        </div>
                      </Popup>
                    </CircleMarker>
                  );
                })}
              </MapContainer>
            </div>
          </div>

          {/* Ranked Hotspots List */}
          <div
            style={{
              ...styles.card,
              display: "flex",
              flexDirection: "column",
              height: "485px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "10px",
              }}
            >
              <h2 style={styles.cardTitle}>Top Predicted Hotspots</h2>
              <span style={{ fontSize: "11px", color: "#94a3b8" }}>
                {filteredHotspots.length} shown
              </span>
            </div>
            <input
              type="text"
              placeholder="Search street name, segment ID, or road type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                ...styles.select,
                width: "100%",
                marginBottom: "10px",
                padding: "8px 10px",
              }}
            />
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                paddingRight: "4px",
              }}
            >
              {loading ? (
                <p style={{ fontSize: "13px", color: "#94a3b8" }}>
                  Scoring Austin road graph...
                </p>
              ) : (
                filteredHotspots.map((road, idx) => {
                  const active =
                    selectedRoad?.road_segment_id === road.road_segment_id;
                  const color = getTierColor(road.risk_tier);
                  return (
                    <div
                      key={road.road_segment_id}
                      onClick={() => inspectRoad(road)}
                      style={{
                        padding: "10px 12px",
                        borderRadius: "8px",
                        cursor: "pointer",
                        background: active ? "#1e293b" : "#0f172a",
                        border: active
                          ? "1px solid #38bdf8"
                          : "1px solid #1e293b",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        transition: "border-color 0.15s",
                      }}
                    >
                      <div style={{ minWidth: 0, paddingRight: "8px" }}>
                        <div
                          style={{
                            fontWeight: 600,
                            fontSize: "13px",
                            color: "#f8fafc",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                        >
                          #{idx + 1} • {getRoadLabel(road)}
                        </div>
                        <div
                          style={{
                            fontSize: "11px",
                            color: "#94a3b8",
                            marginTop: "2px",
                          }}
                        >
                          {road.highway} • {road.maxspeed} mph • {road.lanes}L •{" "}
                          {road.historical_accidents} past crashes
                        </div>
                      </div>
                      <span
                        style={{
                          background: `${color}22`,
                          color,
                          border: `1px solid ${color}`,
                          padding: "3px 8px",
                          borderRadius: "6px",
                          fontSize: "12px",
                          fontWeight: 700,
                          flexShrink: 0,
                        }}
                      >
                        {road.risk_score}%
                      </span>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* Bottom Row: 24-Hour Curve + SHAP + What-If Simulator */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: "18px",
          }}
        >
          {/* 1. 24-Hour Risk Profile */}
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>24-Hour Risk Forecast</h2>
            <p style={styles.subtext}>
              {selectedRoad
                ? `${getRoadLabel(selectedRoad)} across ${DAYS[dayOfWeek]} (${MONTHS[month - 1]})`
                : "Select a segment to view its hourly curve."}
            </p>
            {timeline.length > 0 && (
              <svg
                viewBox="0 0 500 155"
                style={{ width: "100%", height: "175px", marginTop: "8px" }}
              >
                {[25, 50, 75, 100].map((v) => {
                  const y = 130 - (v / 100) * 110;
                  return (
                    <g key={v}>
                      <line
                        x1="20"
                        y1={y}
                        x2="480"
                        y2={y}
                        stroke="#1e293b"
                        strokeDasharray="3,3"
                      />
                      <text x="2" y={y + 3} fill="#64748b" fontSize="9">
                        {v}%
                      </text>
                    </g>
                  );
                })}
                <polyline
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="2.5"
                  points={timelinePoints}
                />
                {timeline[hour] && (
                  <circle
                    cx={(hour / 23) * 460 + 20}
                    cy={130 - (timeline[hour].risk_score / 100) * 110}
                    r="5"
                    fill="#ef4444"
                    stroke="#fff"
                    strokeWidth="1.5"
                  />
                )}
                {[0, 6, 12, 18, 23].map((h) => (
                  <text
                    key={h}
                    x={(h / 23) * 460 + 12}
                    y="148"
                    fill="#94a3b8"
                    fontSize="10"
                  >
                    {String(h).padStart(2, "0")}:00
                  </text>
                ))}
              </svg>
            )}
            {selectedRoad && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "8px",
                  marginTop: "8px",
                  fontSize: "12px",
                  background: "#0f172a",
                  padding: "10px",
                  borderRadius: "6px",
                }}
              >
                <div>
                  Road Class:{" "}
                  <strong>{selectedRoad.highway.replace(/_/g, " ")}</strong>
                </div>
                <div>
                  Length: <strong>{selectedRoad.length} m</strong>
                </div>
                <div>
                  One-Way: <strong>{selectedRoad.oneway ? "Yes" : "No"}</strong>
                </div>
                <div>
                  Past Crashes:{" "}
                  <strong>{selectedRoad.historical_accidents}</strong>
                </div>
              </div>
            )}
          </div>

          {/* 2. SHAP Explainability + Plain-English Summary */}
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>SHAP Risk Attribution</h2>
            <p style={styles.subtext}>
              {explanation
                ? `Model feature contributions • Base log-odds: ${explanation.base_value}`
                : "Select a road segment to inspect feature impact."}
            </p>

            {shapNarrative && (
              <div
                style={{
                  background: "#0f172a",
                  borderLeft: "3px solid #38bdf8",
                  borderRadius: "6px",
                  padding: "10px 12px",
                  marginTop: "10px",
                  fontSize: "12px",
                  lineHeight: 1.5,
                  color: "#e2e8f0",
                }}
              >
                <div
                  style={{
                    fontSize: "10px",
                    fontWeight: 700,
                    color: "#38bdf8",
                    letterSpacing: "0.05em",
                    marginBottom: "4px",
                  }}
                >
                  IN PLAIN ENGLISH
                </div>
                {shapNarrative}
              </div>
            )}

            {explanation && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "9px",
                  marginTop: "12px",
                }}
              >
                {explanation.top_factors.map((f) => {
                  const isPos = f.shap_value > 0;
                  const pct = Math.min(
                    (Math.abs(f.shap_value) / 1.5) * 100,
                    100
                  );
                  return (
                    <div key={f.feature} style={{ fontSize: "12px" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "3px",
                        }}
                      >
                        <span style={{ color: "#cbd5e1", fontWeight: 500 }}>
                          {f.feature}
                        </span>
                        <span
                          style={{
                            color: isPos ? "#f87171" : "#60a5fa",
                            fontWeight: 700,
                          }}
                        >
                          {isPos ? `+${f.shap_value}` : f.shap_value}
                        </span>
                      </div>
                      <div
                        style={{
                          background: "#0f172a",
                          height: "7px",
                          borderRadius: "4px",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            background: isPos ? "#ef4444" : "#3b82f6",
                            borderRadius: "4px",
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* 3. What-If Scenario Simulator */}
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>What-If Scenario Simulator</h2>
            <p style={styles.subtext}>
              Model-estimated risk sensitivity under modified road & traffic
              parameters:
            </p>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "12px",
                marginTop: "10px",
              }}
            >
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "12px",
                  }}
                >
                  <span>Speed Limit Override</span>
                  <strong>{simSpeed} mph</strong>
                </div>
                <input
                  type="range"
                  min="15"
                  max="75"
                  step="5"
                  value={simSpeed}
                  onChange={(e) => setSimSpeed(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "#38bdf8" }}
                />
              </div>

              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "12px",
                  }}
                >
                  <span>Simulated Hour</span>
                  <strong>{String(simHour).padStart(2, "0")}:00</strong>
                </div>
                <input
                  type="range"
                  min="0"
                  max="23"
                  value={simHour}
                  onChange={(e) => setSimHour(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "#38bdf8" }}
                />
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "10px",
                }}
              >
                <div>
                  <label style={styles.label}>TRAFFIC REGIME</label>
                  <select
                    value={simRushHour}
                    onChange={(e) => setSimRushHour(Number(e.target.value))}
                    style={{ ...styles.select, width: "100%" }}
                  >
                    <option value={1}>Rush Hour</option>
                    <option value={0}>Off-Peak</option>
                  </select>
                </div>
                <div>
                  <label style={styles.label}>LANES</label>
                  <select
                    value={simLanes}
                    onChange={(e) => setSimLanes(Number(e.target.value))}
                    style={{ ...styles.select, width: "100%" }}
                  >
                    {[1, 2, 3, 4, 5, 6].map((l) => (
                      <option key={l} value={l}>
                        {l} Lane{l > 1 ? "s" : ""}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                onClick={runSimulation}
                disabled={simLoading || !selectedRoad}
                style={styles.button}
              >
                {simLoading ? "Simulating..." : "Run Scenario Estimate"}
              </button>

              {simResult && (
                <div
                  style={{
                    background: "#0f172a",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                    padding: "12px",
                    fontSize: "13px",
                  }}
                >
                  <div
                    style={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <span style={{ color: "#94a3b8" }}>Baseline Risk:</span>
                    <strong>
                      {simResult.baseline.risk_score}% (
                      {simResult.baseline.risk_tier})
                    </strong>
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginTop: "4px",
                    }}
                  >
                    <span style={{ color: "#94a3b8" }}>Simulated Risk:</span>
                    <strong>
                      {simResult.simulated.risk_score}% (
                      {simResult.simulated.risk_tier})
                    </strong>
                  </div>
                  <div
                    style={{
                      marginTop: "8px",
                      paddingTop: "8px",
                      borderTop: "1px solid #1e293b",
                      display: "flex",
                      justifyContent: "space-between",
                      fontWeight: 700,
                      color:
                        simResult.risk_score_change <= 0 ? "#4ade80" : "#f87171",
                    }}
                  >
                    <span>Net Change:</span>
                    <span>
                      {simResult.risk_score_change > 0
                        ? `+${simResult.risk_score_change}%`
                        : `${simResult.risk_score_change}%`}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function KpiCard({ title, value, accent }) {
  return (
    <div
      style={{
        ...styles.card,
        borderTop: `3px solid ${accent}`,
        padding: "14px 16px",
      }}
    >
      <div
        style={{
          fontSize: "11px",
          color: "#94a3b8",
          fontWeight: 600,
          letterSpacing: "0.04em",
        }}
      >
        {title}
      </div>
      <div
        style={{
          fontSize: "24px",
          fontWeight: 700,
          marginTop: "4px",
          color: "#f8fafc",
        }}
      >
        {value}
      </div>
    </div>
  );
}

const styles = {
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    flexWrap: "wrap",
    gap: "16px",
    background: "#111827",
    border: "1px solid #1e293b",
    padding: "16px 20px",
    borderRadius: "10px",
    marginBottom: "18px",
  },
  card: {
    background: "#111827",
    border: "1px solid #1e293b",
    borderRadius: "10px",
    padding: "16px",
  },
  cardTitle: {
    margin: 0,
    fontSize: "15px",
    fontWeight: 700,
    color: "#f8fafc",
  },
  subtext: {
    margin: "4px 0 0",
    fontSize: "12px",
    color: "#94a3b8",
  },
  controlGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  label: {
    fontSize: "10px",
    fontWeight: 700,
    color: "#94a3b8",
    letterSpacing: "0.05em",
  },
  select: {
    background: "#0f172a",
    color: "#f8fafc",
    border: "1px solid #334155",
    borderRadius: "6px",
    padding: "6px 10px",
    fontSize: "12px",
    outline: "none",
  },
  button: {
    background: "linear-gradient(135deg, #2563eb, #0284c7)",
    color: "#ffffff",
    border: "none",
    borderRadius: "6px",
    padding: "10px",
    fontWeight: 700,
    fontSize: "13px",
    cursor: "pointer",
  },
};