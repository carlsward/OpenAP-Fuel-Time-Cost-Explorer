import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px  # pip install plotly

# OpenAP – officiell Python-implementation
# Docs/exempel: https://openap.dev/  (FuelFlow/Emission/prop)
from openap import FuelFlow, Emission, prop  # pip install openap


# ---------- Hjälpfunktioner ----------
def kts_to_nm_per_s(kts: float) -> float:
    return kts / 3600.0  # 1 knot = 1 NM per timme


def ftmin_to_fts(vs_ftmin: float) -> float:
    return vs_ftmin / 60.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ---- Plot-helper: titel + lock/unlock ----
def plot_with_lock(fig, title: str, key: str):
    # Vi visar titel till vänster och en toggle till höger
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.markdown(f"### {title}")
    with c2:
        unlocked = st.toggle("🔓 Lås upp", value=False, key=key, help="Tillåt zoom/pan och verktygsrad")

    # Låst = helt statisk figur (ingen pan/zoom). Upplåst = interaktiv + verktygsrad.
    config = {"staticPlot": not unlocked, "displayModeBar": unlocked}
    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=config)



@dataclass
class SegmentResult:
    name: str
    seconds: float
    distance_nm: float
    fuel_kg: float
    co2_kg: float


# ---------- Simuleringskärna (oförändrad) ----------
def simulate_flight(
    ac_code: str,
    mass_takeoff_kg: float,
    trip_distance_nm: float,
    cruise_alt_ft: float,
    climb_tas_kts: float = 300.0,
    cruise_tas_kts: float = 450.0,
    descent_tas_kts: float = 300.0,
    climb_vs_ftmin: float = 1800.0,
    descent_vs_ftmin: float = -1500.0,
    dt: float = 1.0,
) -> Tuple[List[SegmentResult], pd.DataFrame]:

    ac = ac_code.upper()
    fuelflow = FuelFlow(ac=ac)
    emission = Emission(ac=ac)

    t = 0.0
    alt_ft = 0.0
    mass_kg = float(mass_takeoff_kg)
    dist_nm = 0.0

    rows = []
    segs: List[SegmentResult] = []

    # grov nedstigningsdistans
    descent_time_s_est = (cruise_alt_ft - 0.0) / abs(descent_vs_ftmin) * 60.0
    descent_dist_nm_est = kts_to_nm_per_s(descent_tas_kts) * descent_time_s_est

    # CLIMB
    climb_seconds = climb_fuel = climb_co2_g = 0.0
    while alt_ft < cruise_alt_ft and dist_nm < max(1e-6, trip_distance_nm - descent_dist_nm_est):
        vs = climb_vs_ftmin
        tas = climb_tas_kts
        ff = float(fuelflow.enroute(mass=mass_kg, tas=tas, alt=alt_ft, vs=vs))
        co2_flow_gs = float(emission.co2(ff))

        fuel = ff * dt
        co2_g = co2_flow_gs * dt
        dist_nm += kts_to_nm_per_s(tas) * dt
        alt_ft += ftmin_to_fts(vs) * dt
        mass_kg -= fuel
        t += dt

        climb_seconds += dt
        climb_fuel += fuel
        climb_co2_g += co2_g

        rows.append(dict(time_s=t, phase="climb", alt_ft=alt_ft, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm))
        if dist_nm >= trip_distance_nm:
            break

    segs.append(SegmentResult("Climb", climb_seconds, dist_nm, climb_fuel, climb_co2_g / 1000.0))

    # CRUISE
    cruise_seconds = cruise_fuel = cruise_co2_g = 0.0
    while dist_nm < max(trip_distance_nm - descent_dist_nm_est, 0.0):
        vs = 0.0
        tas = cruise_tas_kts
        ff = float(fuelflow.enroute(mass=mass_kg, tas=tas, alt=clamp(alt_ft, 1000.0, cruise_alt_ft), vs=vs))
        co2_flow_gs = float(emission.co2(ff))

        fuel = ff * dt
        co2_g = co2_flow_gs * dt
        dist_nm += kts_to_nm_per_s(tas) * dt
        mass_kg -= fuel
        t += dt

        cruise_seconds += dt
        cruise_fuel += fuel
        cruise_co2_g += co2_g

        rows.append(dict(time_s=t, phase="cruise", alt_ft=alt_ft, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm))
        if dist_nm >= trip_distance_nm:
            break

    segs.append(SegmentResult("Cruise", cruise_seconds, max(dist_nm - segs[0].distance_nm, 0.0), cruise_fuel, cruise_co2_g / 1000.0))

    # DESCENT
    descent_seconds = descent_fuel = descent_co2_g = 0.0
    while dist_nm < trip_distance_nm and alt_ft > 0.0:
        vs = descent_vs_ftmin
        tas = descent_tas_kts
        ff = float(fuelflow.enroute(mass=mass_kg, tas=tas, alt=alt_ft, vs=vs))
        co2_flow_gs = float(emission.co2(ff))

        fuel = ff * dt
        co2_g = co2_flow_gs * dt
        dist_nm += kts_to_nm_per_s(tas) * dt
        alt_ft = max(0.0, alt_ft + ftmin_to_fts(vs) * dt)
        mass_kg -= fuel
        t += dt

        descent_seconds += dt
        descent_fuel += fuel
        descent_co2_g += co2_g

        rows.append(dict(time_s=t, phase="descent", alt_ft=alt_ft, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm))
        if alt_ft <= 0.0 and dist_nm >= trip_distance_nm:
            break

    # fail-safe “level-off”
    while dist_nm < trip_distance_nm:
        tas = cruise_tas_kts
        ff = float(fuelflow.enroute(mass=mass_kg, tas=tas, alt=1000.0, vs=0.0))
        co2_flow_gs = float(emission.co2(ff))
        fuel = ff * dt
        co2_g = co2_flow_gs * dt
        dist_nm += kts_to_nm_per_s(tas) * dt
        mass_kg -= fuel
        t += dt

        descent_seconds += dt
        descent_fuel += fuel
        descent_co2_g += co2_g
        rows.append(dict(time_s=t, phase="level-off", alt_ft=1000.0, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm))

    segs.append(SegmentResult("Descent", descent_seconds, max(trip_distance_nm - segs[0].distance_nm - segs[1].distance_nm, 0.0),
                              descent_fuel, descent_co2_g / 1000.0))

    df = pd.DataFrame(rows)
    return segs, df


def pareto_sweep(
    ac_code: str,
    mass_takeoff_kg: float,
    trip_distance_nm: float,
    cruise_alt_ft: float,
    cruise_tas_range_kts: Tuple[int, int] = (390, 470),
    steps: int = 10,
) -> pd.DataFrame:
    records = []
    low, high = cruise_tas_range_kts
    for tas in np.linspace(low, high, steps):
        segs, df = simulate_flight(
            ac_code=ac_code,
            mass_takeoff_kg=mass_takeoff_kg,
            trip_distance_nm=trip_distance_nm,
            cruise_alt_ft=cruise_alt_ft,
            cruise_tas_kts=float(tas),
        )
        fuel_total = sum(s.fuel_kg for s in segs)
        time_total_min = sum(s.seconds for s in segs) / 60.0
        co2_total = sum(s.co2_kg for s in segs)
        records.append(dict(cruise_tas_kts=float(tas), fuel_kg=fuel_total, block_time_min=time_total_min, co2_kg=co2_total))
    return pd.DataFrame(records)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="OpenAP Fuel–Time–Cost Explorer", page_icon="✈️", layout="wide")

# Styling
st.markdown("""
<style>
:root{
  --bg1:#0b1220; --bg2:#0e172a; --acc:#60a5fa; --acc2:#34d399;
  --card:#0b1220e6; --border:#1f2a44; --text:#e5e7eb; --muted:#9ca3af;
}
html, body, [class*="css"] { font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; }

/* luft upptill så hero inte klipps */
.block-container { padding-top: 2.4rem !important; }

/* inputs sida-vid-sida på desktop */
@media (min-width: 992px){
  [data-testid="stHorizontalBlock"]{ gap:22px !important; }
  [data-testid="column"]{ width:calc(50% - 11px) !important; flex:0 0 calc(50% - 11px) !important; }
}

/* Kort, knappar, hero */
.hero{
  margin-top:.3rem;
  background: radial-gradient(1000px 260px at 10% -20%, rgba(96,165,250,.25), transparent),
              linear-gradient(135deg, var(--bg1), var(--bg2));
  border:1px solid var(--border); border-radius:22px; padding:26px 22px; color:var(--text);
  box-shadow:0 10px 30px rgba(0,0,0,.25);
}
.hero h1{ margin:0 0 .3rem 0; font-weight:800; letter-spacing:.2px; }
.hero p{ margin:0; color:var(--muted); }

.card{ background:var(--card); border:1px solid var(--border); border-radius:18px; padding:18px; color:var(--text); box-shadow:0 6px 18px rgba(0,0,0,.16); }
.section-title{ font-weight:700; margin-bottom:.6rem; }
.badge{ display:inline-block; padding:.22rem .5rem; border:1px solid var(--border); border-radius:999px; color:var(--muted); font-size:.85rem; }

.stButton>button{
  background: linear-gradient(135deg, var(--acc), var(--acc2));
  color:#0b1220; border:0; border-radius:12px; padding:.7rem 1rem; font-weight:700; letter-spacing:.2px;
  box-shadow:0 8px 20px rgba(52,211,153,.2);
  transition: transform .08s ease, box-shadow .2s ease, opacity .2s ease;
}
.stButton>button:hover{ transform: translateY(-1px); box-shadow:0 10px 26px rgba(96,165,250,.28); }

[data-testid="stMetric"]{ background:rgba(15,23,42,.6); border:1px solid var(--border); border-radius:14px; padding:10px 12px; }
[data-testid="stMetricValue"]{ color:var(--text); }
            




</style>
""", unsafe_allow_html=True)

# Hero
st.markdown(
    """
    <div class="hero">
      <h1>✈️ OpenAP Fuel–Time–Cost Explorer</h1>
      <p>Simulerar bränsle, blocktid och CO₂ med en 3-fas-modell (Climb → Cruise → Descent) via OpenAP.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Inputs
st.markdown('<div class="section-title">Indata</div>', unsafe_allow_html=True)
col1, col2 = st.columns((1, 1), gap="large")

supported = [x.upper() for x in prop.available_aircraft()]
with col1:
    ac_code = st.selectbox("Flygplanstyp (ICAO)", options=supported, index=supported.index("A320") if "A320" in supported else 0)
    ac_data: Dict = prop.aircraft(ac_code)
    oew = float(ac_data.get("oew", 0))
    mtow = float(ac_data.get("mtow", 0))
    default_mass = (oew and mtow and (oew + mtow) / 2.0) or 65000.0

    mass_takeoff_kg = st.slider(
        "Antagen startmassa (kg)",
        min_value=int(max(20000, oew if oew else 20000)),
        max_value=int(max(50000, mtow if mtow else 120000)),
        value=int(default_mass), step=500
    )

    trip_distance_nm = st.number_input("Ruttlängd (NM)", value=500.0, min_value=50.0, step=50.0)
    cruise_alt_ft = st.number_input("Cruise-höjd (ft)", value=35000.0, min_value=10000.0, max_value=43000.0, step=1000.0)

with col2:
    st.markdown("**Hastigheter & vertikalhastigheter**")
    climb_tas_kts = st.number_input("Climb TAS (kts)", value=300.0, step=10.0)
    cruise_tas_kts = st.number_input("Cruise TAS (kts)", value=450.0, step=10.0)
    descent_tas_kts = st.number_input("Descent TAS (kts)", value=300.0, step=10.0)
    climb_vs_ftmin = st.number_input("Climb VS (ft/min)", value=1800.0, step=100.0)
    descent_vs_ftmin = st.number_input("Descent VS (ft/min)", value=-1500.0, step=100.0)

run = st.button("Kör simulering", use_container_width=True)

if run:
    segs, df = simulate_flight(
        ac_code,
        mass_takeoff_kg,
        trip_distance_nm,
        cruise_alt_ft,
        climb_tas_kts=climb_tas_kts,
        cruise_tas_kts=cruise_tas_kts,
        descent_tas_kts=descent_tas_kts,
        climb_vs_ftmin=climb_vs_ftmin,
        descent_vs_ftmin=descent_vs_ftmin,
        dt=1.0,
    )

    # Resultat-tabell
    st.markdown('<div class="section-title">Resultat</div>', unsafe_allow_html=True)
    res_df = pd.DataFrame([
        dict(
            Segment=s.name,
            Tid_min=round(s.seconds / 60.0, 1),
            Distans_NM=round(s.distance_nm, 1),
            Bränsle_kg=round(s.fuel_kg, 1),
            CO2_kg=round(s.co2_kg, 1),
        ) for s in segs
    ])
    total = dict(
        Segment="Totalt",
        Tid_min=round(res_df["Tid_min"].sum(), 1),
        Distans_NM=round(trip_distance_nm, 1),
        Bränsle_kg=round(res_df["Bränsle_kg"].sum(), 1),
        CO2_kg=round(res_df["CO2_kg"].sum(), 1),
    )
    res_df = pd.concat([res_df, pd.DataFrame([total])], ignore_index=True)

    # Metrics
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    with tcol1: st.metric("Bränsle totalt", f"{total['Bränsle_kg']:.1f} kg")
    with tcol2: st.metric("CO₂ totalt", f"{total['CO2_kg']:.1f} kg")
    with tcol3: st.metric("Blocktid", f"{total['Tid_min']:.1f} min")
    with tcol4: st.metric("Distans", f"{total['Distans_NM']:.1f} NM")

    st.markdown(
        f'<div class="badge">Typ: <b>{ac_code}</b> · MTOW: {int(mtow) if mtow else "—"} kg · OEW: {int(oew) if oew else "—"} kg</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Tabell", "📈 Profiler", "🟣 Fuel–Time trade-off"])

    # ---------- Tabell ----------
    with tab1:
        st.dataframe(res_df, use_container_width=True)

        # --- Segment-kort under tabellen (bara i Tabell-fliken) ---
        for _, r in res_df[res_df["Segment"] != "Totalt"].iterrows():
            st.markdown(
                f"""
                <div class="card" style="padding:12px; margin:12px 0;">
                <div style="font-weight:700; font-size:1.05rem;">{r['Segment']}</div>
                <div style="color:var(--muted); font-size:.95rem; line-height:1.4;">
                    Tid: {r['Tid_min']:.1f} min · Distans: {r['Distans_NM']:.1f} NM<br>
                    Bränsle: {r['Bränsle_kg']:.1f} kg · CO₂: {r['CO2_kg']:.1f} kg
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        

    

    # ---------- Profiler ----------
    with tab2:
        # Höjdprofil
        fig_alt = px.line(df, x="time_s", y="alt_ft", color="phase",
                          labels={"time_s":"Tid (s)", "alt_ft":"Höjd (ft)", "phase":"Fas"},
                          title="Höjdprofil")
        fig_alt.update_layout(
    margin=dict(l=10, r=10, t=40, b=60),  # extra bottenmarginal för legend
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.18,   # under grafen
        xanchor="center", x=0.5,
        title_text=""
    )
)
        st.plotly_chart(fig_alt, use_container_width=True, theme="streamlit", config={"displayModeBar": False})

        # Fartprofil
        fig_tas = px.line(df, x="time_s", y="tas_kts", color="phase",
                          labels={"time_s":"Tid (s)", "tas_kts":"TAS (kts)", "phase":"Fas"},
                          title="Fartprofil")
        fig_tas.update_layout(
    margin=dict(l=10, r=10, t=40, b=60),
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.18,
        xanchor="center", x=0.5,
        title_text=""
    )
)
        st.plotly_chart(fig_tas, use_container_width=True, theme="streamlit", config={"displayModeBar": False})

        # Bränsleflöde och kumulativt
        df_plot = df.copy()
        df_plot["fuel_cum_kg"] = df_plot["fuel_kg"].cumsum()

        fig_ff = px.line(df_plot, x="time_s", y="ff_kgs",
                         labels={"time_s":"Tid (s)", "ff_kgs":"Bränsleflöde (kg/s)"},
                         title="Bränsleflöde")
        fig_ff.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_ff, use_container_width=True, theme="streamlit", config={"displayModeBar": False})

        fig_cum = px.line(df_plot, x="time_s", y="fuel_cum_kg",
                          labels={"time_s":"Tid (s)", "fuel_cum_kg":"Kumulativt bränsle (kg)"},
                          title="Kumulativt bränsle")
        fig_cum.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cum, use_container_width=True, theme="streamlit", config={"displayModeBar": False})

    # ---------- Fuel–Time trade-off ----------
    with tab3:
       
        sweep_df = pareto_sweep(
            ac_code=ac_code,
            mass_takeoff_kg=mass_takeoff_kg,
            trip_distance_nm=trip_distance_nm,
            cruise_alt_ft=cruise_alt_ft,
            cruise_tas_range_kts=(int(max(360, cruise_tas_kts - 60)), int(min(500, cruise_tas_kts + 60))),
            steps=9,
        )

        if sweep_df.empty:
            st.info("Ingen data för svepet.")
        else:
                    # --- Fuel–Time utan färgskala/gradient ---
            fig_sw = px.scatter(
                sweep_df,
                x="block_time_min",
                y="fuel_kg",
                labels={
                    "block_time_min": "Blocktid (min)",
                    "fuel_kg": "Bränsle (kg)",
                },
                title="Fuel–Time trade-off",
                hover_data={"cruise_tas_kts": True},  # visa TAS i hover
            )

            # Stil & format
            fig_sw.update_traces(marker=dict(size=9))
            fig_sw.update_layout(
                margin=dict(l=10, r=10, t=40, b=30),
                showlegend=False  # ingen legend behövs
            )
            fig_sw.update_xaxes(tickformat=".0f")
            fig_sw.update_yaxes(tickformat=".0f")

            # Stjärnmarkörer (behåll)
            r_min_fuel = sweep_df.loc[sweep_df["fuel_kg"].idxmin()]
            r_min_time = sweep_df.loc[sweep_df["block_time_min"].idxmin()]
            fig_sw.add_scatter(
                x=[r_min_fuel["block_time_min"]], y=[r_min_fuel["fuel_kg"]],
                mode="markers+text", text=["Min fuel"], textposition="top center",
                marker=dict(size=14, symbol="star", line=dict(width=1, color="white")),
                showlegend=False, hoverinfo="skip",
            )
            fig_sw.add_scatter(
                x=[r_min_time["block_time_min"]], y=[r_min_time["fuel_kg"]],
                mode="markers+text", text=["Snabbast"], textposition="bottom center",
                marker=dict(size=14, symbol="star", line=dict(width=1, color="white")),
                showlegend=False, hoverinfo="skip",
            )

            st.plotly_chart(fig_sw, use_container_width=True, theme="streamlit", config={"displayModeBar": False})
            st.caption("Punkter = olika cruise-hastigheter. Stjärnor = snålast respektive snabbast.")

      

    st.markdown("---")
    st.caption("Drivs av OpenAP (pip-paketet `openap`).")
else:
    st.caption("Ställ in indata ovan och klicka **Kör simulering**.")
