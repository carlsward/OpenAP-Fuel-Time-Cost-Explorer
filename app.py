import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# OpenAP – officiell Python-implementation
# Docs/exempel: https://openap.dev/  (FuelFlow/Emission/prop)
from openap import FuelFlow, Emission, prop  # pip install openap


# ---------- Hjälpfunktioner ----------
def kts_to_nm_per_s(kts: float) -> float:
    # 1 knot = 1 NM per timme
    return kts / 3600.0


def ftmin_to_fts(vs_ftmin: float) -> float:
    return vs_ftmin / 60.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class SegmentResult:
    name: str
    seconds: float
    distance_nm: float
    fuel_kg: float
    co2_kg: float


# ---------- Simuleringskärna ----------
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
    """
    Enkel 3-fas: climb -> cruise -> descent.
    Integrerar bränsleflöde (kg/s) och emission (g/s) från OpenAP över tiden.
    """

    ac = ac_code.upper()
    fuelflow = FuelFlow(ac=ac)
    emission = Emission(ac=ac)

    # Förbered tidsserie
    t = 0.0
    alt_ft = 0.0
    mass_kg = float(mass_takeoff_kg)
    dist_nm = 0.0

    rows = []  # för tidsserie/plot
    segs: List[SegmentResult] = []

    # --- Förberäkna ungefärlig nedstigningsdistans (för att veta hur mycket cruise som behövs)
    # d_desc ≈ tid * TAS
    # tid (s) = höjdskillnad(ft)/|vs| (ft/min) * 60
    descent_time_s_est = (cruise_alt_ft - 0.0) / abs(descent_vs_ftmin) * 60.0
    descent_dist_nm_est = kts_to_nm_per_s(descent_tas_kts) * descent_time_s_est

    # --- CLIMB ---
    climb_seconds = 0.0
    climb_fuel = 0.0
    climb_co2_g = 0.0
    while alt_ft < cruise_alt_ft and dist_nm < max(1e-6, trip_distance_nm - descent_dist_nm_est):
        vs = climb_vs_ftmin  # ft/min
        tas = climb_tas_kts  # kts
        # Fuel flow i kg/s enligt OpenAP
        ff = float(fuelflow.enroute(mass=mass_kg, tas=tas, alt=alt_ft, vs=vs))
        # Emissioner (g/s). CO2 är linjärt av FF i OpenAP.
        co2_flow_gs = float(emission.co2(ff))

        # Integrera
        fuel = ff * dt  # kg
        co2_g = co2_flow_gs * dt  # g
        dist_nm += kts_to_nm_per_s(tas) * dt
        alt_ft += ftmin_to_fts(vs) * dt
        mass_kg -= fuel
        t += dt

        climb_seconds += dt
        climb_fuel += fuel
        climb_co2_g += co2_g

        rows.append(
            dict(time_s=t, phase="climb", alt_ft=alt_ft, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm)
        )

        if dist_nm >= trip_distance_nm:  # extrem-två kort rutt
            break

    segs.append(
        SegmentResult(
            name="Climb",
            seconds=climb_seconds,
            distance_nm=dist_nm,
            fuel_kg=climb_fuel,
            co2_kg=climb_co2_g / 1000.0,
        )
    )

    # --- CRUISE ---
    cruise_seconds = 0.0
    cruise_fuel = 0.0
    cruise_co2_g = 0.0
    while dist_nm < max(trip_distance_nm - descent_dist_nm_est, 0.0):
        vs = 0.0
        tas = cruise_tas_kts
        ff = float(fuelflow.enroute(mass=mass_kg, tas=tas, alt=clamp(alt_ft, 1000.0, cruise_alt_ft), vs=vs))
        co2_flow_gs = float(emission.co2(ff))

        fuel = ff * dt
        co2_g = co2_flow_gs * dt
        dist_nm += kts_to_nm_per_s(tas) * dt
        # alt konstant i cruise
        mass_kg -= fuel
        t += dt

        cruise_seconds += dt
        cruise_fuel += fuel
        cruise_co2_g += co2_g

        rows.append(
            dict(time_s=t, phase="cruise", alt_ft=alt_ft, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm)
        )

        if dist_nm >= trip_distance_nm:
            break

    segs.append(
        SegmentResult(
            name="Cruise",
            seconds=cruise_seconds,
            distance_nm=max(dist_nm - segs[0].distance_nm, 0.0),
            fuel_kg=cruise_fuel,
            co2_kg=cruise_co2_g / 1000.0,
        )
    )

    # --- DESCENT ---
    descent_seconds = 0.0
    descent_fuel = 0.0
    descent_co2_g = 0.0
    while dist_nm < trip_distance_nm and alt_ft > 0.0:
        vs = descent_vs_ftmin  # negativ
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

        rows.append(
            dict(time_s=t, phase="descent", alt_ft=alt_ft, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm)
        )

        if alt_ft <= 0.0 and dist_nm >= trip_distance_nm:
            break

    # Om vi slog i marken före distansmålet, "taxisera" sista biten i cruise-hastighet (enkel fail-safe)
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
        rows.append(
            dict(time_s=t, phase="level-off", alt_ft=1000.0, tas_kts=tas, ff_kgs=ff, fuel_kg=fuel, co2_g=co2_g, dist_nm=dist_nm)
        )

    segs.append(
        SegmentResult(
            name="Descent",
            seconds=descent_seconds,
            distance_nm=max(trip_distance_nm - segs[0].distance_nm - segs[1].distance_nm, 0.0),
            fuel_kg=descent_fuel,
            co2_kg=descent_co2_g / 1000.0,
        )
    )

    # tidsserie
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
    """Beräkna trade-off mellan total bränsle och blocktid för olika cruise-TAS."""
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
st.set_page_config(page_title="OpenAP Fuel–Time–Cost Explorer", layout="wide")

st.title("OpenAP Fuel–Time–Cost Explorer")
st.caption(
    "Simulerar bränsle, blocktid och CO₂ med OpenAP (FuelFlow + Emission). "
    "Tre-fas-modell: Climb → Cruise → Descent. "
)

# Lista stödda typer från OpenAP (handboken visar exakt dessa)
supported = [x.upper() for x in prop.available_aircraft()]
col1, col2 = st.columns(2)
with col1:
    ac_code = st.selectbox("Flygplanstyp (ICAO)", options=supported, index=supported.index("A320") if "A320" in supported else 0)

    # Hämta typdata för default-massor
    ac_data: Dict = prop.aircraft(ac_code)
    oew = float(ac_data.get("oew", 0))
    mtow = float(ac_data.get("mtow", 0))
    default_mass = (oew + mtow) / 2.0 if (oew and mtow) else 65000.0

    mass_takeoff_kg = st.slider("Antagen startmassa (kg)", min_value=int(max(20000, oew if oew else 20000)),
                                max_value=int(max(50000, mtow if mtow else 120000)),
                                value=int(default_mass), step=500)

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

    # Resultattabell
    st.subheader("Resultat per segment")
    res_df = pd.DataFrame(
        [
            dict(
                Segment=s.name,
                Tid_min=round(s.seconds / 60.0, 1),
                Distans_NM=round(s.distance_nm, 1),
                Bränsle_kg=int(s.fuel_kg),
                CO2_kg=int(s.co2_kg),
            )
            for s in segs
        ]
    )
    total = dict(
        Segment="Totalt",
        Tid_min=round(res_df["Tid_min"].sum(), 1),
        Distans_NM=round(trip_distance_nm, 1),
        Bränsle_kg=int(res_df["Bränsle_kg"].sum()),
        CO2_kg=int(res_df["CO2_kg"].sum()),
    )
    res_df = pd.concat([res_df, pd.DataFrame([total])], ignore_index=True)
    st.dataframe(res_df, use_container_width=True)

    # Ladda ner råtidsserie
    st.download_button(
        label="Ladda ner tidsserie (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="timeline.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Höjd & hastighet över tid")
    colA, colB = st.columns(2)
    with colA:
        st.line_chart(df.set_index("time_s")[["alt_ft"]])
    with colB:
        st.line_chart(df.set_index("time_s")[["tas_kts"]])

    st.subheader("Bränsleflöde (kg/s) och kumulativ bränsle (kg)")
    df["fuel_cum_kg"] = df["fuel_kg"].cumsum()
    colC, colD = st.columns(2)
    with colC:
        st.line_chart(df.set_index("time_s")[["ff_kgs"]])
    with colD:
        st.line_chart(df.set_index("time_s")[["fuel_cum_kg"]])

    # Pareto-svep (valfritt)
    st.subheader("Fuel–Time trade-off (svep över Cruise TAS)")
    sweep_df = pareto_sweep(
        ac_code=ac_code,
        mass_takeoff_kg=mass_takeoff_kg,
        trip_distance_nm=trip_distance_nm,
        cruise_alt_ft=cruise_alt_ft,
        cruise_tas_range_kts=(int(max(360, cruise_tas_kts - 60)), int(min(500, cruise_tas_kts + 60))),
        steps=9,
    )
    st.scatter_chart(sweep_df, x="block_time_min", y="fuel_kg", size=None, color=None)
    st.caption("Punkter = olika cruise-hastigheter (kts). Välj den balans du vill visa/provjämföra.")

st.markdown("---")
st.caption(
    "Drivs av OpenAP (pip-paketet `openap`). Grundidé: estimera bränsleflöde från mass/TAS/alt/VS, "
    "integrera över tid och summera segment. Emissioner (CO₂) från OpenAP:s Emission-klass."
)
