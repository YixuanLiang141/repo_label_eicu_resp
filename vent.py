import numpy as np
import pandas as pd

def identify_invasive_ventilation(
    respiratory_charting_csv,
    respiratory_care_csv,
    output_csv,
    max_gap_hours=4,
):
    columns = [
        "stay_id", "starttime", "endtime",
        "evidence_level", "end_inferred",
    ]
    flags = [
        "airway_on", "airway_off",
        "vent_strong", "vent_weak",
        "noninvasive", "stop",
    ]

    if not np.isfinite(max_gap_hours) or max_gap_hours <= 0:
        raise ValueError("max_gap_hours must be a positive finite number.")

    max_gap = float(max_gap_hours) * 60
    base_time = pd.Timestamp("2050-01-01")
    event_parts = []

    device_labels = {
        "O2 Device",
        "RETIRED O2 Device",
        "Oxygen Delivery Method",
    }
    mode_labels = {
        "Mechanical Ventilator Mode",
        "Ventilator Support Mode",
    }
    airway_labels = {
        "Endotracheal Tube Placement",
        "Endotracheal Tube Placement Checked",
        "Endotracheal Position at Lip",
    }
    relevant_labels = (
        device_labels
        | mode_labels
        | airway_labels
        | {
            "RT Vent On/Off",
            "Vent Rate",
            "Tidal Volume (set)",
            "Bipap Delivery Mode",
            "Non-invasive Ventilation Mode",
        }
    )

    invasive_modes = {
        "ac/cmv", "cmv", "simv", "simv+",
        "pcv w/assist", "aprv", "apv", "pressure control",
    }
    ambiguous_modes = {"cpap", "ps", "spontaneous"}

    noninvasive_devices = {
        "none (room air)", "room air",
        "nasal cannula",
        "high flow nasal cannula",
        "heated high flow nasal cannula",
        "bi-pap", "bipap", "cpap",
        "non-rebreather", "non-rebreather mask",
        "face mask", "aerosol mask", "simple mask",
        "high flow mask", "venturi mask", "venti mask",
        "face tent", "blow-by", "blowby", "oxyhood",
        "trach mask/collar", "trach collar", "trach mask",
        "t-piece",
    }
    mask_interfaces = {
        "facial bipap mask", "nasal bipap mask", "nasal pillows",
    }
    noninvasive_modes = {
        "s/t", "avaps", "cpap", "nasal cannula", "hfnc",
        "venturi", "non-rebreather mask",
    }
    airway_placements = {
        "oral endotracheal", "teeth", "gum", "lip",
        "tracheostomy", "stoma", "left naris", "right naris",
    }
    invasive_airways = {
        "oral ett", "nasal ett", "tracheostomy",
        "double-lumen tube", "cricothyrotomy",
    }

    # Extract charting evidence in chunks to limit memory usage.
    chart_columns = [
        "patientunitstayid", "respchartoffset",
        "respchartvaluelabel", "respchartvalue",
    ]
    for chunk in pd.read_csv(
        respiratory_charting_csv,
        usecols=chart_columns,
        dtype="string",
        chunksize=500_000,
    ):
        labels = chunk["respchartvaluelabel"].str.strip()
        chunk = chunk.loc[labels.isin(relevant_labels)].copy()

        if chunk.empty:
            continue

        label = chunk["respchartvaluelabel"].str.strip()
        value = chunk["respchartvalue"].str.strip().str.casefold()
        numeric = pd.to_numeric(value, errors="coerce")

        events = chunk[
            ["patientunitstayid", "respchartoffset"]
        ].rename(columns={
            "patientunitstayid": "stay_id",
            "respchartoffset": "offset",
        })

        # Artificial airway evidence does not independently imply ventilation.
        events["airway_on"] = (
            (
                label.eq("Endotracheal Tube Placement")
                & value.isin(airway_placements)
            )
            | (
                label.eq("Endotracheal Tube Placement Checked")
                & value.eq("y")
            )
            | (
                label.eq("Endotracheal Position at Lip")
                & numeric.gt(0)
                & np.isfinite(numeric)
            )
            | (
                label.isin(device_labels)
                & value.eq("ett")
            )
        )
        events["airway_off"] = False

        # Explicit ventilator devices and typical modes support probable IMV.
        events["vent_strong"] = (
            (
                label.isin(device_labels)
                & value.isin({"ventilator", "mechanical ventilator"})
            )
            | (
                label.isin(mode_labels)
                & value.isin(invasive_modes)
            )
        )

        # These signals require recent artificial airway evidence.
        events["vent_weak"] = (
            (
                label.eq("RT Vent On/Off")
                & value.isin({"start", "continued"})
            )
            | (
                label.isin(mode_labels)
                & value.isin(ambiguous_modes)
            )
            | (
                label.isin({"Vent Rate", "Tidal Volume (set)"})
                & numeric.gt(0)
                & np.isfinite(numeric)
            )
        )

        # Trach collar and T-piece indicate breathing without the ventilator.
        events["noninvasive"] = (
            (
                label.isin(device_labels)
                & value.isin(noninvasive_devices)
            )
            | (
                label.isin(mode_labels)
                & value.isin({"trach mask", "t-piece"})
            )
            | (
                label.eq("Bipap Delivery Mode")
                & value.isin(mask_interfaces)
            )
            | (
                label.eq("Non-invasive Ventilation Mode")
                & value.isin(noninvasive_modes)
            )
        )
        events["stop"] = (
            label.eq("RT Vent On/Off")
            & value.isin({"off", "suspended"})
        )

        events[flags] = events[flags].fillna(False).astype(bool)
        events = events.loc[events[flags].any(axis=1)].copy()

        if not events.empty:
            events["offset"] = pd.to_numeric(
                events["offset"], errors="raise"
            )
            if (
                events["stay_id"].isna().any()
                or not np.isfinite(events["offset"]).all()
            ):
                raise ValueError("Relevant charting records have invalid IDs or offsets.")

            event_parts.append(
                events.groupby(
                    ["stay_id", "offset"], as_index=False
                )[flags].max()
            )

    # Use respiratoryCare for airway evidence, not raw vent start/end offsets.
    care_columns = [
        "patientunitstayid", "respcarestatusoffset", "airwaytype",
    ]
    for chunk in pd.read_csv(
        respiratory_care_csv,
        usecols=care_columns,
        dtype="string",
        chunksize=200_000,
    ):
        airway = chunk["airwaytype"].str.strip().str.casefold()
        keep = airway.isin(invasive_airways | {"no artificial airway"})
        chunk = chunk.loc[keep].copy()
        airway = airway.loc[keep]

        if chunk.empty:
            continue

        events = chunk[
            ["patientunitstayid", "respcarestatusoffset"]
        ].rename(columns={
            "patientunitstayid": "stay_id",
            "respcarestatusoffset": "offset",
        })
        events["offset"] = pd.to_numeric(
            events["offset"], errors="raise"
        )

        if (
            events["stay_id"].isna().any()
            or not np.isfinite(events["offset"]).all()
        ):
            raise ValueError("Relevant care records have invalid IDs or offsets.")

        for flag in flags:
            events[flag] = False

        events["airway_on"] = airway.isin(invasive_airways)
        events["airway_off"] = airway.eq("no artificial airway")

        event_parts.append(
            events.groupby(
                ["stay_id", "offset"], as_index=False
            )[flags].max()
        )

    if not event_parts:
        return pd.DataFrame(columns=columns)

    # Combine simultaneous evidence from both tables.
    events = (
        pd.concat(event_parts, ignore_index=True)
        .groupby(["stay_id", "offset"], as_index=False)[flags]
        .max()
        .sort_values(["stay_id", "offset"])
    )

    intervals = []

    for stay_id, group in events.groupby("stay_id", sort=False):
        airway_time = None
        vent_time = None
        vent_kind = None

        start = None
        last_positive = None
        episode_level = None

        for row in group.itertuples(index=False):
            time = float(row.offset)

            # Close an episode when its last positive evidence expires.
            if (
                start is not None
                and time - last_positive > max_gap
            ):
                intervals.append((
                    stay_id, start, last_positive + max_gap,
                    episode_level, True,
                ))
                start = None
                last_positive = None
                episode_level = None

            # Historical evidence expires and never refreshes itself.
            if (
                airway_time is not None
                and time - airway_time > max_gap
            ):
                airway_time = None

            if (
                vent_time is not None
                and time - vent_time > max_gap
            ):
                vent_time = None
                vent_kind = None

            negative = row.airway_off or row.noninvasive or row.stop

            # Conflicting explicit evidence is unknown, not automatically IMV.
            explicit_positive = (
                row.vent_strong
                or (row.airway_on and row.vent_weak)
            )
            conflict = (
                (row.airway_on and row.airway_off)
                or (explicit_positive and negative)
            )

            if conflict:
                if start is not None:
                    intervals.append((
                        stay_id, start, time,
                        episode_level, True,
                    ))
                start = None
                last_positive = None
                episode_level = None
                airway_time = None
                vent_time = None
                vent_kind = None
                continue

            if negative:
                if start is not None:
                    intervals.append((
                        stay_id, start, time,
                        episode_level, False,
                    ))

                start = None
                last_positive = None
                episode_level = None
                vent_time = None
                vent_kind = None

                # A new interface invalidates the previous airway context.
                if row.airway_off or row.noninvasive:
                    airway_time = None
                elif row.airway_on:
                    airway_time = time
                continue

            if row.airway_on:
                airway_time = time

            if row.vent_strong:
                vent_time = time
                vent_kind = "strong"
            elif row.vent_weak:
                vent_time = time
                vent_kind = "weak"

            # Strong IMV requires both recent airway and ventilation evidence.
            if airway_time is not None and vent_time is not None:
                level = "strong"
            elif vent_time is not None and vent_kind == "strong":
                level = "probable"
            else:
                level = None

            if level is not None:
                if start is None:
                    start = time
                    episode_level = level
                elif level == "strong":
                    # Report the strongest evidence found in this episode.
                    episode_level = "strong"

                last_positive = time

        # Without an explicit stop, infer the end from the evidence timeout.
        if start is not None:
            intervals.append((
                stay_id, start, last_positive + max_gap,
                episode_level, True,
            ))

    if not intervals:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(
        intervals,
        columns=[
            "stay_id", "start_offset", "end_offset",
            "evidence_level", "end_inferred",
        ],
    )

    # Zero-duration episodes do not define a ventilation interval.
    result = result.loc[
        result["end_offset"] > result["start_offset"]
    ].copy()

    result["starttime"] = (
        base_time + pd.to_timedelta(result["start_offset"], unit="min")
    )
    result["endtime"] = (
        base_time + pd.to_timedelta(result["end_offset"], unit="min")
    )

    output = result[columns].sort_values(["stay_id", "starttime"]).reset_index(drop=True)
    output.to_csv(output_csv, index=False)

    return output

if __name__ == "__main__":
    identify_invasive_ventilation("../../eicu/respiratoryCharting.csv", "../../eicu/respiratoryCare.csv", "vent.csv")