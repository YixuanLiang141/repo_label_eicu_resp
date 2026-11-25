import pandas as pd
import numpy as np

def filter_spo2(spo2_file1, spo2_file2, pao2_file, respiratory_file, output_file):
    # Keep the existing four positional arguments compatible.
    # The optional fourth input has stay_id, charttime, item, value columns.
    # Files are read in full, as in the original implementation.
    keys = ["stay_id", "charttime"]
    columns = keys + ["item", "value"]
    selected_items = ["SpO2", "FiO2", "LPM O2"]

    parts = [
        pd.read_csv(
            path,
            usecols=keys + ["value"],
            dtype={"stay_id": "string"},
        ).assign(item="SpO2")
        for path in (spo2_file1, spo2_file2)
    ]
    parts.append(pd.read_csv(
        pao2_file,
        usecols=columns,
        dtype={"stay_id": "string"},
    ))
    parts.append(pd.read_csv(
        respiratory_file,
        usecols=columns,
        dtype={"stay_id": "string"},
    ))

    # Parse timestamps consistently across all supplied inputs.
    for frame in parts:
        frame["charttime"] = pd.to_datetime(
            frame["charttime"], format="mixed", errors="raise"
        ).astype("datetime64[ns]")
        if frame[keys].isna().any().any():
            raise ValueError("stay_id and charttime must not be missing.")

    observations = pd.concat(parts, ignore_index=True)
    targets = observations.loc[
        observations["item"].eq("paO2"), keys
    ].drop_duplicates()

    # Preserve all non-selected items, including every original paO2 row.
    retained = [observations.loc[
        ~observations["item"].isin(selected_items), columns
    ]]

    # Apply the original time-selection logic independently to each item.
    # Identical duplicates are removed; different values at a selected time stay.
    for item in selected_items:
        source = observations.loc[
            observations["item"].eq(item), columns
        ].drop_duplicates(subset=columns)
        if targets.empty or source.empty:
            continue

        source_times = source[keys].drop_duplicates()
        target_matches = targets.merge(
            source_times, on=keys, how="left", indicator=True
        )
        exact_times = target_matches.loc[
            target_matches["_merge"].eq("both"), keys
        ]
        unmatched = target_matches.loc[
            target_matches["_merge"].eq("left_only"), keys
        ]
        selected_times = [exact_times]

        if not unmatched.empty:
            # merge_asof requires globally sorted time columns.
            left = unmatched.sort_values("charttime", kind="stable")
            right = source_times.rename(
                columns={"charttime": "_source_time"}
            ).sort_values("_source_time", kind="stable")

            # With no exact match, keep the nearest time on each side.
            # Never cross stay boundaries; retain the original unlimited range.
            for direction in ("backward", "forward"):
                matched = pd.merge_asof(
                    left, right,
                    left_on="charttime", right_on="_source_time",
                    by="stay_id", direction=direction,
                    allow_exact_matches=False,
                )
                selected_times.append(
                    matched[["stay_id", "_source_time"]]
                    .dropna(subset=["_source_time"])
                    .rename(columns={"_source_time": "charttime"})
                )

        selected_times = pd.concat(
            selected_times, ignore_index=True
        ).drop_duplicates()
        retained.append(source.merge(
            selected_times, on=keys, how="inner", validate="many_to_one"
        ))

    output = (
        pd.concat(retained, ignore_index=True)[columns]
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    output.to_csv(output_file, index=False)
    return output

def classify_arterial_pao2(input_csv, output_csv, fio2_unit="percent"):

    keys = ["stay_id", "charttime"]
    items = [
        "paO2", "FiO2", "Hgb", "sO2", "paCO2",
        "bicarbonate", "Total CO2", "lactate",
        "pH", "LPM O2", "SpO2",
    ]

    if fio2_unit not in ("percent", "fraction"):
        raise ValueError("fio2_unit must be 'percent' or 'fraction'.")

    df = pd.read_csv(
        input_csv,
        usecols=keys + ["item", "value"],
        dtype={"stay_id": "string"},
    )
    df = df[df["item"].isin(items)].copy()

    df["charttime"] = pd.to_datetime(
        df["charttime"], format="mixed", errors="raise"
    ).astype("datetime64[ns]")

    if df[keys].isna().any().any():
        raise ValueError("stay_id and charttime must not be missing.")

    # Treat invalid numeric values as unavailable measurements.
    df["value"] = pd.to_numeric(
        df["value"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    # Average duplicate observations for the same stay, time, and item.
    scores = df.groupby(
        keys + ["item"], as_index=False
    )["value"].mean()

    # Build exact-time features for every recorded paO2 timestamp.
    wide = (
        scores.pivot(index=keys, columns="item", values="value")
        .reindex(columns=items)
        .reset_index()
    )
    pao2_times = scores.loc[
        scores["item"].eq("paO2"), keys
    ]
    features = wide.merge(
        pao2_times, on=keys, how="inner", validate="one_to_one"
    )

    # SpO2, FiO2, and LPM O2 may use the latest valid measurement
    # within the preceding four hours, including exact matches.
    for item in ("SpO2", "FiO2", "LPM O2"):
        source = (
            scores.loc[
                scores["item"].eq(item),
                keys + ["value"],
            ]
            .dropna(subset=["value"])
            .rename(columns={"value": item})
        )
        features = features.drop(columns=[item])

        if features.empty or source.empty:
            features[item] = np.nan
            continue

        features = pd.merge_asof(
            features.sort_values("charttime", kind="stable"),
            source.sort_values("charttime", kind="stable"),
            on="charttime",
            by="stay_id",
            direction="backward",
            tolerance=pd.Timedelta(hours=4),
            allow_exact_matches=True,
        )

    # The classifier uses FiO2 in percent.
    # Preserve the original FiO2 units in the returned data.
    fio2_percent = features["FiO2"]
    if fio2_unit == "fraction":
        fio2_percent = fio2_percent * 100

    # Estimate A-a oxygen gradient using available measurements only.
    # Missing inputs leave aado2 missing so its model fallback is used.
    features["aado2"] = (
        (760.0 - 47.0) * (fio2_percent / 100.0)
        - features["paCO2"] / 0.8
        - features["paO2"]
    )
    features["_fio2_percent"] = fio2_percent

    # Each fallback includes the original SQL missingness adjustment.
    # Retain only the FiO2 term with coefficient 0.01480.
    terms = [
        ("SpO2",          -0.15356, 97.49420,  0.13429),
        ("Hgb",            0.10559, 10.32307,  0.05954),
        ("sO2",            0.13251, 93.66539, -0.23172),
        ("paCO2",         -0.01511, 42.08866, -0.01630),
        ("_fio2_percent",  0.01480, 63.97836, -0.31142),
        ("aado2",         -0.00200, 442.21186, -0.01328),
        ("bicarbonate",   -0.03220, 22.96894, -0.06535),
        ("Total CO2",      0.05384, 24.72632, -0.01405),
        ("lactate",        0.08202, 3.06436,   0.06038),
        ("pH",             0.10956, 7.36233,  -0.00617),
        ("LPM O2",         0.00848, 7.59362,  -0.35803),
    ]

    logit = -0.02544 + 0.04598 * features["paO2"]

    for column, coefficient, default_value, adjustment in terms:
        contribution = coefficient * features[column]
        logit += contribution.fillna(
            coefficient * default_value + adjustment
        )

    # Evaluate the sigmoid without exponential overflow.
    probability = np.exp(-np.logaddexp(0.0, -logit))
    features["art"] = (probability > 0.75).astype("int64")

    # Retain only original paO2 and FiO2 timestamps.
    # Historical FiO2 matches are not added as new observations.
    result = scores.loc[
        scores["item"].isin(["paO2", "FiO2"])
    ].merge(
        features[keys + ["art"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )

    # The arterial classification applies only to paO2 rows.
    result["art"] = result["art"].astype("Int64")
    result.loc[result["item"].eq("FiO2"), "art"] = pd.NA

    output = result[keys + ["item", "value", "art"]].sort_values(keys + ["item"], kind="stable").reset_index(drop=True)

    output.to_csv(output_csv, index=False)

    return output

def calculate_pao2fio2ratio(input_csv, output_csv, ventilation_csv=None, fio2_unit="percent"):

    data = pd.read_csv(input_csv)

    keys = ["stay_id", "charttime"]
    columns = keys + ["pao2fio2ratio"]
    output_columns = columns + ["vent"]

    if fio2_unit not in ("percent", "fraction"):
        raise ValueError("fio2_unit must be 'percent' or 'fraction'.")

    df = data[keys + ["item", "value", "art"]].copy()
    df["charttime"] = pd.to_datetime(
        df["charttime"], format="mixed", errors="raise"
    ).astype("datetime64[ns]")

    if df[keys].isna().any().any():
        raise ValueError("stay_id and charttime must not be missing.")

    df["value"] = pd.to_numeric(
        df["value"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    df["art"] = pd.to_numeric(df["art"], errors="coerce")

    # Select arterial paO2 observations only.
    pao2 = (
        df.loc[
            df["item"].eq("paO2") & df["art"].eq(1),
            keys + ["value"],
        ]
        .dropna(subset=["value"])
        .groupby(keys, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "pao2"})
    )

    # Average duplicate FiO2 measurements at the same timestamp.
    fio2 = (
        df.loc[df["item"].eq("FiO2"), keys + ["value"]]
        .dropna(subset=["value"])
        .groupby(keys, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "fio2"})
    )

    # Convert FiO2 to a fraction for the ratio calculation.
    if fio2_unit == "percent":
        fio2["fio2"] = fio2["fio2"] / 100.0

    # Exclude invalid oxygen fractions before temporal matching.
    fio2 = fio2.loc[fio2["fio2"].gt(0) & fio2["fio2"].le(1)]

    if pao2.empty or fio2.empty:
        return pd.DataFrame(columns=output_columns)

    # Prefer exact matches; otherwise use the latest valid FiO2
    # within the preceding four hours, including the four-hour boundary.
    matched = pd.merge_asof(
        pao2.sort_values("charttime", kind="stable"),
        fio2.sort_values("charttime", kind="stable"),
        on="charttime",
        by="stay_id",
        direction="backward",
        tolerance=pd.Timedelta(hours=4),
        allow_exact_matches=True,
    )

    matched["pao2fio2ratio"] = matched["pao2"] / matched["fio2"]

    # Retain only observations with a finite calculated ratio.
    result = matched.loc[
        np.isfinite(matched["pao2fio2ratio"]), columns
    ]

    output = result.sort_values(keys, kind="stable").reset_index(drop=True)

    # Preserve the existing call signature; an omitted interval file gives vent=0.
    output["vent"] = 0
    if ventilation_csv is not None:
        intervals = pd.read_csv(
            ventilation_csv,
            usecols=["stay_id", "starttime", "endtime"],
            dtype={"stay_id": "string"},
        )
        for column in ("starttime", "endtime"):
            intervals[column] = pd.to_datetime(
                intervals[column], format="mixed", errors="raise"
            ).astype("datetime64[ns]")

        if intervals[["stay_id", "starttime", "endtime"]].isna().any().any():
            raise ValueError("Ventilation stay_id, starttime and endtime must not be missing.")
        if intervals["endtime"].lt(intervals["starttime"]).any():
            raise ValueError("Ventilation endtime must not precede starttime.")

        # Empty intervals contain no observations under [starttime, endtime).
        intervals = intervals.loc[
            intervals["endtime"].gt(intervals["starttime"])
        ].copy()

        if not output.empty and not intervals.empty:
            # Use a temporary string key without changing the output stay_id type.
            output["_vent_stay"] = output["stay_id"].astype("string")
            intervals["_vent_stay"] = intervals["stay_id"].astype("string")
            intervals = intervals.sort_values(
                ["_vent_stay", "starttime", "endtime"], kind="stable"
            )

            # Running maximum ends handle overlapping and nested intervals.
            intervals["_vent_end"] = intervals.groupby(
                "_vent_stay", sort=False
            )["endtime"].cummax()

            output = pd.merge_asof(
                output.sort_values("charttime", kind="stable"),
                intervals[["_vent_stay", "starttime", "_vent_end"]]
                .sort_values("starttime", kind="stable"),
                left_on="charttime",
                right_on="starttime",
                by="_vent_stay",
                direction="backward",
                allow_exact_matches=True,
            )

            # The matched start is inclusive; the interval end is exclusive.
            output["vent"] = output["charttime"].lt(
                output["_vent_end"]
            ).astype("int64")

    output = (
        output[output_columns]
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )

    output.to_csv(output_csv, index=False)

    return output

if __name__ == "__main__":
    #filter_spo2('vitalresp.csv', 'nurseresp.csv', 'labresp.csv', 'chartresp.csv', 'resp.csv')
    #classify_arterial_pao2('resp.csv', 'pafi_raw.csv')
    calculate_pao2fio2ratio('pafi_raw.csv', 'pafi.csv','vent.csv')
