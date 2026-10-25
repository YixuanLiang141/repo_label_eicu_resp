import pandas as pd

def filter_large_csv_by_stay_ids(input_file, output_file, key_col, chunksize=100_000):
    valid_ids = ['paO2', 'FiO2', 'Hgb', 'O2 Sat (%)', 'paCO2', 'bicarbonate', 'Total CO2', 'lactate', 'pH', 'LPM O2']

    reader = pd.read_csv(input_file, chunksize=chunksize)
    first_chunk = True

    total_rows_kept = 0
    for i, chunk in enumerate(reader):
        filtered_chunk = chunk[chunk[key_col].isin(valid_ids)]
        total_rows_kept += len(filtered_chunk)

        if first_chunk:
            filtered_chunk.to_csv(output_file, index=False, mode='w')
            first_chunk = False
        else:
            filtered_chunk.to_csv(output_file, index=False, mode='a', header=False)

        print(f"✔ 已处理第 {i+1} 块，保留 {len(filtered_chunk)} 行")

def process_columns_rename_time(data_file, output_file):

    df = pd.read_csv(data_file)

    # Remove spaces
    df["labresulttext"] = df["labresulttext"].astype("string")
    df["labresulttext"] = df["labresulttext"].str.rstrip()

    # Remove O2 flow with %
    mask = (
        (df["labname"] == 'LPM O2')
        & (df["labresulttext"].astype(str).str.endswith("%"))
    )
    df = df[~mask]

    # Remove symbols
    mask = df["labresulttext"].astype(str).str.endswith("%")
    df.loc[mask, "labresulttext"] = df.loc[mask, "labresulttext"].str.rstrip("%")
    mask = df["labresulttext"].astype(str).str.startswith(">")
    df.loc[mask, "labresulttext"] = df.loc[mask, "labresulttext"].str.lstrip(">")
    mask = df["labresulttext"].astype(str).str.startswith("<")
    df.loc[mask, "labresulttext"] = df.loc[mask, "labresulttext"].str.lstrip("<")
    df["labresulttext"] = pd.to_numeric(df["labresulttext"], errors="coerce")

    df = df[["patientunitstayid", "labresultoffset", "labname", "labresulttext"]]

    df.rename(columns={'patientunitstayid': 'stay_id'}, inplace=True)
    df.rename(columns={'labresultoffset': 'charttime'}, inplace=True)
    df.rename(columns={'labname': 'item'}, inplace=True)
    df.rename(columns={'labresulttext': 'value'}, inplace=True)

    mapping = {'O2 Sat (%)': 'sO2'}
    df["item"] = df["item"].map(mapping).fillna(df["item"])

    base_time = pd.Timestamp("2050-01-01 00:00:00")
    df['charttime'] = (
        base_time
        + pd.to_timedelta(pd.to_numeric(df['charttime'], errors="coerce"),
                          unit="min")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Remove empty values
    df["value"] = df["value"].astype("string").str.strip()

    df = df[
        df["value"].notna()
        & df["value"].ne("")
    ]

    df.to_csv(output_file, index=False)
    print(f"输出文件保存至：{output_file}")

if __name__ == "__main__":

    filter_large_csv_by_stay_ids('../../eicu/lab.csv', './labresp.csv', 'labname')
    process_columns_rename_time('./labresp.csv', './labresp.csv')