import pandas as pd

def filter_large_csv_by_stay_ids(input_file, output_file, key_col, chunksize=100_000):
    valid_ids = ['FIO2 (%)', 'FiO2', 'O2 Percentage', 'Set Fraction of Inspired Oxygen (FIO2)',
                 'Adult Con Pt/Vent SpO2', 'NIV Pt/Vent SpO2_5', 'PULSE OX RESULTS VT',
                 'SaO2', 'LPM O2', 'Oxygen Flow Rate']

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

    mapping = {
        'FiO2': 'FiO2',
        'FIO2 (%)': 'FiO2',
        'O2 Percentage': 'FiO2',
        'Set Fraction of Inspired Oxygen (FIO2)': 'FiO2',
        'Adult Con Pt/Vent SpO2': 'SpO2',
        'NIV Pt/Vent SpO2_5': 'SpO2',
        'PULSE OX RESULTS VT': 'SpO2',
        'SaO2': 'SpO2',
        'LPM O2': 'LPM O2',
        'Oxygen Flow Rate': 'LPM O2',
    }
    df["respchartvaluelabel"] = df["respchartvaluelabel"].map(mapping).fillna(df["respchartvaluelabel"])

    # Remove O2 flow with %
    mask = (
        (df["respchartvaluelabel"] == 'LPM O2')
        & (df["respchartvalue"].astype(str).str.endswith("%"))
    )
    df = df[~mask]

    # Remove %
    mask = df["respchartvalue"].astype(str).str.endswith("%")
    df.loc[mask, "respchartvalue"] = df.loc[mask, "respchartvalue"].str.rstrip("%")
    df["respchartvalue"] = pd.to_numeric(df["respchartvalue"], errors="coerce")

    # Process FiO2
    mask = df["respchartvaluelabel"] == 'FiO2'

    df.loc[mask & df["respchartvalue"].gt(0.2) & df["respchartvalue"].le(1), "respchartvalue"] *= 100

    valid_mask = (
        (df["respchartvaluelabel"] != 'FiO2')
        | (
            df["respchartvalue"].ge(20)
            & df["respchartvalue"].le(100)
        )
    )
    df = df.loc[valid_mask]

    # Process SpO2
    mask = df["respchartvaluelabel"] == 'SpO2'
    df.loc[mask & df["respchartvalue"].gt(100)] = 100

    df = df[["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"]]
    
    df.rename(columns={'patientunitstayid': 'stay_id'}, inplace=True)
    df.rename(columns={'respchartoffset': 'charttime'}, inplace=True)
    df.rename(columns={'respchartvaluelabel': 'item'}, inplace=True)
    df.rename(columns={'respchartvalue': 'value'}, inplace=True)

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

    df.sort_values(by=['stay_id', 'charttime'], inplace=True)

    df.to_csv(output_file, index=False)
    print(f"输出文件保存至：{output_file}")

if __name__ == "__main__":

    filter_large_csv_by_stay_ids('../../eicu/respiratoryCharting.csv', './chartresp.csv', 'respchartvaluelabel')
    process_columns_rename_time('./chartresp.csv', './chartresp.csv')