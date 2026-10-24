import pandas as pd

def filter_large_csv_by_stay_ids(input_file, output_file, key_col, chunksize=100_000):
    valid_ids = ['O2 Saturation', 'SpO2'] # Both are SpO2

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

    df = df[["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel", "nursingchartvalue"]]

    df.rename(columns={'patientunitstayid': 'stay_id'}, inplace=True)
    df.rename(columns={'nursingchartoffset': 'charttime'}, inplace=True)
    df.rename(columns={'nursingchartcelltypevallabel': 'item'}, inplace=True)
    df.rename(columns={'nursingchartvalue': 'value'}, inplace=True)

    mapping = {'O2 Saturation': 'SpO2'}
    df["item"] = df["item"].map(mapping).fillna(df["item"])

    base_time = pd.Timestamp("2050-01-01 00:00:00")
    df['charttime'] = (
        base_time
        + pd.to_timedelta(pd.to_numeric(df['charttime'], errors="coerce"),
                          unit="min")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(output_file, index=False)
    print(f"输出文件保存至：{output_file}")

def process_spo2(
    input_path="input.csv",
    output_path="output.csv"
):
    df = pd.read_csv(input_path)

    valid_mask = (df["value"].ge(0) & df["value"].le(100))
        
    df = df.loc[valid_mask].copy()

    df.to_csv(output_path, index=False)

    print(f"处理后记录数: {len(df)}")

    return df

if __name__ == "__main__":

    filter_large_csv_by_stay_ids('../../eicu/nurseCharting.csv', './nurseresp.csv', 'nursingchartcelltypevallabel')
    process_columns_rename_time('./nurseresp.csv', './nurseresp.csv')
    process_spo2('./nurseresp.csv', './nurseresp.csv')