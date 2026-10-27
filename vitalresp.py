import pandas as pd

def filter_csv_by_cols(input_file, output_file, chunksize=100000):
    columns = ['sao2']

    first_chunk = True
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        filtered_chunk = chunk.dropna(subset=columns, how="all")
        filtered_chunk = filtered_chunk[["patientunitstayid", "observationoffset", 'sao2']]
        filtered_chunk.to_csv(output_file, mode='w' if first_chunk else 'a', index=False, header=first_chunk)
        first_chunk = False

    print(f"Done, result saved to: {output_file}")

def process_columns_rename_time(data_file, output_file):

    df = pd.read_csv(data_file)

    df = df[["patientunitstayid", "observationoffset", 'sao2']]

    df.rename(columns={'patientunitstayid': 'stay_id'}, inplace=True)
    df.rename(columns={'observationoffset': 'charttime'}, inplace=True)
    df.rename(columns={'sao2': 'value'}, inplace=True)

    base_time = pd.Timestamp("2050-01-01 00:00:00")
    df['charttime'] = (
        base_time
        + pd.to_timedelta(pd.to_numeric(df['charttime'], errors="coerce"),
                          unit="min")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(output_file, index=False)
    print(f"输出文件保存至：{output_file}")

if __name__ == "__main__":
    filter_csv_by_cols('../../eicu/vitalPeriodic.csv', './vitalresp.csv')
    process_columns_rename_time('./vitalresp.csv', './vitalresp.csv')