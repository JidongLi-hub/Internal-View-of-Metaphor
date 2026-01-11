from utils import *


def parse_raw_data_piece(dataset_name, row):
    if dataset_name == "CII-Bench":
        parse_result = {
                        "id": row["id"],
                        "image_name": row["local_path"].replace("images/test/", "").replace("images/dev/", ""),
                        "image_type": row["image_type"] if "choices" not in row else json.dumps(json.loads(row["image_type"])["choices"]),
                        "domain": row["domain"],
                        "explanation": row["explanation"],
                    }
        return parse_result
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def parse_raw_data(dataset_name, dataset_file):
    precessed_data = []
    if dataset_name == "CII-Bench":
        df = pd.read_parquet(dataset_file)
        for _, row in df.iterrows():
            precessed_data.append(parse_raw_data_piece(dataset_name, row))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    os.makedirs(f"data/processed_data/", exist_ok=True)
    with open(f"data/processed_data/{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(precessed_data, f, indent=4, ensure_ascii=False)
    print(f"Preprocessed data saved to data/processed_data/{dataset_name}.json")


if __name__ == "__main__":

    # 预处理 CII-Bench 数据集，提取所需字段
    dataset_name="CII-Bench"
    dataset_file = "../dataset/CII-Bench/data/dev-00000-of-00001.parquet"
    parse_raw_data(dataset_name=dataset_name, dataset_file=dataset_file)