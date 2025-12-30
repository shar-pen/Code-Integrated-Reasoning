import os
import fire
import pandas as pd

def concat_datasets(seperated_dataset_folder_path: str, output_concated_dataset_path:str):
	dataframes = []
	for filename in os.listdir(seperated_dataset_folder_path):
		if filename.endswith(".parquet"):
			file_path = os.path.join(seperated_dataset_folder_path, filename)
			df = pd.read_parquet(file_path)
			dataframes.append(df)
	
	concatenated_df = pd.concat(dataframes, ignore_index=True)
	concatenated_df.to_parquet(output_concated_dataset_path)
	print(f"Concatenated {len(dataframes)} datasets into {output_concated_dataset_path} with {len(concatenated_df)} total records.")
    
if __name__ == "__main__":
    fire.Fire(concat_datasets)
	# concat_datasets(
	# 	seperated_dataset_folder_path="data/aime",
	# 	output_concated_dataset_path="data/val/concatenated.parquet"
	# )
 
 