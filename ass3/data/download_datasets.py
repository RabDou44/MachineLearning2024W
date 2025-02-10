import kagglehub
import  os
import shutil

# Download latest version
path_alzheimer = kagglehub.dataset_download("ankushpanday1/alzheimers-prediction-dataset-global")
path_placement = kagglehub.dataset_download("ruchikakumbhar/placement-prediction-dataset")
common_path = os.path.commonpath([path_alzheimer, path_placement])

destination_path = os.path.join(os.getcwd(),"data","our")

for source_path in [path_alzheimer, path_placement]:
    for file in os.listdir(source_path):
        source_file = os.path.join(source_path, file)
        end_file = os.path.join(destination_path,file)
        shutil.copy(source_file, end_file)

shutil.rmtree(common_path, ignore_errors=True)