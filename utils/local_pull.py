import kagglehub
import shutil

path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")

print("CPath to dataset files:", path)

shutil.copytree(path, './data/')
