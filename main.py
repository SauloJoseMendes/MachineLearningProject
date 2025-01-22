from Classes.DataReader import DataReader
from Classes.Tester import Tester

# Read configuration
config = Tester.read_config("config.init")
data_type = config.get("data_type", "")
model_type = config.get("model_type", "dl")

# Dataset initialization based on data_type
if data_type == "img_feature":
    dataset = DataReader(path="data/COVID_numerics.csv", image_feature_path="data/feature_vectors.csv")
    dataset.drop_feature(["MARITAL STATUS"])
    X, T = dataset.X, dataset.Y
elif data_type == "img":
    dataset = DataReader(path="data/COVID_numerics.csv", image_dataset_path="data/COVID_IMG.csv")
    dataset.drop_feature(["MARITAL STATUS"])
    X, T = dataset.images, dataset.Y
elif data_type == "numeric":
    dataset = DataReader(path="data/COVID_numerics.csv")
    dataset.drop_feature(["MARITAL STATUS"])
    X, T = dataset.X, dataset.Y
else:
    raise ValueError("Unknown data type")

#Testing
tester = Tester(X, T, model_type)
tester.print_info()
tester.plot()