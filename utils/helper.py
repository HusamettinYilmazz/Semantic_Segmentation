import os 
import yaml

class Config:
    def __init__(self, config_dict):
        self.experiment = config_dict.get("experiment", {})
        self.data = config_dict.get("data", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})

    def __reper__(self):
        return f"Config(experiment={self.experiment}, data={self.data} model={self.model}, training={self.training})"
     

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config