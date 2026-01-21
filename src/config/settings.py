# Central configuration for TON_IoT project

DATA_PATH = "./datasets/"
OUTPUT_PATH = "./output/"
LOG_PATH = "./output/logs/"

# Example: model parameters
def get_model_params(model_name):
    params = {
        "cnn": {"epochs": 10, "batch_size": 64},
        "tabnet": {"epochs": 20, "batch_size": 128},
    }
    return params.get(model_name, {})
