from yaml import safe_load


def load_config(*args: str):
    config = dict()
    for file in args:
        with open(file, 'r') as f:
            config = config | safe_load(f)
    return config
