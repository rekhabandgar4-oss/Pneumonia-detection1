import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print("Config type:", type(config))
print("Config keys:", config.keys() if config else "Empty")
print("Data section:", config.get('data') if config else "Not found")
