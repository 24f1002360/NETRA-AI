import yaml
from app import create_app

with open("configs/app.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = create_app(config)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=config.get("app", {}).get("port", 5000),
        debug=config.get("app", {}).get("debug", True)
    )
