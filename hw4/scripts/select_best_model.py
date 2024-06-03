import json

def main():
    with open("/tmp/best_log_reg.json", "r") as f:
        best_log_reg = json.load(f)

    with open("/tmp/best_rf.json", "r") as f:
        best_rf = json.load(f)

    with open("/tmp/best_log_reg_spark.json", "r") as f:
        best_log_reg_spark = json.load(f)

    with open("/tmp/best_rf_spark.json", "r") as f:
        best_rf_spark = json.load(f)

    all_models = {
        "log_reg_standard": best_log_reg,
        "rf_standard": best_rf,
        "log_reg_spark": best_log_reg_spark,
        "rf_spark": best_rf_spark,
    }

    best_model = max(all_models, key=lambda k: all_models[k]["best_score"])

    print(f"The best model is: {best_model}")
    print(f"Best parameters: {all_models[best_model]['best_params']}")
    print(f"Best score: {all_models[best_model]['best_score']}")

if __name__ == "__main__":
    main()
