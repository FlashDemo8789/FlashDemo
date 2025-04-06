import os

# optional define how many Monte Carlo runs
MC_SIMULATION_RUNS= 2000

MONGO_URI= os.getenv("MONGO_URI","mongodb://localhost:27017/flash_dna")

XGB_PARAMS= {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": 42,
    "enable_categorical": False
}
