import optuna

def hyperparam_space(trial: optuna.Trial) -> dict:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    
    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "batch_size": batch_size,
        "optimizer": optimizer,
    }