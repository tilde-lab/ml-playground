# Evaluation metrics: R2 coefficient


def r2_score(model, total):
    return (
        len(total)
        * (model.training_rmse**2)
        / sum((total["Value"] - total["Value"].mean()) ** 2)
    )
