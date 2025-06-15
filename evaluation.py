from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    abs_y_true = np.abs(y_true)
    nrmse = rmse / np.mean(abs_y_true)

    print(f"{model_name}\tMAE: {mae}, RMSE: {rmse}, NRMSE: {nrmse}")
    return {"MAE": mae, "RMSE": rmse, "NRMSE": nrmse}


def evaluate_plot(y_true, y_pred, model_name, region, energy='solar'):
    plt.figure(figsize=(12,3))
    plt.plot(y_true.index, y_true, color='black', label='True Values')
    plt.plot(y_true.index, y_pred, color='red', label=model_name.capitalize(), linestyle='--')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.title(f"{region}: Actual vs {model_name.upper()} Predictions")
    plt.xlabel('Date')
    plt.ylabel(f'{energy.capitalize()} Energy')
    plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(y_true.index[:168], y_true[:168], color='black', label='True Values')
    plt.plot(y_true.index[:168], y_pred[:168], color='blue', label=model_name.capitalize(), linestyle='--')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    #plt.title(f"{region}: Actual vs {model_name.upper()} Predictions \n First week of Jan 2022")
    plt.xlabel('Date')
    plt.ylabel(f'{energy.capitalize()} Energy')
    plt.show()


