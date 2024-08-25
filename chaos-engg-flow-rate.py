import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
import random

# Define a function to introduce jitter in packet transmission times
def introduce_jitter(df, src_ip, dst_ip, jitter_amount):
    mask = (df['ip_src'] == src_ip) & (df['ip_dst'] == dst_ip)
    df.loc[mask, 'time'] += np.random.uniform(-jitter_amount, jitter_amount, size=np.sum(mask))
    return df

# Introduce jitter to specific IP paths
df = introduce_jitter(df, '192.168.1.80', '192.181.25.100', 0.2)  # Introducing jitter of ±0.2 seconds

# Encode categorical variables
df['ip_src'] = pd.factorize(df['ip_src'])[0]
df['ip_dst'] = pd.factorize(df['ip_dst'])[0]

X = df.drop(columns=['time'])  # Exclude only the 'time' column
y = df['time']

X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y, test_size=0.2, random_state=42)

y_test5 = y_test5.astype(float)

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=2000)
early_stopping_rounds = 50 #helps in dealing with overfitting of model

xgb_reg.fit(X_train5, y_train5, eval_set=[(X_test5, y_test5)], eval_metric=["mae", "rmse"], early_stopping_rounds=early_stopping_rounds, verbose=True)

y_pred5 = xgb_reg.predict(X_test5)

mae = mean_absolute_error(y_test5, y_pred5)
rmse = mean_squared_error(y_test5, y_pred5, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

naive_predictions = y_train5.iloc[-1] 
y_naive_pred = np.full_like(y_test5, naive_predictions)

baseline_mae = mean_absolute_error(y_test5, y_naive_pred)
baseline_rmse = mean_squared_error(y_test5, y_naive_pred, squared=False)

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

results = xgb_reg.evals_result()
train_loss = results['validation_0']['rmse']

if np.any(y_test5 == 0):
    print("Warning: y_test5 contains zeros, which can affect the percentage error calculation.")
    y_test5 = y_test5.replace(0, np.nan)  # Replace zeros with NaNs to avoid division by zero

y_test5 = y_test5.astype(float)

# Replace zeros with a small non-zero value in both y_test5 and y_pred5
y_test5 = y_test5.replace(0, 1e-6) 
y_pred5 = np.where(y_pred5 == 0, 1e-6, y_pred5)

average_percentage_error_mae = np.mean(np.abs((y_test5 - y_pred5) / y_test5)) * 100

mae_history = results['validation_0']['mae']
mpcm_mae = np.mean(np.abs(np.diff(mae_history) / mae_history[:-1])) * 100

chaos_mae = np.max(mae_history)

change_in_mae = baseline_mae - chaos_mae

baseline_epoch = np.argmin(mae_history)
stabilization_epoch = next((i for i, v in enumerate(mae_history[baseline_epoch:], start=baseline_epoch) if v > baseline_mae * 1.05), len(mae_history))
time_to_baseline = stabilization_epoch - baseline_epoch

def calculate_rto(mae_history, baseline_mae, threshold=0.05):
    return next((i for i, v in enumerate(mae_history) if v <= baseline_mae * (1 + threshold)), len(mae_history))

rto = calculate_rto(mae_history, baseline_mae)

# Create a DataFrame to store results
results_df = pd.DataFrame({'Epoch': range(1, len(train_loss)+1),
                           'MAE': mae_history,
                           'RMSE': train_loss,
                           'Training Loss (RMSE)': train_loss})

print(results_df)

print(f"Average Percentage Error in MAE: {average_percentage_error_mae:.8f}%")
print(f"Mean Percentage Change in MAE (MPCM-MAE): {mpcm_mae:.2f}%")
print(f"Chaos MAE: {chaos_mae}")
print(f"Change in MAE: {change_in_mae}")
print(f"Baseline MAE : {baseline_mae}")
print(f"Time to Return to Baseline MAE: {time_to_baseline} epochs")
#print(f"Recovery Time Objective (RTO): {rto} epochs")

plt.figure(figsize=(14, 7))
plt.plot(results_df['Epoch'], results_df['MAE'], label='Validation MAE')
plt.plot(results_df['Epoch'], results_df['RMSE'], label='Validation RMSE')
plt.axhline(y=baseline_mae, color='r', linestyle='--', label='Baseline MAE')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training and Validation Metrics Over Epochs')
plt.legend()
plt.show()

#method 1 of forecasting
new_data = X_test5.copy()

new_predictions = xgb_reg.predict(new_data)

print("Forecasted Time:", new_predictions)

#method 2 of forecasting
future_timestamps5 = pd.date_range(start=df['time'].iloc[-1], periods=900, freq='T')

future_df5 = pd.DataFrame({'time': future_timestamps5, 'proto': np.nan, 'data_len': np.nan,
                           'ip_src': np.nan, 'ip_dst': np.nan, 'src_port': np.nan, 'dst_port': np.nan})

# Filling featuresthat are not available with placeholder values like -1
future_df5['proto'] = -1  
future_df5['data_len'] = -1  
future_df5['src_port'] = -1  
future_df5['dst_port'] = -1  

future_df5['time'] = future_df5['time'].astype(np.int64) // 10**9  # Convert to seconds

future_predictions5 = xgb_reg.predict(future_df5.drop(columns=['time']))

forecast_df5 = pd.DataFrame({'time': future_timestamps5, 'forecasted_time_data(yhat)': future_predictions5})

print(forecast_df5.head())

