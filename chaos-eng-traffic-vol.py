# Simulate a sudden drop in overall network traffic volume
# For example, reducing the entire traffic volume by 50%
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt

df['altered_traffic_volume'] = df['data_len'] * 0.5

X1 = df.drop(columns=['proto', 'ip_src', 'ip_dst', 'src_port', 'dst_port', 'data_len'])
y1 = df['altered_traffic_volume']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

X_train1 = X_train1.drop(columns=['ds'], errors='ignore')
X_test1 = X_test1.drop(columns=['ds'], errors ='ignore')
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=2000)

xgb_reg.fit(X_train1, y_train1, eval_set=[(X_test1, y_test1)], eval_metric=["mae", "rmse"], verbose=True)

y_pred1 = xgb_reg.predict(X_test1)

mae = mean_absolute_error(y_test1, y_pred1)
rmse = mean_squared_error(y_test1, y_pred1, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

# Calculate the naive forecast baseline
naive_predictions = y_train1.iloc[-1] 
y_naive_pred = np.full_like(y_test1, naive_predictions)

# Calculate baseline metrics
baseline_mae = mean_absolute_error(y_test1, y_naive_pred)
baseline_rmse = mean_squared_error(y_test1, y_naive_pred, squared=False)

results = xgb_reg.evals_result()
train_loss = results['validation_0']['rmse']

# Calculate average percentage error in MAE
average_percentage_error_mae = np.mean(np.abs((y_test1 - y_pred1) / y_test1)) * 100

# Calculate Mean Percentage Change in MAE (MPCM-MAE)
mae_history = results['validation_0']['mae']
mpcm_mae = np.mean(np.abs(np.diff(mae_history) / mae_history[:-1])) * 100

# Calculate Chaos MAE (the maximum MAE during training)
chaos_mae = np.max(mae_history)

# Calculate the change in MAE from baseline
change_in_mae = baseline_mae - mae

# Calculate the time to return to baseline MAE
baseline_epoch = np.argmin(mae_history)
stabilization_epoch = next((i for i, v in enumerate(mae_history[baseline_epoch:], start=baseline_epoch) if v > baseline_mae * 1.05), len(mae_history))
time_to_baseline = stabilization_epoch - baseline_epoch

results_df = pd.DataFrame({'Epoch': range(1, len(train_loss)+1),
                           'MAE': [mae]*len(train_loss),
                           'RMSE': [rmse]*len(train_loss),
                           'Training Loss (RMSE)': train_loss})

print(results_df)

print(f"Average Percentage Error in MAE: {average_percentage_error_mae:.2f}%")
print(f"Mean Percentage Change in MAE (MPCM-MAE): {mpcm_mae:.2f}%")
print(f"Chaos MAE: {chaos_mae}")
print(f"Change in MAE: {change_in_mae}")
print(f"Baseline MAE : {baseline_mae}")
print(f"Time to Return to Baseline MAE: {time_to_baseline} epochs")

#plot the results
plt.figure(figsize=(14, 7))
plt.plot(results_df['Epoch'], results_df['MAE'], label='Validation MAE')
plt.plot(results_df['Epoch'], results_df['RMSE'], label='Validation RMSE')
plt.axhline(y=baseline_mae, color='r', linestyle='--', label='Baseline MAE')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training and Validation Metrics Over Epochs')
plt.legend()
plt.show()

# Generate future timestamps for forecasting
future_timestamps = pd.date_range(start=df['time'].iloc[-1], periods=900, freq='T')

future_df = pd.DataFrame({'time': future_timestamps, 'altered_traffic_volume': np.nan})

future_df['time'] = future_df['time'].astype(np.int64) // 10**9  # Convert to seconds

future_predictions = xgb_reg.predict(future_df)

forecast_df = pd.DataFrame({'time': future_timestamps, 'forecasted_altered_traffic_volume': future_predictions})

print(forecast_df.head())


# Simulate a sudden Increase in overall network traffic volume
# For example, reducing the entire traffic volume by 50%
df['altered_traffic_volume'] = df['data_len'] * 1.5 

X4 = df.drop(columns=['proto', 'ip_src', 'ip_dst', 'src_port', 'dst_port', 'data_len'])
y4 = df['altered_traffic_volume']

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2, random_state=42)

X_train4 = X_train4.drop(columns=['ds'], errors='ignore')
X_test4 = X_test4.drop(columns=['ds'], errors='ignore')

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=2000)

xgb_reg.fit(X_train4, y_train4, eval_set=[(X_test4, y_test4)], eval_metric=["mae", "rmse"], verbose=True)

y_pred4 = xgb_reg.predict(X_test4)

mae = mean_absolute_error(y_test4, y_pred4)
rmse = mean_squared_error(y_test4, y_pred4, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

naive_predictions = y_train4.iloc[-1]
y_naive_pred = np.full_like(y_test4, naive_predictions)

baseline_mae = mean_absolute_error(y_test4, y_naive_pred)
baseline_rmse = mean_squared_error(y_test4, y_naive_pred, squared=False)

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

results = xgb_reg.evals_result()
train_loss = results['validation_0']['rmse']

average_percentage_error_mae = np.mean(np.abs((y_test4 - y_pred4) / y_test4)) * 100

mae_history = results['validation_0']['mae']
mpcm_mae = np.mean(np.abs(np.diff(mae_history) / mae_history[:-1])) * 100

chaos_mae = np.max(mae_history)

change_in_mae = baseline_mae - chaos_mae

baseline_epoch = np.argmin(mae_history)
stabilization_epoch = next((i for i, v in enumerate(mae_history[baseline_epoch:], start=baseline_epoch) if v > baseline_mae * 1.05), len(mae_history))
time_to_baseline = stabilization_epoch - baseline_epoch

results_df = pd.DataFrame({'Epoch': range(1, len(train_loss)+1),
                           'MAE': mae_history,
                           'RMSE': train_loss,
                           'Training Loss (RMSE)': train_loss})

print(results_df)

print(f"Average Percentage Error in MAE: {average_percentage_error_mae:.2f}%")
print(f"Mean Percentage Change in MAE (MPCM-MAE): {mpcm_mae:.2f}%")
print(f"Chaos MAE: {chaos_mae}")
print(f"Change in MAE: {change_in_mae}")
print(f"Baseline MAE : {baseline_mae}")
print(f"Time to Return to Baseline MAE: {time_to_baseline} epochs")

plt.figure(figsize=(14, 7))
plt.plot(results_df['Epoch'], results_df['MAE'], label='Validation MAE')
plt.plot(results_df['Epoch'], results_df['RMSE'], label='Validation RMSE')
plt.axhline(y=baseline_mae, color='r', linestyle='--', label='Baseline MAE')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training and Validation Metrics Over Epochs')
plt.legend()
plt.show()

future_timestamps4 = pd.date_range(start=df['time'].iloc[-1], periods=900, freq='T')

future_df4 = pd.DataFrame({'time': future_timestamps4, 'altered_traffic_volume': np.nan})

future_df4['time'] = future_df4['time'].astype(np.int64) // 10**9  # Convert to seconds

future_predictions4 = xgb_reg.predict(future_df4)

forecast_df4 = pd.DataFrame({'time': future_timestamps4, 'forecasted_altered_data_len(yhat)': future_predictions4})

print(forecast_df4.head())
