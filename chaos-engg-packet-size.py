# Simulate a temporary shift in packet size distribution
# For example, increasing proportion of smaller packets
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt

df['altered_data_len'] = df['data_len'] * 0.8 

X2 = df.drop(columns=['proto', 'ip_src', 'ip_dst', 'src_port', 'dst_port', 'data_len', 'altered_traffic_volume'])
y2 = df['altered_data_len']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

X_train2 = X_train2.drop(columns=['ds'], errors='ignore')
X_test2 = X_test2.drop(columns=['ds'], errors='ignore')

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=2000)

xgb_reg.fit(X_train2, y_train2, eval_set=[(X_test2, y_test2)], eval_metric=["mae", "rmse"], verbose=True)

y_pred2 = xgb_reg.predict(X_test2)

mae = mean_absolute_error(y_test2, y_pred2)
rmse = mean_squared_error(y_test2, y_pred2, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

naive_predictions = y_train2.iloc[-1] 
y_naive_pred = np.full_like(y_test2, naive_predictions)

baseline_mae = mean_absolute_error(y_test2, y_naive_pred)
baseline_rmse = mean_squared_error(y_test2, y_naive_pred, squared=False)

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

results = xgb_reg.evals_result()
train_loss = results['validation_0']['rmse']

average_percentage_error_mae = np.mean(np.abs((y_test2 - y_pred2) / y_test2)) * 100

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

future_timestamps2 = pd.date_range(start=df['time'].iloc[-1], periods=900, freq='T')

future_df2 = pd.DataFrame({'time': future_timestamps2, 'altered_data_len': np.nan})

future_df2['time'] = future_df2['time'].astype(np.int64) // 10**9  # Convert to seconds

future_predictions2 = xgb_reg.predict(future_df2)

forecast_df2 = pd.DataFrame({'time': future_timestamps2, 'forecasted_altered_data_len(yhat)': future_predictions2})

print(forecast_df2.head())
