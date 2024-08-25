import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt

# Define a function to introduce a temporary shift in traffic distribution across ports
def introduce_port_traffic_shift(df, port_range, traffic_shift):
    # Define the mask for the specified port range
    mask = (df['dst_port'] >= port_range[0]) & (df['dst_port'] <= port_range[1])
    # Introduce the traffic shift to the selected port range
    df.loc[mask, 'data_len'] += traffic_shift
    return df

df['ip_src'] = pd.factorize(df['ip_src'])[0]
df['ip_dst'] = pd.factorize(df['ip_dst'])[0]

X6 = df.drop(columns=['time'])
y6 = df['time']

X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size=0.2, random_state=42)

y_test6 = y_test6.astype(float)

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=2000)

X_train6 = X_train6.drop(columns=['ds'], errors='ignore')
X_test6 = X_test6.drop(columns=['ds'], errors='ignore')

# Introduce a traffic shift to ports in the range 1000-2000
X_train6 = introduce_port_traffic_shift(X_train6, (1000, 2000), 100)
X_test6 = introduce_port_traffic_shift(X_test6, (1000, 2000), 100)
early_stopping_rounds = 50

xgb_reg.fit(X_train6, y_train6, eval_set=[(X_test6, y_test6)], eval_metric=["mae", "rmse"],early_stopping_rounds=early_stopping_rounds, verbose=True)

y_pred6 = xgb_reg.predict(X_test6)

mae = mean_absolute_error(y_test6, y_pred6)
rmse = mean_squared_error(y_test6, y_pred6, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

naive_predictions = y_train1.iloc[-1]  
y_naive_pred = np.full_like(y_test6, naive_predictions)

baseline_mae = mean_absolute_error(y_test6, y_naive_pred)
baseline_rmse = mean_squared_error(y_test6, y_naive_pred, squared=False)

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

results = xgb_reg.evals_result()
train_loss = results['validation_0']['rmse']

average_percentage_error_mae = np.mean(np.abs((y_test6 - y_pred6) / y_test6)) * 100

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

#Alternatively, you can redistribute traffic to mess with port usage
# Define the targeted ports for redistribution during chaos
target_ports = [53, 443, 53954]

# Define the shift period for chaos
chaos_shift_period = 3600  # seconds

# Randomly redistribute traffic across ports for the chaos shift period
chaos_shift_start_time = df['time'].min() + np.random.randint(0, df['time'].max() - chaos_shift_period)
chaos_shift_end_time = chaos_shift_start_time + chaos_shift_period

# Select rows within the chaos shift period
chaos_shift_data = df[(df['time'] >= chaos_shift_start_time) & (df['time'] <= chaos_shift_end_time)]

# Loop through each row in the selected data and randomly assign ports for chaos
for index, row in chaos_shift_data.iterrows():
    if row['dst_port'] in target_ports:
        # Randomly select a new port from the target ports for chaos
        new_port = random.choice(target_ports)

        # Update the port in the DataFrame for chaos
        df.at[index, 'dst_port'] = new_port

# After the chaos shift period, the traffic distribution returns to normal

X7 = df.drop(columns=['dst_port','proto','ip_src','ip_dst','time','src_port'])  # Exclude only the 'time' column
y7 = df['dst_port']

X_train7, X_test7, y_train7, y_test7 = train_test_split(X7, y7, test_size=0.2, random_state=42)

y_test7 = y_test7.astype(float)

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=2000, reg_alpha=0.1, reg_lambda = 0.5 )
early_stopping_rounds = 40

xgb_reg.fit(X_train7, y_train7, eval_set=[(X_test7, y_test7)], eval_metric=["mae", "rmse"], early_stopping_rounds=early_stopping_rounds, verbose=True)

y_pred7 = xgb_reg.predict(X_test7)

mae = mean_absolute_error(y_test7, y_pred7)
rmse = mean_squared_error(y_test7, y_pred7, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

window_size = 3  
baseline_predictions = X7.iloc[:, 0].rolling(window=window_size).mean().iloc[-1]

naive_predictions = y_train1.iloc[-1] 
y_naive_pred = np.full_like(y_test7, naive_predictions)

baseline_mae = mean_absolute_error(y_test6, y_naive_pred)
baseline_rmse = mean_squared_error(y_test6, y_naive_pred, squared=False)

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)

results = xgb_reg.evals_result()
train_loss = results['validation_0']['rmse']

average_percentage_error_mae = np.mean(np.abs((y_test6 - y_pred6) / y_test6)) * 100

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

print(df.shape)
print(df.head())

print(f"Average Percentage Error in MAE: {average_percentage_error_mae:.8f}%")
print(f"Mean Percentage Change in MAE (MPCM-MAE): {mpcm_mae:.2f}%")
print(f"Chaos MAE: {chaos_mae}")
print(f"Change in MAE: {change_in_mae}")
print(f"Baseline MAE : {baseline_mae}")
print(f"Time to Return to Baseline MAE: {time_to_baseline} epochs")

print("Minimum MAE:", min(mae_history))
print("Maximum MAE:", max(mae_history))

plt.figure(figsize=(14, 7))
plt.plot(results_df['Epoch'], results_df['MAE'], label='Validation MAE')
plt.plot(results_df['Epoch'], results_df['RMSE'], label='Validation RMSE')
plt.axhline(y=baseline_mae, color='r', linestyle='--', label='Baseline MAE')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training and Validation Metrics Over Epochs')
plt.legend()
plt.show()

future_timestamps = df['time'].max() + np.arange(1, 6) * 60  # 5 new rows, incrementing by 1 minute

# Create new data points based on the last row of the DataFrame
last_row = df.iloc[-1].copy()
future_data = []
for ts in future_timestamps:
    new_row = last_row.copy()
    new_row['time'] = ts
    future_data.append(new_row)

future_df = pd.DataFrame(future_data)

future_df = future_df.drop(columns=['dst_port', 'proto', 'ip_src', 'ip_dst', 'time', 'src_port'])

y_forecast = xgb_reg.predict(future_df)

forecast_df = pd.DataFrame({
    'time': pd.to_datetime(future_timestamps, unit='s'), 
    'forecast': y_forecast
})

print(forecast_df)
