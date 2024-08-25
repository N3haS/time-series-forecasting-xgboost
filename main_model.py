import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import pickle

df1 = pd.read_csv(r'video_180s720p_07.csv')
df2 = pd.read_csv(r'video_210s480p_01.csv')
df3 = pd.read_csv(r'video_x1_02.csv')

df = pd.concat([df1, df2, df3]) 

print(df.tail())

X = df.drop(columns=['proto', 'ip_src', 'ip_dst', 'src_port', 'dst_port'])
y = df['data_len']

df['ds'] = pd.to_datetime(df['time'], unit='s')
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=2000)

xgb_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=["mae", "rmse"], verbose=True)

y_pred = xgb_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)

results = xgb_reg.evals_result()
train_loss = results['validation_0']['rmse']

results_df = pd.DataFrame({'Epoch': range(1, len(train_loss)+1),
                           'MAE': [mae]*len(train_loss),
                           'RMSE': [rmse]*len(train_loss),
                           'Training Loss (RMSE)': train_loss})

print(results_df)

future_timestamps = pd.date_range(start=df['ds'].iloc[-1], periods=900, freq='T')

future_df = pd.DataFrame({'ds': future_timestamps})

future_df = pd.concat([future_df, X.iloc[0:0].reset_index(drop=True)], axis=1)

future_predictions = xgb_reg.predict(future_df.drop(columns=['ds']))

forecast_df = pd.DataFrame({'ds': future_timestamps, 'yhat': future_predictions})

print(forecast_df.head())

