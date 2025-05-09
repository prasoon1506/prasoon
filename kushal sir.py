import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
df = pd.read_csv("DHS_Daily_Report_2020.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df[['Date', 'Total.Individuals.in.Shelter', 'Easter', 'Thanksgiving', 'Christmas', 'Temperature']]
df = df.rename(columns={'Date': 'ds', 'Total.Individuals.in.Shelter': 'y'})
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'])
plt.xlabel('Time')
plt.ylabel('Shelter Demand')
plt.title('Shelter Demand Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
easter_dates = df[df['Easter'] == 1]['ds'].tolist()
easter = pd.DataFrame({'holiday': 'easter','ds': easter_dates,'lower_window': -4,'upper_window': 2})
thanksgiving_dates = df[df['Thanksgiving'] == 1]['ds'].tolist()
thanksgiving = pd.DataFrame({'holiday': 'thanksgiving','ds': thanksgiving_dates,'lower_window': -3,'upper_window': 1})
holidays = pd.concat([easter, thanksgiving])
m = Prophet(holidays=holidays,yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False,seasonality_mode='multiplicative',seasonality_prior_scale=10,holidays_prior_scale=10,changepoint_prior_scale=0.05)
m.add_regressor('Christmas')
m.add_regressor('Temperature')
m.fit(df)
df_cv = cross_validation(model=m,horizon='31 days',period='7 days',initial='2300 days')
cv_metrics = performance_metrics(df_cv)
print(cv_metrics[['mse', 'rmse', 'mae', 'mape']]
param_grid = {'changepoint_prior_scale': [0.05, 0.1],'seasonality_prior_scale': [5, 10, 15],'holidays_prior_scale': [5, 10],'seasonality_mode': ['multiplicative', 'additive']}
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []
for params in all_params:
    print(f"Fitting with parameters: {params}")
    m = Prophet(yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False,holidays=holidays,seasonality_mode=params['seasonality_mode'],seasonality_prior_scale=params['seasonality_prior_scale'],holidays_prior_scale=params['holidays_prior_scale'],changepoint_prior_scale=params['changepoint_prior_scale'])
    m.add_regressor('Christmas')
    m.add_regressor('Temperature')
    m.fit(df)
    df_cv = cross_validation(model=m,horizon='31 days',period='14 days',initial='2400 days')
    cv_metrics = performance_metrics(df_cv)
    rmses.append(cv_metrics['rmse'].values[0])
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
best_params = all_params[np.argmin(rmses)]
print('Best parameters:')
print(best_params)
print(f'Best RMSE: {min(rmses)}')
df = pd.read_csv("Udemy_wikipedia_visits.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.rename(columns={'Date': 'ds', 'Visits': 'y'})
easter_dates = df[df['Holi'] == 1]['ds'].tolist()
easter = pd.DataFrame({'holiday': 'easter','ds': easter_dates,'lower_window': -2,'upper_window': 4})
christmas_dates = df[df['Christmas'] == 1]['ds'].tolist()
christmas = pd.DataFrame({'holiday': 'christmas','ds': christmas_dates,'lower_window': -6,'upper_window': 3})
holidays = easter
training = df[df['ds'] < '2024-06-01'][['ds', 'y', 'Christmas', 'Black.Friday']]
test = df[df['ds'] >= '2020-12-01'][['ds', 'y', 'Christmas', 'Black.Friday']]
m = Prophet(yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False,seasonality_mode="multiplicative",holidays=holidays,seasonality_prior_scale=10,holidays_prior_scale=10,changepoint_prior_scale=0.05)
m.add_regressor('Christmas')
m.add_regressor('Black.Friday')
m.fit(training)
future = m.make_future_dataframe(periods=len(test))
future_with_regressors = pd.merge(future, df[['ds', 'Black.Friday', 'Christmas']], on='ds', how='left')
forecast = m.predict(future_with_regressors)
fig1 = m.plot(forecast)
plt.title('Forecast')
plt.tight_layout()
plt.show()
fig2 = m.plot_components(forecast)
plt.tight_layout()
plt.show()
fig3 = m.plot(forecast)
a = add_changepoints_to_plot(fig3.gca(), m, forecast)
plt.title('Forecast with Changepoints')
plt.tight_layout()
plt.show()
predictions = forecast['yhat'].tail(len(test)).values
actuals = test['y'].values
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")
