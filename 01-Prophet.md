## 快速开始

1. df要求有两列: ds和y

```python
import pandas as pd
from profhet import Prophet
df = pd.read_csv(path)
m = Prophet()
m.fit(df)

# future只有一列: ds
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

2. 可视化

```python
fig1 = m.plot(forecast)
# 趋势, weekly, yearly
fig2 = m.plot_components(forecast)


from prophet.plot import plot_plotly, plot_components_plotly
plot_plotly(m, forecast)
plot_components_plotly(m, forecast)
```

## 饱和预测

1. 预测增长

```python
df['cap'] = 8.5
m = Prophet(growth='logistic')
m.fit(df)

future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5 # 上界
fcst = m.predict(future)
fig = m.plot(fcst)
```

2. 饱和最小值, 使用具有最小值的`logistic`增长, 必须指定最大值

```python
df['y'] = 10 - df['y']
df['cap'] = 6
df['floor'] = 1.5
future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)
```

## 趋势变化点

1. prophet能够自动检查趋势变化点, 默认25个, 通过参数`n_changepoints`来调整

```py
from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```

2. 调整趋势灵活性, 通过参数`changepoint_prior_scale`来调整稀疏先验的强度, 默认为 0.05

```python
m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

m = Prophet(changepoint_prior_scale=0.001)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)
```

3. 指定变更点的位置, 通过参数`changepoints`来手动指定潜在变化点的位置

```python
m = Prophet(changepoints=['2014-01-01'])
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)
```

## 季节性, 假期效应和回归量

1. 自行设定节假日/重要日期

```python
playoffs = pd.DataFrame({
    'holiday': 'playoff',
    'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                          '2010-01-24', '2010-02-07', '2011-01-08',
                          '2013-01-12', '2014-01-12', '2014-01-19',
                          '2014-02-02', '2015-01-11', '2016-01-17',
                          '2016-01-24', '2016-02-07']),
    'lower_window': 0,
    'upper_window': 1
})
superbowls = pd.DataFrame({
    'holiday': 'superbowl',
    'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
    'lower_window': 0,
    'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
m = Prophet(holidays=holidays)
forecast = m.fit(df).predict(future)

# 预测的节假日
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][['ds', 'playoff', 'superbowl']][-10:]

# 可视化
fig = m.plot_components(forecast)
```

2. 内置各个国家节假日

```python
m = Prophet(holidays=holidays)
m.add_country_holidays(country_name='CN')
m.fit(df)
# 节假日名称
m.train_holiday_names

forecast = m.predict(future)
fig = m.plot_components(forecast)
```

3. 自定义季节性, 默认添加了每周,每年的季节性, 通过参数`add_seasonality`来添加季节性

```python
# 每月的季节性替换每周的季节性
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```

4. 取决于其他因素的季节性, 比如是否休赛期

```python
def is_nfl_season(ds):
    date = pd.to_datetime(ds)
    return (date.month > 8 or date.month < 2)

df['on_season'] = df['ds'].apply(is_nfl_season)
df['off_season'] = ~df['ds'].apply(is_nfl_season)

m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

# 预测区间需要做同样的处理
future['on_season'] = future['ds'].apply(is_nfl_season)
future['off_season'] = ~future['ds'].apply(is_nfl_season)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```

5. 节假日和季节性的先验

```python
m = Prophet(holidays=holidays, holidays_prior_scale=0.05).fit(df)
forecast = m.predict(future)
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][['ds', 'playoff', 'superbowl']][-10:]

m = Prophet()
m.add_seasonality(name='weekly', period=7, fourier_order=3, prior_scale=0.1)
```

6. 附加的回归量

```python
def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0
df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

m = Prophet()
m.add_regressor('nfl_sunday')
m.fit(df)

future['nfl_sunday'] = future['ds'].apply(nfl_sunday)

forecast = m.predict(future)
fig = m.plot_components(forecast)
```

## 乘法季节性

1. `Prophet`默认是加法季节性, 表示季节性的效应加到趋势中从而得到`forecast`

```python
df = pd.read_csv('../examples/example_air_passengers.csv')
m = Prophet() # 默认加法季节性
m.fit(df)
future = m.make_future_dataframe(50, freq='MS')
forecast = m.predict(future)
fig = m.plot(forecast)


m = Prophet(seasonality_mode='multiplicative') # 乘法季节性
m.fit(df)
forecast = m.predict(future)
fig = m.plot(forecast)
fig = m.plot_components(forecast)

m = Prophet(seasonality_mode='multiplicative') # 乘法季节性
m.add_seasonality('quarterly', period=91.25, fourier_order=8, mode='additive') # 季度季节性为加法季节性
m.add_regressor('regressor', mode='additive') # 附加回归量为加法季节性
```

## 不确定区间

预测的不确定性的来源有三: 趋势的不确定性, 季节性估计的不确定性, 额外的观察噪声

1. 趋势的不确定性

```python
forecast = Prophet(interval_width=0.95).fit(df).predict(future)
```

2. 季节性的不确定性, `MCMC`采样

```python
m = Prophet(mcmc_samples=300)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```
