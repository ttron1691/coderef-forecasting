# coderef-forecasting
## Methodology
### Stationarity
### Seasonality
### Autoregressive Models
## AR Model
We consider the definition of the approach which is given by
$$X_t=\sum_{i=1}^p\varphi_i X_{t-i}+\varepsilon_t$$
## MA Model
The definition looks as follows
$$X_t=\sum_{i=1}^p\varphi_i X_{t-i}+\varepsilon_t$$
## ARMA Model
The (Autoregressive moving average) model describes a (weakly) stationary stochastic process in terms of polynomials. The model combines an autoregressive (AR) approach with an moving average (MA) approach. The notation of the ARMA model is given as follows
$$X_t=\sum_{i=1}^p\varphi_i X_{t-i}+\varepsilon_t$$
## ARIMA Model
The ARIMA (Autoregressive Integrated Moving Average) model is a generalization of the ARMA model for the purpose of including non-stationarity by using an initial differencing (integration step). 
## Prophet Model
## ANN Models
## Libraries
We can use the statsmodels Python package which comes with routines for using ARIMA models. For this purpose we include the following libraries
```Python
import statsmodels.api as sm
```
The evaluation of the $ARIMA(p,d,q)(P,D,Q)$ model with the ARIMA parameters $(p,d,q)$ and the ARIMA seasonality paramters $(P,D,Q)$ works as follows
```Python
# Define parameter tuples
param_model = (p, d, q)
param_seasonal = (P, D, Q)

# Define the model
model = sm.tsa.statespace.SARIMAX(y_data, order=param_model, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

# Fit the mode
results = model.fit()
```
In order to visualize diagnostics we can use the following methods
```Python
results.plot_diagnostics(figsize=(15, 12))
plt.show()
```
For the purpose of validation we can use the prediciton method for either dynamic or non-dynamic (one-step ahead forecast) processes 
```Python
model_prediction = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_conf_int = model_prediction.conf_int()
```
In order to run the forecast for multiple steps ahead we use the following routine
```Python
model_prediction_ahead = results.get_forecast(steps=500)
ahead_conf_int = model_prediction_ahead.conf_int()

```

```Python

```

```Python

```

```Python

```

```Python

```

```Python

```

```Python

```

```Python

```

