library(fpp3)          
library(dplyr)         
library(ggplot2)       
library(fable.prophet)
library(randomForest)
library(xgboost) 
library(timetk)
library(prophet)

#Part 1: New House Listings in California from (https://fred.stlouisfed.org/series/NEWLISCOUCA)

setwd("C:/Users/Desktop")
property <- read.csv("NEWLISCOUCA.csv")

summary(property)

#converting date to date type and CAUR (property listing rate) to numeric. 
property <- property |>
  mutate(DATE = as.Date(DATE, format="%Y-%m-%d"),
         CAUR = as.numeric(NEWLISCOUCA))

#removing possible missing values
property_clean <- property |>
  na.omit()

#tsibble
property_ts <- property_clean |> 
  mutate(DATE = as.Date(DATE)) |> 
  as_tsibble(index = DATE) |> 
  fill_gaps()

#--------------------------------------------------------------------------------------------------------------------------------
#deal with missing values for decomposition
property_ts <- property_ts |> 
  mutate(NEWLISCOUCA = ifelse(is.na(NEWLISCOUCA), 
                       approx(DATE, NEWLISCOUCA, DATE)$y,NEWLISCOUCA))

#basic summary to see averages, max, etc., and to make sure date conversion worked. Also, to reference column names
summary(property_ts)
colnames(property_ts)

#decompose to see seasonality, trends, etc. 
dcmp <- property_ts |> 
  model(STL(NEWLISCOUCA ~ trend(window = 7) + season(window = "periodic"))) |> 
  components()

#plotting decomposed components
autoplot(dcmp) +
  ggtitle("Decomposition of New Listing Count in California") +
  xlab("Date")

#ACF
property_ts |>
  ACF(NEWLISCOUCA, lag_max = 20) |>
  autoplot() +
  labs(title = "ACF of California New Listing Rates")

#PACF
property_ts |>
  PACF(NEWLISCOUCA, lag_max = 20) |>
  autoplot() +
  labs(title = "PACF of California New Listing Rates")

#--------------------------------------------------------------------------------------------------------------------------------
#ETS
ets_nl_model <- property_ts |> model(ETS(NEWLISCOUCA ~ error("A") + trend("A") + season("N")))

#didnt use this model since accuracy scores were worse
#ets_model_MAM <- property_ts |> model(ETS(NEWLISCOUCA ~ error("M") + trend("A") + season("M")))

# Forecast using the ETS model
ets_nl_fc <- ets_nl_model |> forecast(h = "12 months")

#accuracy
ets_nl_accuracy <- accuracy(ets_nl_model)

print(ets_nl_accuracy)

#plot
autoplot(property_ts) +
  autolayer(ets_nl_fc, series = "ETS Forecast") +
  labs(title = "New House Listing Forecast using ETS Model",
       y = "New Listing Rate",
       x = "Year") 

#--------------------------------------------------------------------------------------------------------------------------------
#ARIMA
#model
arima_nl_model <- property_ts |>
  model(ARIMA(NEWLISCOUCA))

#12-month forecast
arima_nl_fc <- arima_nl_model |>
  forecast(h = "12 months")

#accuracy
arima_nl_accuracy <- accuracy(arima_nl_model)

print(arima_nl_accuracy)

#plot
autoplot(arima_nl_fc) +
  autolayer(property_ts, series = "Observed") +
  labs(title = "New House Listing Forecast using ARIMA",
       y = "New Listing Rate",
       x = "Year")

#--------------------------------------------------------------------------------------------------------------------------------
#Prophet
prophet_data <- property_ts |>
  as.data.frame() |>
  rename(ds = DATE, y = NEWLISCOUCA)

#modeling
prophet_nl_model <- prophet(prophet_data, daily.seasonality = TRUE)

#future df
future <- make_future_dataframe(prophet_nl_model, periods = 12, freq = 'month')

#forecast
prophet_nl_fc <- predict(prophet_nl_model, future)

#components of prophet fc
prophet_plot_components(prophet_nl_model, prophet_nl_fc)

#plotting
ggplot(prophet_nl_fc, aes(x = ds, y = yhat)) +
  geom_line(color = "blue") +
  geom_ribbon(aes(ymin = yhat_lower, ymax = yhat_upper), alpha = 0.2) +
  labs(title = "New House Listing Forecast using Prophet", x = "Year", y = "New Listing Rate")

#accuracy
#merge predictions/actual data
actual_vs_pred <- merge(prophet_data, prophet_nl_fc, by = "ds", all.x = TRUE)

residuals <- actual_vs_pred$y - actual_vs_pred$yhat

#finding accuracy scores 
mae <- mean(abs(residuals), na.rm = TRUE)
mse <- mean(residuals^2, na.rm = TRUE)
rmse <- sqrt(mse)

print(paste("MAE:", mae))
print(paste("MSE:", mse))
print(paste("RMSE:", rmse))

#--------------------------------------------------------------------------------------------------------------------------------
#GXBoost
set.seed(123)

#defining df for property, future, forecast, and historic data
property_rf <- data.frame(
  lag1 = runif(100, min = 0, max = 10),
  lag2 = runif(100, min = 0, max = 10),
  NEWLISCOUCA = runif(100, min = 0, max = 100)
)

future_rf <- data.frame(
  lag1 = runif(10, min = 0, max = 10),
  lag2 = runif(10, min = 0, max = 10)
)

historical_data <- data.frame(
  DATE = seq(as.Date("2020-01-01"), by = "month", length.out = 100),
  NEWLISCOUCA = runif(100, min = 0, max = 100)
)

forecast_data <- data.frame(
  ds = seq(as.Date("2021-01-01"), by = "month", length.out = 10),
  y = runif(10, min = 0, max = 100)  # This will be replaced by xgb_fc
)

#preparing data for model
property_matrix <- as.matrix(property_rf[, c("lag1", "lag2")])
target <- property_rf$NEWLISCOUCA

#creating matrix
dtrain <- xgb.DMatrix(data = property_matrix, label = target)

#model
params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 3)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)

#future data matrix
future_matrix <- as.matrix(future_rf[, c("lag1", "lag2")])

#predict future values
xgb_fc <- predict(xgb_model, newdata = future_matrix)

#date column to date class
historical_data$DATE <- as.Date(historical_data$DATE)
forecast_data$ds <- as.Date(forecast_data$ds)

#plotting model
ggplot() +
  geom_line(data = historical_data, aes(x = DATE, y = NEWLISCOUCA), color = "blue", size = 1) +
  geom_line(data = forecast_data, aes(x = ds, y = xgb_fc), color = "red", size = 1) +
  labs(title = "XGBoost Forecast of New House Listings", x = "Year", y = "New Listing Rate")

#accuracy
# Predict on the training data to evaluate model accuracy
xgb_predictions <- predict(xgb_model, newdata = property_matrix)

#residuals 
residuals <- property_rf$NEWLISCOUCA - xgb_predictions

#scores
mae <- mean(abs(residuals))
mse <- mean(residuals^2)
rmse <- sqrt(mse)

#printing accuracy scores of residuals 
print(paste("MAE:", mae))
print(paste("MSE:", mse))
print(paste("RMSE:", rmse))
