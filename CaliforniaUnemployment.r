#Part 2: California Unemployment Rate (https://fred.stlouisfed.org/series/CAUR)

setwd("C:/Users/Desktop")
cal_unemploy <- read.csv("CalUnemploymentRate.csv")

#converting date to date type and CAUR (unemployment rate) to numeric. 
cal_unemploy <- cal_unemploy |>
  mutate(DATE = as.Date(DATE, format="%Y-%m-%d"),
         CAUR = as.numeric(CAUR))

#removing possible missing values
cal_clean <- cal_unemploy |>
  na.omit()

#tsibble
cal_ts <- cal_clean |> 
  mutate(DATE = as.Date(DATE)) |> 
  as_tsibble(index = DATE) |> 
  fill_gaps()

#--------
#deal with missing values for decomposition
cal_ts <- cal_ts |> 
  mutate(CAUR = ifelse(is.na(CAUR), 
         approx(DATE, CAUR, DATE)$y,CAUR))

#basic summary to see averages, max, etc. and to make sure date conversion worked. Also to reference column names
summary(cal_ts)
colnames(cal_ts)

#decompose to see seasonality, trends, etc. 
dcmp <- cal_ts |> 
  model(STL(CAUR ~ trend(window = 7) + season(window = "periodic"))) |> 
  components()

#plotting decomposed components
autoplot(dcmp) +
  ggtitle("Decomposition of California Unemployment Rate") +
  xlab("Date")

#ACF
cal_ts |>
  ACF(CAUR, lag_max = 20) |>
  autoplot() +
  labs(title = "ACF of California Unemployment Rate")

#PACF
cal_ts |>
  PACF(CAUR, lag_max = 20) |>
  autoplot() +
  labs(title = "PACF of California Unemployment Rate")
#----------------------------------------------------------------------------------------------------------------------------------------
#ETS
ets_cal_model <- cal_ts |>
  model(ETS(CAUR))

# Forecast using the ETS model
ets_cal_fc <- ets_cal_model |> forecast(h = "12 months")

#accuracy
ets_cal_acc <- accuracy(ets_cal_model)

print(ets_cal_acc)

#plot
autoplot(cal_ts) +
  autolayer(ets_cal_fc, series = "ETS Forecast") +
  labs(title = "California Unemployment Rate Forecast using ETS Model",
       y = "Unemployment Rate",
       x = "Year") 
#----------------------------------------------------------------------------------------------------------------------------------------
#ARIMA
arima_cal_model <- cal_ts |>
  model(ARIMA(CAUR))

arima_cal_fc <- arima_cal_model |>
  forecast(h = "12 months")

#accuracy
arima_cal_acc <- accuracy(arima_cal_fc, cal_ts)

print(arima_cal_acc)

#plot
autoplot(arima_cal_fc) +
  autolayer(cal_ts, series = "Observed") +
  labs(title = "California Unemployment Rate Forecast using ARIMA Model",
       y = "Unemployment Rate",
       x = "Year")
#----------------------------------------------------------------------------------------------------------------------------------------
#Prophet
prophet_data <- cal_ts |>
  as.data.frame() |>
  rename(ds = DATE, y = CAUR)

prophet_cal_model <- prophet(prophet_data, daily.seasonality = TRUE)

#future df
future <- make_future_dataframe(prophet_cal_model, periods = 12, freq = 'month')

#forecast
prophet_cal_fc <- predict(prophet_cal_model, future)

#components of prophet fc
prophet_plot_components(prophet_cal_model, prophet_cal_fc)

#combining current and forecast data and creating ts
prophet_cal_fc <- prophet_cal_fc[, !duplicated(names(prophet_cal_fc))]
prophet_cal_fc <- select(prophet_cal_fc, -ds)
combined_fc <- cbind(future, prophet_cal_fc)
prophet_fc_ts <- as_tsibble(combined_fc, index = ds)

#removing duplicate columns
combined_fc$ds <- as.Date(combined_fc$ds)
prophet_cal_fc <- prophet_cal_fc[, !duplicated(names(prophet_cal_fc))]
prophet_fc_ts <- as_tsibble(combined_fc, index = ds, regular = FALSE)

ggplot(prophet_fc_ts, aes(x = ds, y = yhat)) +
  geom_line(color = "purple") +
  geom_ribbon(aes(ymin = yhat_lower, ymax = yhat_upper), alpha = 0.2) +
  labs(title = "California Unemployment Rate Forecast using Prophet", x = "Year", y = "Forecasted Rate")

#accuracy
#merge predictions/actual data
actual_vs_pred <- merge(prophet_data, combined_fc, by = "ds", all.x = TRUE)

residuals <- actual_vs_pred$y - actual_vs_pred$yhat

#finding accuracy scores 
mae <- mean(abs(residuals), na.rm = TRUE)
mse <- mean(residuals^2, na.rm = TRUE)
rmse <- sqrt(mse)
print(paste("MAE:", mae))
print(paste("MSE:", mse))
print(paste("RMSE:", rmse))
#----------------------------------------------------------------------------------------------------------------------------------------
#Random Forest

#removing rows with missing values
cal_ts_clean <- na.omit(cal_ts)

#model
rf_model <- randomForest(CAUR ~ DATE, data = as.data.frame(cal_ts_clean), ntree = 500)

#12-month prediction using cleaned data 
future_dates <- seq(from = max(cal_ts_clean$DATE) + 1, 
                    by = "month", length.out = 12)
future_data <- data.frame(DATE = future_dates)
forecast_values <- predict(rf_model, newdata = future_data)

#combing future dates with the forecasted values
forecast_df <- data.frame(DATE = future_dates, Forecast = forecast_values)

#train and testing sets
set.seed(123)
train_size <- floor(0.8 * nrow(cal_ts_clean))
train_data <- cal_ts_clean[1:train_size, ]
test_data <- cal_ts_clean[(train_size + 1):nrow(cal_ts_clean), ]

#training model
rf_model <- randomForest(CAUR ~ DATE, data = as.data.frame(train_data), ntree = 500)

#testing 
predictions <- predict(rf_model, newdata = as.data.frame(test_data))

#accuracy
mae <- mean(abs(test_data$CAUR - predictions))
rmse <- sqrt(mean((test_data$CAUR - predictions)^2))
mape <- mean(abs((test_data$CAUR - predictions) / test_data$CAUR)) * 100

print(paste("Mean Absolute Error (MAE):", round(mae, 4)))
print(paste("Root Mean Squared Error (RMSE):", round(rmse, 4)))
print(paste("Mean Absolute Percentage Error (MAPE):", round(mape, 2), "%"))

#combining actual/predicted data
plot_data <- data.frame(DATE = test_data$DATE, Actual = test_data$CAUR, Predicted = predictions)

#plot
ggplot(plot_data, aes(x = DATE)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed") +
  labs(title = "California Unemployment Rate using Random Forests",
       y = "Unemployment Rate", x = "Year",
       color = "Legend")
