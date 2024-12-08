install.packages("MLmetrics",)
install.packages("lubridate")
install.packages("ggplot2")
install.packages("readr")
install.packages("dplyr")
install.packages("caret")
install.packages("ggcorrplot")
install.packages("forecast")
install.packages("reshape2")
install.packages("plotrix")
install.packages("glmnet")
install.packages("PerformanceAnalytics")
install.packages("generalhoslem")
install.packages("DescTools")
install.packages("leaps")
library(MLmetrics)
library(lubridate)
library(ggplot2)
library(readr)
library(dplyr)
library(caret)
library(ggcorrplot)
library(forecast)
library(reshape2)
library(DescTools)
library(e1071)
library(leaps)
library(PerformanceAnalytics)
library(generalhoslem)
library(caTools)
library(scales)
library(pROC)
library(plotrix)
library(plyr)
library(glmnet)

# Import the yearly temperature data from nity18442004.csv file
year_temp <- read.csv("nity18442004.csv")

# Import the monthly temperature data from nitm18442004.csv file
month_temp <- read.csv("nitm18442004.csv")
#Viewing the Dataset for month Temp
head(month_temp)
# Convert data frame to time series object
month_ts <- ts(month_temp$x, start = c(1844, 1), frequency = 12)
#Viewing the First five Time series Object
head(month_ts)
#Viewing the Summary for the Time series Object
summary(month_ts)
# Plot time series with ggplot2
ggplot(data = data.frame(time = time(month_ts), value = as.vector(month_ts)), aes(x = time, y = value)) + 
  geom_line(color = "blue") + 
  xlab("Year") + 
  ylab("Temperature") + 
  ggtitle("Monthly Temperature Time Series") + 
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) 
# Get the frequency of the time series
freq <- frequency(month_ts)
freq
# Set color palette
mub_colors <- c("blue", "red", "green", "orange")

# Decompose the time series using a multiplicative model
fit.month_ts <- decompose(month_ts, type = "multiplicative")

# Create a plot of the decomposition with colors
plot(fit.month_ts, col = mub_colors)
#Creating a Plot for Average Temp
autoplot(month_ts) + 
  ggtitle("Time Plot: Average Temperatures") +  # add title
  xlab("Year") +  # add x-axis label
  ylab("Temperature") +  # add y-axis label
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5)) + # rotate x-axis labels
  scale_color_manual(values=c("#00BFC4", "#F8766D"))  # set custom colors
# Splitting into test and train
training_month <- window(month_ts, end = c(2003, 12))
testing_month <- window(month_ts, start = c(2004, 1))
testing_month
# Simple Exponential Smoothing model
fcast.ses <- ses(training_month, h = 12)

# Increase plot size
options(repr.plot.width = 10, repr.plot.height = 6)

# Plot forecast with blue line color
plot(fcast.ses, main = "SES Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(2004, 2005), col = "blue")

# Add actual values with black line color
lines(testing_month, col = "black")

# Add legend
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("red", "black"))

# Accuracy evaluation
acc_ses <- accuracy(fcast.ses, testing_month)
summary(acc_ses)
cat("Simple Exponential Smoothing model (MAE): ", round(mean(acc_ses[,"MAE"]), 2))
# Fit a Holt-Winters model
fit <- hw(training_month)

# Generate forecasts
fcast_hw <- forecast(fit, h = 12)
# Plot forecasts
plot(fcast_hw, main = "Holt-Winters Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(2004, 2005))
lines(testing_month, col = "black")
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("blue", "black"))
#Accuracy evaluation
acc_hw <- data.frame(accuracy(fcast_hw, testing_month))
summary(acc_hw)
cat("Holt-Winters accuracy (MAE): ", round(mean(acc_hw$MAE), 2))
# Fitting an ARIMA model
fit_arima <- auto.arima(training_month)

# Generate forecasts
fcast_arima <- forecast(fit_arima, h = 12)

# Plot forecasts
plot(fcast_arima, main = "ARIMA Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(2004, 2005))
lines(testing_month, col = "black")
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("blue", "black"))

# Accuracy evaluation
acc_arima <- accuracy(fcast_arima, testing_month)
summary(acc_arima)
cat("ARIMA accuracy (MAE): ", round(mean(acc_arima[,"MAE"]), 2))

# Simple Time Series Mean model
fcast.mean <- meanf(training_month, h = 12)

# Plot forecast
plot(fcast.mean, main = "Mean Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(2004, 2005))
lines(testing_month, col = "black")
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("blue", "black"))  

# Accuracy evaluation
acc_mm <- accuracy(fcast.mean, testing_month)
summary(acc_mm)
cat("Simple time series (MAE): ", round(mean(acc_mm[,"MAE"]), 2))
#seasonal naive model
fcast.seasonalnaive<-snaive(training_month, h=12)
# Plot forecast
plot(fcast.seasonalnaive, main = "Seasonal Naive Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(2004, 2005))
lines(testing_month, col = "black")
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("blue", "black"))
# Accuracy evaluation
acc_naive <- accuracy(fcast.seasonalnaive, testing_month)
summary(acc_naive)
cat("seasonal naive model (MAE): ", round(mean(acc_naive[,"MAE"]), 2))
# Year Data
year_ts <- ts(yearly_temp$x, start = c(1844, 1), frequency = 1)
plot(year_ts)
# Splitting into test and train
training_year <- window(year_ts, end = 2003)
testing_year <- window(year_ts, start = 2004)
# Simple Exponential Smoothing model
fcast.ses <- ses(training_year, h = 12)
# Plot forecast
plot(fcast.ses, main = "SES Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(1844, 2005))
lines(testing_year, col = "black")
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("orange", "yellow"))  
# Accuracy evaluation
acc_ses <- accuracy(fcast.ses, testing_year)
summary(acc_ses)
cat("Simple Exponential Smoothing model (MAE): ", round(mean(acc_ses[,"MAE"]), 2))
# Fitting an ARIMA model
fit_arima <- auto.arima(training_year)

# Generate forecasts
fcast_arima <- forecast(fit_arima, h = 12)

# Plot forecasts
plot(fcast_arima, main = "ARIMA Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(1844, 2005))
lines(testing_year, col = "black")
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("orange", "yellow"))

# Accuracy evaluation
acc_arima <- accuracy(fcast_arima, testing_year)
summary(acc_arima)
cat("ARIMA accuracy (MAE): ", round(mean(acc_arima[,"MAE"]), 2))
# Simple Time Series Mean model
fcast.mean <- meanf(training_year, h = 12)

# Plot forecast
plot(fcast.mean, main = "Mean Model - Actual vs Forecast", xlab = "Year", ylab = "Value", xlim = c(1844, 2005))
lines(testing_year, col = "black")
legend("topleft", legend = c("Forecast", "Actual"), lty = 1, col = c("orange", "yellow"))  

# Accuracy evaluation
acc_mm <- accuracy(fcast.mean, testing_year)
summary(acc_mm)
cat("Simple time series (MAE): ", round(mean(acc_mm[,"MAE"]), 2))
