# Read and prepare data
library(tidyverse)
library(caret)
library(ggplot2)
library(randomForest)
library(xgboost)
library(e1071)

# Set script current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read CSV file 
kepler_raw <- read.csv("cumulative_2024.11.28_06.01.23.csv", 
                       skip = 86,  
                       header = TRUE,
                       stringsAsFactors = FALSE)

names(kepler_raw)

library(tidyverse)
library(caret)
library(ggplot2)

# Prepare dataset with habitability scoring
kepler_cleaned <- kepler_raw %>%
  select(
    koi_disposition,
    koi_teq,         # Temperature
    koi_insol,       # Insolation
    koi_prad,        # Planet radius
    koi_sma,         # Semi-major axis
    koi_period,      # Orbital period
    koi_steff,       # Star temperature
    koi_smet,        # Star metallicity
    koi_srad,        # Star radius
    koi_smass        # Star mass
  ) %>%
  filter(koi_disposition %in% c("CONFIRMED", "CANDIDATE")) %>%
  na.omit() %>%
  mutate(
    # Create component scores (0-1 scale)
    temp_score = exp(-(abs(koi_teq - 288) / 100)^2),      # Earth temp = 288K
    radius_score = exp(-(abs(koi_prad - 1) / 0.5)^2),     # Earth radius = 1
    insol_score = exp(-(abs(log10(koi_insol)) / 1)^2),    # Earth insolation = 1
    orbit_score = exp(-(abs(koi_sma - 1) / 0.5)^2),       # Earth orbit = 1 AU
    
    # Calculate final habitability score
    habitability_score = (temp_score + radius_score + insol_score + orbit_score) / 4
  )

# Visualize the distribution of habitability scores
ggplot(kepler_cleaned, aes(x = habitability_score)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribution of Habitability Scores",
       x = "Habitability Score",
       y = "Count")

# Prepare features and target for modeling
X <- kepler_cleaned %>%
  select(koi_teq, koi_insol, koi_prad, koi_sma, koi_period, 
         koi_steff, koi_smet, koi_srad, koi_smass) %>%
  scale() %>%
  as.data.frame()

y <- kepler_cleaned$habitability_score

# Split data
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Train multiple models
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

# Random Forest
rf_model <- train(
  x = X_train,
  y = y_train,
  method = "rf",
  trControl = train_control,
  metric = "RMSE"
)

# Gradient Boosting
gbm_model <- train(
  x = X_train,
  y = y_train,
  method = "gbm",
  trControl = train_control,
  metric = "RMSE",
  verbose = FALSE
)

# Linear Model with Elastic Net
glmnet_model <- train(
  x = X_train,
  y = y_train,
  method = "glmnet",
  trControl = train_control,
  metric = "RMSE"
)

# Compare model performances
models <- list(
  RandomForest = rf_model,
  GradientBoosting = gbm_model,
  ElasticNet = glmnet_model
)

# Function to evaluate models
evaluate_model <- function(model, X_test, y_test) {
  predictions <- predict(model, X_test)
  rmse <- sqrt(mean((predictions - y_test)^2))
  r2 <- cor(predictions, y_test)^2
  list(RMSE = rmse, R2 = r2)
}

# Evaluate all models
results <- lapply(models, evaluate_model, X_test, y_test)
print(results)

# Visualize feature importance (for Random Forest)
importance_df <- varImp(rf_model)$importance
importance_df$Feature <- rownames(importance_df)

ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance for Habitability Score Prediction",
       x = "Features",
       y = "Importance")

# Plot predicted vs actual values
predictions <- predict(rf_model, X_test)
plot_df <- data.frame(
  Actual = y_test,
  Predicted = predictions
)

ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(title = "Predicted vs Actual Habitability Scores",
       x = "Actual Score",
       y = "Predicted Score")