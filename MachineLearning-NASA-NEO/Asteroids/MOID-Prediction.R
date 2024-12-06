# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)

# Set script current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read and prepare data
asteroids <- read.csv("Asteroid_Updated.csv")

# Clean the data (excluding pha)
clean_asteroids <- asteroids %>%
  select(a, e, i, om, w, q, ad, per_y, H, n, per, ma, moid) %>%
  na.omit()

# Print dimensions of cleaned data
print("Dataset dimensions:")
print(dim(clean_asteroids))

# Create risk categories
asteroid_risk <- clean_asteroids %>%
  mutate(risk_category = case_when(
    moid <= 0.05 & H <= 22 & e > 0.6 ~ "VeryHighRisk",
    moid <= 0.05 & H <= 22 ~ "HighRisk",
    moid <= 0.1 & H <= 25 ~ "ModerateRisk",
    TRUE ~ "LowRisk"
  )) %>%
  mutate(risk_category = factor(risk_category, 
                                levels = c("LowRisk", 
                                           "ModerateRisk",
                                           "HighRisk",
                                           "VeryHighRisk"))) %>%
  select(-moid, -H) %>%  # Remove direct predictors
  select(a, e, i, om, w, q, ad, per_y, n, per, ma, risk_category)  # Final column selection

# Print initial class distribution
print("Initial class distribution:")
print(table(asteroid_risk$risk_category))

# Create balanced dataset
set.seed(123)
low_risk <- asteroid_risk %>% 
  filter(risk_category == "LowRisk") %>% 
  sample_n(10000)
moderate_risk <- asteroid_risk %>% 
  filter(risk_category == "ModerateRisk") %>% 
  sample_n(10000, replace = TRUE)
high_risk <- asteroid_risk %>% 
  filter(risk_category == "HighRisk") %>% 
  sample_n(10000, replace = TRUE)
very_high_risk <- asteroid_risk %>% 
  filter(risk_category == "VeryHighRisk") %>% 
  sample_n(10000, replace = TRUE)

# Combine and shuffle
balanced_dataset <- bind_rows(low_risk, moderate_risk, high_risk, very_high_risk) %>%
  slice_sample(n = nrow(.))

# Print balanced class distribution
print("\nBalanced class distribution:")
print(table(balanced_dataset$risk_category))

# Create numeric response for stepwise regression
stepwise_data <- balanced_dataset %>%
  mutate(risk_numeric = as.numeric(risk_category) - 1) %>%
  select(-risk_category)

# Fit full model
print("\nPerforming Stepwise Feature Selection...")
full_model <- lm(risk_numeric ~ ., data = stepwise_data)

# Perform stepwise selection
library(MASS)
step_model <- stepAIC(full_model, direction = "both")

# Print summary of stepwise model
print("\nStepwise Model Summary:")
print(summary(step_model))

best_features <- c("e", "n", "i", "a", "per_y", "q", "w", "om", "risk_category")

# Create dataset with only best features
simplified_dataset <- balanced_dataset %>%
  dplyr::select(all_of(best_features))

# Split data
set.seed(123)
train_index <- createDataPartition(simplified_dataset$risk_category, p = 0.7, list = FALSE)
train_data <- simplified_dataset[train_index, ]
test_data <- simplified_dataset[-train_index, ]

# Set up cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  savePredictions = "final"
)

# Train Random Forest
print("\nTraining Random Forest...")
rf_model <- train(
  risk_category ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  metric = "Kappa",
  tuneLength = 5
)

# Print RF training results
print("\nRandom Forest CV Results:")
print(rf_model$results)

# RF predictions and confusion matrix
rf_pred <- predict(rf_model, test_data)
rf_conf_matrix <- confusionMatrix(rf_pred, test_data$risk_category)

print("\nRandom Forest Confusion Matrix:")
print(rf_conf_matrix)

# RF feature importance
rf_importance <- varImp(rf_model)
print("\nRandom Forest Feature Importance:")
print(rf_importance)

# Train Logistic Regression
print("\nTraining Logistic Regression...")
log_model <- train(
  risk_category ~ .,
  data = train_data,
  method = "multinom",
  trControl = ctrl,
  metric = "Kappa"
)

# Logistic Regression predictions and confusion matrix
log_pred <- predict(log_model, test_data)
log_conf_matrix <- confusionMatrix(log_pred, test_data$risk_category)

print("\nLogistic Regression Confusion Matrix:")
print(log_conf_matrix)



# Model Comparison
model_comparison <- data.frame(
  Model = c("Random Forest", "Logistic Regression"),
  Accuracy = c(rf_conf_matrix$overall["Accuracy"], 
               log_conf_matrix$overall["Accuracy"]),
  Kappa = c(rf_conf_matrix$overall["Kappa"],
            log_conf_matrix$overall["Kappa"])
)

print("\nModel Comparison:")
print(model_comparison)
