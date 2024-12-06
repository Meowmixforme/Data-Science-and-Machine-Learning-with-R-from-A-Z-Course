# Load required libraries
library(tidyverse)
library(caret)
library(dplyr)
library(MASS)
library(randomForest)
library(xgboost)
library(e1071)
library(keras)
library(lightgbm)
library(corrplot)
library(RColorBrewer)

# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read the raw data
kepler_data <- read.csv("cumulative.csv")

# 1. First create numeric class for stepwise
stepwise_data <- kepler_data %>%
  mutate(class_numeric = case_when(
    koi_disposition == "CONFIRMED" ~ 2,
    koi_disposition == "CANDIDATE" ~ 1,
    koi_disposition == "FALSE POSITIVE" ~ 0
  ))

# 2. Select initial features for stepwise analysis
stepwise_data <- stepwise_data %>%
  dplyr::select(
    # Main transit features
    koi_period, koi_impact, koi_duration, koi_depth,
    
    # Planet characteristics
    koi_prad, koi_teq, koi_insol,
    
    # Star characteristics
    koi_steff, koi_slogg, koi_srad,
    
    # Signal quality
    koi_model_snr, koi_score,
    
    # Outcome
    class_numeric
  )

# 3. Run stepwise selection
# Convert target to factor if it isn't already
kepler_data$koi_disposition <- as.factor(kepler_data$koi_disposition)

# Create initial model with all predictors
initial_model <- glm(koi_disposition ~ koi_period + koi_impact + koi_duration + 
                       koi_depth + koi_prad + koi_teq + koi_insol + koi_steff + 
                       koi_slogg + koi_srad + koi_model_snr + koi_score,
                     family = binomial(link = 'logit'),
                     data = kepler_data)

# Apply stepwise selection
stepwise_model <- stepAIC(initial_model, direction = "both")

# Print summary of final model with stars
print(summary(stepwise_model))




selected_features <- c(
  # Highly significant (p < 0.001) ***
  "koi_period",     # Very strong negative effect
  "koi_score",      # Very strong negative effect
  "koi_steff",      # Negative effect
  "koi_duration",   # Positive effect
  "koi_model_snr",  # Positive effect
  
  # Moderately significant (p < 0.01) **
  "koi_impact",     # Negative effect
  "koi_slogg",      # Positive effect
  
  # Add target variable
  "koi_disposition"
)

# Update final dataset with only significant features
final_data <- kepler_data %>%
  dplyr::select(all_of(selected_features)) %>%
  na.omit() %>%
  mutate(koi_disposition = factor(koi_disposition))




# 5. NOW clean and prepare final dataset with selected features
final_data <- kepler_data %>%
  dplyr::select(all_of(selected_features)) %>%
  na.omit() %>%
  mutate(koi_disposition = factor(koi_disposition))

print("Selected features:")
print(selected_features)

print("Final dataset dimensions:")
print(dim(final_data))

# Create correlation matrix

# Select numeric features for correlation
cor_features <- stepwise_data %>%
  dplyr::select(
    koi_impact, koi_depth, koi_prad, koi_teq, 
    koi_steff, koi_slogg, koi_model_snr, koi_score,
    class_numeric
  )

# Create correlation matrix using final_data
cor_features <- final_data %>%
  mutate(class_numeric = as.numeric(koi_disposition) - 1) %>%  # Convert factor to numeric
  dplyr::select(-koi_disposition)  # Remove original categorical variable

# Calculate correlation matrix
cor_matrix <- cor(cor_features, use = "complete.obs")

# Create correlation plot
corrplot(cor_matrix, 
         method = "color",
         type = "upper",
         order = "hclust",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45,
         diag = FALSE,
         col = brewer.pal(n = 8, name = "RdYlBu"))




clean_data <- final_data %>%  # Use final_data instead of kepler_data
  mutate(koi_disposition = case_when(
    koi_disposition == "CANDIDATE" ~ "candidate",
    koi_disposition == "CONFIRMED" ~ "confirmed",
    koi_disposition == "FALSE POSITIVE" ~ "false_positive"
  ) %>% factor())

# Verify the new levels
print("New factor levels:")
print(levels(clean_data$koi_disposition))

# Create balanced dataset
set.seed(123)
balanced_data <- clean_data %>%
  group_by(koi_disposition) %>%
  sample_n(size = min(table(clean_data$koi_disposition))) %>%
  ungroup()

# Split data
set.seed(123)
train_index <- createDataPartition(balanced_data$koi_disposition, p = 0.7, list = FALSE)
train_data <- balanced_data[train_index, ]
test_data <- balanced_data[-train_index, ]

# Rest of your model training code remains the same...
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  savePredictions = "final"
)

# Train models with stepwise-selected features
rf_model <- train(
  koi_disposition ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  metric = "Kappa",
  tuneLength = 5
)

xgb_model <- train(
  koi_disposition ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  metric = "Kappa",
  tuneLength = 5
)

svm_model <- train(
  koi_disposition ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = ctrl,
  metric = "Kappa",
  tuneLength = 5
)


# Create models list
models <- list(
  RandomForest = rf_model,
  XGBoost = xgb_model,
  SVM = svm_model
)

# Now compare results
results <- resamples(models)
summary(results)

# Create confusion matrices
model_cms <- lapply(models, function(model) {
  predictions <- predict(model, test_data)
  confusionMatrix(predictions, test_data$koi_disposition)
})

# Print results
print("\nModel Comparison:")
for(name in names(model_cms)) {
  print(paste("\n", name, "Results:"))
  print(model_cms[[name]])
}