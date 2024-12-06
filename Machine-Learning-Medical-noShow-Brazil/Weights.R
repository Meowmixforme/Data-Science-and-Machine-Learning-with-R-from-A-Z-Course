# Import required libraries
library(dplyr)
library(tidyr)
library(corrgram)
library(ggplot2)
library(caret)
library(MASS)
library(pROC)
library(lubridate)
library(skimr)
library(corrplot)
library(smotefamily)

# Set script current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read dataset
noShow <- read.csv("KaggleV2-May-2016.csv")

# Remove ID columns
noShow <- noShow %>% 
  dplyr::select(-PatientId, -AppointmentID)

# Basic dataset exploration
dim(noShow)
str(noShow)
summary(noShow)
skim(noShow)

# Correlation Analysis for numeric variables
numeric_vars <- noShow[sapply(noShow, is.numeric)]
corrplot(cor(numeric_vars), 
         type="lower", 
         tl.srt = 90,
         main = "Correlation Matrix of Numeric Variables")

# Rename columns for UK context
noShow <- noShow %>%
  rename(
    Hypertension = Hipertension,
    Disabled = Handcap,
    WelfareBenefits = Scholarship
  )

# Create a new Categorical variable for age groups based on the NHS 
noShow <- noShow %>%
  mutate(
    AgeGroup = factor(
      cut(Age, 
          breaks = c(-Inf, 15, 24, 44, 64, 84, Inf),
          labels = c("Children", "Young People", "Working Age", 
                     "Middle Age", "Senior", "Elderly")
      ),
      ordered = TRUE
    )
  )

# Create waiting days variables
noShow <- noShow %>%
  mutate(
    WaitingDays = as.numeric(difftime(as.Date(AppointmentDay), 
                                      as.Date(ScheduledDay), units = "days")),
    WaitingDays_cat = factor(
      cut(WaitingDays,
          breaks = c(-Inf, 0, 1, 7, 30, Inf),
          labels = c("Same day", "Next day", "Within week",
                     "Within month", "Over month")
      ),
      ordered = TRUE
    )
  )

# Add weekday feature
noShow <- noShow %>%
  mutate(
    WeekDay = factor(
      wday(as.Date(AppointmentDay)),
      levels = 1:7,
      labels = c("Sunday", "Monday", "Tuesday", "Wednesday", 
                 "Thursday", "Friday", "Saturday")
    )
  )

# Remove weekends
noShow <- noShow %>%
  filter(WeekDay %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"))

# Calculate and add neighborhood risk
neighborhood_rates <- noShow %>%
  group_by(Neighbourhood) %>%
  summarise(
    noshow_rate = mean(No.show == "Yes"),
    .groups = 'drop'
  )

noShow <- noShow %>%
  left_join(neighborhood_rates, by = "Neighbourhood") %>%
  mutate(
    NeighbourhoodRisk = factor(
      case_when(
        noshow_rate <= quantile(neighborhood_rates$noshow_rate, 0.33) ~ "Low Risk",
        noshow_rate <= quantile(neighborhood_rates$noshow_rate, 0.66) ~ "Medium Risk",
        TRUE ~ "High Risk"
      ),
      levels = c("Low Risk", "Medium Risk", "High Risk"),
      ordered = TRUE
    )
  ) %>%
  dplyr::select(-noshow_rate, -Neighbourhood)

# Convert character columns to factors
noShow$Gender <- as.factor(noShow$Gender)
noShow$ScheduledDay <- as.factor(noShow$ScheduledDay)
noShow$AppointmentDay <- as.factor(noShow$AppointmentDay)

# Convert factor columns to numeric (excluding the target column)
noShow_numeric <- noShow[, -which(names(noShow) == "No.show")]  # Exclude the target column
noShow_numeric[] <- lapply(noShow_numeric, function(x) {
  if (is.factor(x)) {
    as.numeric(as.factor(x))  # Convert factors to numeric
  } else {
    x  # Keep numeric columns as is
  }
})

# Convert the target variable to factor
noShow$No.show <- as.factor(noShow$No.show)

# Apply BLSMOTE with adjusted parameters
genData_BLSMOTE = BLSMOTE(noShow_numeric, noShow$No.show, 
                          K = 4,     # Adjusted
                          C = 8,     # Adjusted
                          method = "type1")

# Combine original and synthetic data
combined_data <- rbind(genData_BLSMOTE$data, genData_BLSMOTE$syn_data)

# Convert the target variable to factor
combined_data$class <- as.factor(combined_data$class)

# Select relevant features
balanced_data <- combined_data %>%
  dplyr::select(
    class,              # Target variable
    Gender,
    WelfareBenefits,
    Disabled,
    SMS_received,
    WaitingDays_cat,
    WeekDay,
    NeighbourhoodRisk,
    AgeGroup
  )

# Custom metric for model training
custom_summary <- function(data, lev = NULL, model = NULL) {
  cm <- confusionMatrix(data$pred, data$obs)
  c(twoClassSummary(data, lev = lev, model = model),
    balanced_metric = (cm$byClass["Sensitivity"] + cm$byClass["Specificity"]) / 2)
}

# Create training control with balanced optimization
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     summaryFunction = custom_summary,
                     selectionFunction = "best")

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(balanced_data$class, p = 0.8, list = FALSE)
train_data <- balanced_data[train_index, ]
test_data <- balanced_data[-train_index, ]

# Add balanced class weights
class_weights <- ifelse(train_data$class == "Yes", 
                        1,    # Weight for majority class
                        2)    # Reduced weight for minority class

# Train GLM Model
glm_model <- train(class ~ ., 
                   data = train_data,
                   method = "glm",
                   family = "binomial",
                   weights = class_weights,
                   trControl = ctrl,
                   metric = "balanced_metric")

# Train GLMNET Model
tune_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = 10^seq(-6, -1, length.out = 20)
)

glmnet_model <- train(class ~ .,
                      data = train_data,
                      method = "glmnet",
                      family = "binomial",
                      weights = class_weights,
                      tuneGrid = tune_grid,
                      trControl = ctrl,
                      metric = "balanced_metric")

# Enhanced threshold optimization function
find_optimal_threshold <- function(model, test_data) {
  pred_probs <- predict(model, newdata = test_data, type = "prob")
  
  thresholds <- seq(0.3, 0.7, by = 0.01)  # More balanced range
  results <- data.frame(threshold = thresholds,
                        sensitivity = NA,
                        specificity = NA,
                        accuracy = NA,
                        balanced_accuracy = NA,
                        f1_score = NA)
  
  for(i in seq_along(thresholds)) {
    predictions <- ifelse(pred_probs[,"Yes"] > thresholds[i], "Yes", "No")
    predictions <- factor(predictions, levels = levels(test_data$class))
    cm <- confusionMatrix(predictions, test_data$class)
    
    precision <- cm$byClass["Pos Pred Value"]
    recall <- cm$byClass["Sensitivity"]
    f1 <- 2 * (precision * recall) / (precision + recall)
    
    results$sensitivity[i] <- cm$byClass["Sensitivity"]
    results$specificity[i] <- cm$byClass["Specificity"]
    results$accuracy[i] <- cm$overall["Accuracy"]
    results$balanced_accuracy[i] <- cm$byClass["Balanced Accuracy"]
    results$f1_score[i] <- f1
  }
  
  return(results)
}

# Find optimal thresholds
threshold_results_glm <- find_optimal_threshold(glm_model, test_data)
threshold_results_glmnet <- find_optimal_threshold(glmnet_model, test_data)

# Plot threshold results with all metrics
ggplot(threshold_results_glm, aes(x = threshold)) +
  geom_line(aes(y = sensitivity, color = "Sensitivity")) +
  geom_line(aes(y = specificity, color = "Specificity")) +
  geom_line(aes(y = accuracy, color = "Accuracy")) +
  geom_line(aes(y = balanced_accuracy, color = "Balanced Accuracy")) +
  geom_line(aes(y = f1_score, color = "F1 Score")) +
  theme_minimal() +
  labs(title = "GLM Performance Metrics vs Threshold",
       y = "Metric Value")

# Select optimal thresholds balancing all metrics
optimal_threshold_glm <- threshold_results_glm$threshold[which.max(
  (threshold_results_glm$sensitivity + 
     threshold_results_glm$specificity + 
     threshold_results_glm$accuracy) / 3)]

optimal_threshold_glmnet <- threshold_results_glmnet$threshold[which.max(
  (threshold_results_glmnet$sensitivity + 
     threshold_results_glmnet$specificity + 
     threshold_results_glmnet$accuracy) / 3)]

# Make final predictions using optimal thresholds
predictions_glm <- predict(glm_model, newdata = test_data, type = "prob")
predictions_glmnet <- predict(glmnet_model, newdata = test_data, type = "prob")

final_pred_glm <- ifelse(predictions_glm[,"Yes"] > optimal_threshold_glm, "Yes", "No")
final_pred_glm <- factor(final_pred_glm, levels = levels(test_data$class))

final_pred_glmnet <- ifelse(predictions_glmnet[,"Yes"] > optimal_threshold_glmnet, "Yes", "No")
final_pred_glmnet <- factor(final_pred_glmnet, levels = levels(test_data$class))

# Evaluate final results
print("GLM Results with Optimal Threshold:")
print(confusionMatrix(final_pred_glm, test_data$class))

print("GLMNET Results with Optimal Threshold:")
print(confusionMatrix(final_pred_glmnet, test_data$class))

# Plot ROC curves
roc_glm <- roc(test_data$class, predictions_glm[,"Yes"])
roc_glmnet <- roc(test_data$class, predictions_glmnet[,"Yes"])

plot(roc_glm, col = "blue", main = "ROC Curves Comparison")
lines(roc_glmnet, col = "red")
legend("bottomright", 
       legend = c("GLM", "GLMNET"),
       col = c("blue", "red"),
       lwd = 2)

# Print comprehensive metrics
print(paste("GLM AUC:", round(auc(roc_glm), 4)))
print(paste("GLMNET AUC:", round(auc(roc_glmnet), 4)))