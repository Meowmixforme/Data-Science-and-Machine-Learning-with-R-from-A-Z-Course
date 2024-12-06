# Load required libraries
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
library(e1071)  # Added for SVM
library(xgboost)

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

# Check the original class distribution
class_distribution <- table(noShow$No.show)
print(class_distribution)

# Convert character columns to factors
noShow$Gender <- as.factor(noShow$Gender)
noShow$ScheduledDay <- as.factor(noShow$ScheduledDay)
noShow$AppointmentDay <- as.factor(noShow$AppointmentDay)

# Convert factor columns to numeric (excluding the target column)
noShow_numeric <- noShow[, -which(names(noShow) == "No.show")]
noShow_numeric[] <- lapply(noShow_numeric, function(x) {
  if (is.factor(x)) {
    as.numeric(as.factor(x))
  } else {
    x
  }
})

# Convert the target variable to factor
noShow$No.show <- as.factor(noShow$No.show)

# Apply BLSMOTE for Borderline SMOTE
genData_BLSMOTE = BLSMOTE(noShow_numeric, noShow$No.show, K = 14, C = 45)

# Check class distributions
class_distribution_balanced_BLSMOTE <- table(genData_BLSMOTE$data$class)
print(class_distribution_balanced_BLSMOTE)
class_distribution_syn_BLSMOTE <- table(genData_BLSMOTE$syn_data$class)
print(class_distribution_syn_BLSMOTE)

# Combine original and synthetic data
combined_data <- rbind(genData_BLSMOTE$data, genData_BLSMOTE$syn_data)
combined_data$class <- as.factor(combined_data$class)

# Select relevant features
combined_data <- combined_data %>%
  dplyr::select(
    class,
    WaitingDays_cat,
    AgeGroup
  )

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(combined_data$class, p = 0.8, list = FALSE)
train_data <- combined_data[train_index, ]
test_data <- combined_data[-train_index, ]

# Train GLM model
glm_model <- train(class ~ ., data = train_data, method = "glm", family = "binomial")

# Train GLMNET model with hyperparameter tuning
tune_grid <- expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0, 1, by = 0.1))
glmnet_model <- train(
  class ~ ., 
  data = train_data, 
  method = "glmnet",
  family = "binomial",
  tuneGrid = tune_grid, 
  trControl = trainControl(method = "cv", number = 10)
)

# Train SVM model with tuning
svm_tune <- tune.svm(
  class ~ ., 
  data = train_data,
  kernel = "radial",
  cost = c(0.1, 1, 10),
  gamma = c(0.1, 1, 10),
  tunecontrol = tune.control(cross = 5)
)

# Get the best SVM model
svm_model <- svm(
  class ~ ., 
  data = train_data,
  kernel = "radial",
  cost = svm_tune$best.parameters$cost,
  gamma = svm_tune$best.parameters$gamma,
  probability = TRUE
)

# Make predictions
# GLM
predictions1 <- predict(glm_model, newdata = test_data, type = "prob")
predictions1 <- ifelse(predictions1[,"Yes"] > 0.4, "Yes", "No")
predictions1 <- factor(predictions1, levels = levels(test_data$class))

# GLMNET
predictions2 <- predict(glmnet_model, newdata = test_data, type = "prob")
predictions2 <- ifelse(predictions2[,"Yes"] > 0.5, "Yes", "No")
predictions2 <- factor(predictions2, levels = levels(test_data$class))

# SVM
predictions_svm <- predict(svm_model, newdata = test_data, probability = TRUE)
predictions_svm <- factor(predictions_svm, levels = levels(test_data$class))

# Evaluate model performance
conf_matrix1 <- confusionMatrix(predictions1, test_data$class)
conf_matrix2 <- confusionMatrix(predictions2, test_data$class)
conf_matrix_svm <- confusionMatrix(predictions_svm, test_data$class)

print("GLM Model Results:")
print(conf_matrix1)

print("GLMNET Model Results:")
print(conf_matrix2)

print("SVM Model Results:")
print(conf_matrix_svm)

# Convert data to DMatrix format for XGBoost
train_matrix <- xgb.DMatrix(
  data = as.matrix(train_data[, -1]),  # exclude class column
  label = as.numeric(train_data$class) - 1  # convert to 0/1
)

test_matrix <- xgb.DMatrix(
  data = as.matrix(test_data[, -1]),
  label = as.numeric(test_data$class) - 1
)

# Set XGBoost parameters
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train XGBoost model
xgb_model <- xgb.train(
  params = xgb_params,
  data = train_matrix,
  nrounds = 100,
  verbose = 0
)

# Make XGBoost predictions
predictions_xgb <- predict(xgb_model, test_matrix)
predictions_xgb <- ifelse(predictions_xgb > 0.5, "Yes", "No")
predictions_xgb <- factor(predictions_xgb, levels = levels(test_data$class))

# Add XGBoost evaluation
conf_matrix_xgb <- confusionMatrix(predictions_xgb, test_data$class)


# Compare ROC curves
roc_glm <- roc(test_data$class, as.numeric(predictions1))
roc_glmnet <- roc(test_data$class, as.numeric(predictions2))
roc_svm <- roc(test_data$class, as.numeric(predictions_svm))

# Plot ROC curves
plot(roc_glm, col = "blue", main = "ROC Curves Comparison")
lines(roc_glmnet, col = "red")
lines(roc_svm, col = "green")
legend("bottomright", 
       legend = c("GLM", "GLMNET", "SVM"),
       col = c("blue", "red", "green"),
       lwd = 2)
