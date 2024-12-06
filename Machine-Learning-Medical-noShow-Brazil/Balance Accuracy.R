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

# Check the original class distribution
class_distribution <- table(noShow$No.show)
print(class_distribution)

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

# Convert the target variable (11th column) to factor
noShow$No.show <- as.factor(noShow$No.show)

# Apply SLS for Safe-level SMOTE
genData = SLS(noShow_numeric, noShow$No.show)  # Use the numeric dataset and the target variable

# Apply BLSMOTE for Borderline SMOTE
genData_BLSMOTE = BLSMOTE(noShow_numeric, noShow$No.show, K = 14, C = 45)  # Adjust parameters as needed

# Check the new class distribution
class_distribution_balanced_BLSMOTE <- table(genData_BLSMOTE$data$class)  # Access the 'data' component and the 'class' column
print(class_distribution_balanced_BLSMOTE)

# Check the synthetic data distribution
class_distribution_syn_BLSMOTE <- table(genData_BLSMOTE$syn_data$class)  # Check the synthetic data class distribution
print(class_distribution_syn_BLSMOTE)

combined_data <- rbind(genData_BLSMOTE$data, genData_BLSMOTE$syn_data)

# Convert the target variable to factor if necessary
combined_data$class <- as.factor(combined_data$class)

combined_data <- combined_data %>%
  dplyr::select(
    # Target variable
    #No.show, - renamed class
    class,
    # Significant predictors from stepwise regression
    #Gender,                # Significant (p < 0.001)
    #WelfareBenefits,      # Significant (p < 0.001)
    #Disabled,             # Significant (p < 0.01)
    #SMS_received,         # Significant (p < 0.001)
    WaitingDays_cat,      # Very significant (all levels p < 0.001)
    #WeekDay,              # Some days significant
    #NeighbourhoodRisk,    # Significant (p < 0.001)
    #HealthConditions,     # Some levels significant
    AgeGroup             # Very significant (most levels p < 0.001)
  )

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(combined_data$class, p = 0.8, list = FALSE)
train_data <- combined_data[train_index, ]
test_data <- combined_data[-train_index, ]




# Define a grid of hyperparameters to tune
tune_grid <- expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0, 1, by = 0.1))

# Train the model with hyperparameter tuning
glmnet_model <- train(class ~ ., data = train_data, method = "glmnet",family = "binomial",tuneGrid = tune_grid, trControl = trainControl(method = "cv", number = 10))
summary (glmnet_model)


# Train a logistic regression model
glm_model <- train(class ~ ., data = train_data, method = "glm", family = "binomial")


summary (glm_model)


# Adjusting thresholds


# For glm model
predictions1 <- predict(glm_model, newdata = test_data, type = "prob")
predictions1 <- ifelse(predictions1[,"Yes"] > 0.4, "Yes", "No")
predictions1 <- factor(predictions1, levels = levels(test_data$class))

# For glmnet model
predictions2 <- predict(glmnet_model, newdata = test_data, type = "prob")
predictions2 <- ifelse(predictions2[,"Yes"] > 0.5, "Yes", "No")
predictions2 <- factor(predictions2, levels = levels(test_data$class))

# Evaluate model performance
conf_matrix1 <- confusionMatrix(predictions1, test_data$class)
conf_matrix2 <- confusionMatrix(predictions2, test_data$class)

print("GLM Model Results:")
print(conf_matrix1)

print("GLMNET Model Results:")
print(conf_matrix2)

# Evaluate model performance
confusion_matrix <- confusionMatrix(predictions1, test_data$class)
print(confusion_matrix)

# After your existing prediction code, add these metrics calculations

# For GLM Model
conf_matrix1 <- confusionMatrix(predictions1, test_data$class)
balanced_accuracy1 <- conf_matrix1$byClass['Balanced Accuracy']

# For GLMNET Model
conf_matrix2 <- confusionMatrix(predictions2, test_data$class)
balanced_accuracy2 <- conf_matrix2$byClass['Balanced Accuracy']

# Print comprehensive results
print("GLM Model Results:")
print(conf_matrix1)
print(paste("Balanced Accuracy (GLM):", round(balanced_accuracy1, 4)))

print("\nGLMNET Model Results:")
print(conf_matrix2)
print(paste("Balanced Accuracy (GLMNET):", round(balanced_accuracy2, 4)))

# Optional: Compare models across different thresholds
thresholds <- seq(0.3, 0.7, by = 0.1)

# Function to calculate balanced accuracy for different thresholds
calc_balanced_accuracy <- function(model, test_data, threshold) {
  pred_prob <- predict(model, newdata = test_data, type = "prob")
  predictions <- ifelse(pred_prob[,2] > threshold, levels(test_data$class)[2], levels(test_data$class)[1])
  predictions <- factor(predictions, levels = levels(test_data$class))
  conf <- confusionMatrix(predictions, test_data$class)
  return(conf$byClass['Balanced Accuracy'])
}

# Calculate balanced accuracy for different thresholds
results <- data.frame(
  Threshold = thresholds,
  GLM_BA = sapply(thresholds, function(t) calc_balanced_accuracy(glm_model, test_data, t)),
  GLMNET_BA = sapply(thresholds, function(t) calc_balanced_accuracy(glmnet_model, test_data, t))
)

print("\nBalanced Accuracy across different thresholds:")
print(results)

# Optional: Visualize the results
if(require(ggplot2)) {
  results_long <- tidyr::pivot_longer(results, 
                                      cols = c(GLM_BA, GLMNET_BA),
                                      names_to = "Model",
                                      values_to = "Balanced_Accuracy")
  
  ggplot(results_long, aes(x = Threshold, y = Balanced_Accuracy, color = Model)) +
    geom_line() +
    geom_point() +
    theme_minimal() +
    labs(title = "Balanced Accuracy vs Threshold",
         x = "Threshold",
         y = "Balanced Accuracy")
}