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

undersample_data <- function(data, target_col, ratio = 1) {
  # Get the minority and majority classes
  class_counts <- table(data[[target_col]])
  minority_class <- names(which.min(class_counts))
  majority_class <- names(which.max(class_counts))
  
  # Separate minority and majority classes
  minority_data <- data[data[[target_col]] == minority_class,]
  majority_data <- data[data[[target_col]] == majority_class,]
  
  # Calculate number of samples to keep from majority class
  n_minority <- nrow(minority_data)
  n_keep <- floor(n_minority * ratio)
  
  # Randomly sample from majority class
  set.seed(123)
  majority_sampled <- majority_data[sample(nrow(majority_data), n_keep),]
  
  # Combine minority and sampled majority data
  balanced_data <- rbind(minority_data, majority_sampled)
  
  # Shuffle the data
  balanced_data[sample(nrow(balanced_data)),]
}

# Apply undersampling to your data
balanced_data <- undersample_data(noShow, "No.show", ratio = 1)

# Check class distribution after undersampling
print("Class distribution after undersampling:")
print(table(balanced_data$No.show))

# Ensure all categorical variables are factors
balanced_data <- balanced_data %>%
  mutate(
    No.show = as.factor(No.show),
    Gender = as.factor(Gender),
    WelfareBenefits = as.factor(WelfareBenefits),
    #Disabled = as.factor(Disabled),
    SMS_received = as.factor(SMS_received),
    WaitingDays_cat = factor(WaitingDays_cat, ordered = TRUE),
    #WeekDay = as.factor(WeekDay),
    NeighbourhoodRisk = factor(NeighbourhoodRisk, ordered = TRUE),
    AgeGroup = factor(AgeGroup, ordered = TRUE)
  )

balanced_data <- balanced_data %>%
  dplyr::select(
    # Target variable
    No.show,
    # Significant predictors from stepwise regression
    #Gender,                # Significant (p < 0.001)
    WelfareBenefits,      # Significant (p < 0.001)
    #Disabled,             # Significant (p < 0.01)
    SMS_received,         # Significant (p < 0.001)
    WaitingDays_cat,      # Very significant (all levels p < 0.001)
    #WeekDay,              # Some days significant
    NeighbourhoodRisk,    # Significant (p < 0.001)
    AgeGroup             # Very significant (most levels p < 0.001)
)

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(balanced_data$No.show, p = 0.8, list = FALSE)
train_data <- balanced_data[train_index, ]
test_data <- balanced_data[-train_index, ]




# Define a grid of hyperparameters to tune
tune_grid <- expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0, 1, by = 0.1))

# Train the model with hyperparameter tuning
glmnet_model <- train(No.show ~ ., data = train_data, method = "glmnet",family = "binomial",tuneGrid = tune_grid, trControl = trainControl(method = "cv", number = 10))
summary (glmnet_model)


# Train a logistic regression model
glm_model <- train(No.show ~ ., data = train_data, method = "glm", family = "binomial")


summary (glm_model)


# Adjusting thresholds


# For glm model
predictions1 <- predict(glm_model, newdata = test_data, type = "prob")
predictions1 <- ifelse(predictions1[,"Yes"] > 0.5, "Yes", "No")
predictions1 <- factor(predictions1, levels = levels(test_data$No.show))

# For glmnet model
predictions2 <- predict(glmnet_model, newdata = test_data, type = "prob")
predictions2 <- ifelse(predictions2[,"Yes"] > 0.5, "Yes", "No")
predictions2 <- factor(predictions2, levels = levels(test_data$No.show))

# Evaluate model performance
conf_matrix1 <- confusionMatrix(predictions1, test_data$No.show)
conf_matrix2 <- confusionMatrix(predictions2, test_data$No.show)

print("GLM Model Results:")
print(conf_matrix1)

print("GLMNET Model Results:")
print(conf_matrix2)

# Evaluate model performance
confusion_matrix <- confusionMatrix(predictions1, test_data$No.show)
print(confusion_matrix)
