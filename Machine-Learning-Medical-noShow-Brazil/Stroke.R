# Import required packages
library(caret)
library(corrplot)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggthemes)
library(MASS)
library(rpart)
library(rpart.plot)

# Load the data
stroke_data <- read.csv("healthcare-dataset-stroke-data.csv")

# Initial data cleaning
stroke_data$stroke <- as.factor(stroke_data$stroke)
stroke_data$bmi <- as.numeric(stroke_data$bmi)
stroke_data <- na.omit(stroke_data) # Handle missing values

# Create visualization for numerical variables
numerical_vars <- stroke_data %>% 
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value")

ggplot(numerical_vars) +
  geom_boxplot(aes(x = as.factor(stroke_data$stroke), y = value)) +
  facet_wrap(~variable, scales = "free") +
  labs(title = "Numerical Variables Distribution by Stroke",
       x = "Stroke",
       y = "Value") +
  theme_minimal()

# Correlation plot for numerical variables
numerical_data <- stroke_data %>% select_if(is.numeric)
corrplot(cor(numerical_data), method = "circle", type = "lower")

# Create training and testing sets
set.seed(123)
trainIndex <- createDataPartition(stroke_data$stroke, p = 0.7, list = FALSE)
training <- stroke_data[trainIndex, ]
testing <- stroke_data[-trainIndex, ]

# Create preprocessing object
preProcess_range <- preProcess(training, method = c("center", "scale"))
training_processed <- predict(preProcess_range, training)
testing_processed <- predict(preProcess_range, testing)

# Create training and testing sets
set.seed(123)
trainIndex <- createDataPartition(stroke_data$stroke, p = 0.7, list = FALSE)
training <- stroke_data[trainIndex, ]
testing <- stroke_data[-trainIndex, ]

# Create preprocessing object
preProcess_range <- preProcess(training, method = c("center", "scale"))
training_processed <- predict(preProcess_range, training)
testing_processed <- predict(preProcess_range, testing)

# Train multiple models
control <- trainControl(method = "cv", number = 10, classProbs = TRUE)

# Logistic Regression
set.seed(123)
model_glm <- train(stroke ~ ., 
                   data = training_processed,
                   method = "glm",
                   trControl = control)

# Random Forest
set.seed(123)
model_rf <- train(stroke ~ .,
                  data = training_processed,
                  method = "rf",
                  trControl = control)

# SVM
set.seed(123)
model_svm <- train(stroke ~ .,
                   data = training_processed,
                   method = "svmRadial",
                   trControl = control)

# Compare models
models <- list(GLM = model_glm, RF = model_rf, SVM = model_svm)
results <- resamples(models)
summary(results)

# Plot model comparison
dotplot(results)

# Get predictions for best model (assuming RF is best)
predictions <- predict(model_rf, testing_processed)
conf_matrix <- confusionMatrix(predictions, testing_processed$stroke)
print(conf_matrix)