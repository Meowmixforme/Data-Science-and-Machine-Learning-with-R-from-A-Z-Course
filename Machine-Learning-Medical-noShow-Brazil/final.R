#James Fothergill
#v8255920


##################
# 1. DATA LOADING AND INITIAL CLEANING
##################
# Load required packages (xgboost, dplyr, tidyr, etc.)
# Read appointment data
# Remove unnecessary ID columns
# Rename columns for UK context (Hypertension, Disabled, WelfareBenefits)

##################
# 2. EXPLORATORY DATA ANALYSIS
##################
# Basic dataset exploration (dimensions, structure, summary)
# Correlation analysis of numeric variables
# Initial visualizations of key variables
# Identify data quality issues (extreme ages)

##################
# 3. FEATURE ENGINEERING
##################
# Time-based Features:
# Calculate WaitingDays from appointment dates
# Create WaitingDays categories (Same day to Over month)
# Extract WeekDay from AppointmentDay
# Remove weekend appointments

# Location Feature:
# Calculate Neighbourhood no-show rates
# Create NeighbourhoodRisk categories (Low/Medium/High)

# Health Features:
# Combine Alcoholism, Diabetes, Hypertension into HealthConditions
# Create NHS age groups (Children to Elderly)

# Data Cleaning:
# Remove extreme ages (>100)
# Remove original variables replaced by categories

##################
# 4. DATA PREPROCESSING AND MODELING
##################
# Data Preparation:
# Remove date columns
# Convert character variables to factors
# Convert ordered factors to regular factors

# Model Development:
# Balance classes using ROSE
# Perform feature selection using stepwise regression
# Analyze odds ratios for no-show risk factors



library(xgboost)
library(dplyr)
library(tidyr)
library(corrgram)
library(ggplot2)
library(ggthemes)
library(cluster)
library(caret)
library(MASS)
library(randomForest)
library(e1071)
library(pROC)
library(rpart.plot)
library(ROSE)
library(lubridate)
library(skimr)
library(corrplot)

# Set script current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read and prepare dataset
noShow <- read.csv("KaggleV2-May-2016.csv")

# Remove the Id's as they are not useful
noShow <- noShow %>% dplyr::select(-PatientId, -AppointmentID)

# Basic dataset exploration
dim(noShow)
sapply(noShow, class)
head(noShow, n=20)
str(noShow)
summary(noShow)

# Detailed summary using skimr
skim(noShow)

# Correlation Analysis
numeric_vars <- noShow[sapply(noShow, is.numeric)]
corrplot(cor(numeric_vars), 
         type="lower", 
         tl.srt = 90,
         main = "Correlation Matrix of Numeric Variables")


# Medium correlations between Diabetes and Hipertension, less so with alcoholism.
# Strong correlation with Age and Hipertension, Medium with Diabetes and less so with Alcoholism.



# Rename three of the columns to have better meanings in the UK

noShow <- noShow %>%
  rename(
    Hypertension = Hipertension,
    Disabled = Handcap,
    WelfareBenefits = Scholarship
  )


# Create waiting days variable by calculating the difference between ScheduleDay and AppointmentDay.
noShow <- noShow %>%
  mutate(
    WaitingDays = as.numeric(difftime(AppointmentDay, ScheduledDay, units = "days")),
    
    WaitingDays_cat = factor(
      cut(WaitingDays,
          breaks = c(-Inf, 0, 1, 7, 30, Inf),
          labels = c("Same day", "Next day", "Within week",
                     "Within month", "Over month")
      ),
      ordered = TRUE
    )
  )

# Create a WeekDay variable 
noShow <- noShow %>%
  mutate(
    WeekDay = factor(
      wday(AppointmentDay),    # Convert date to day of week
      levels = 1:7,            # All days: Sunday(1) to Saturday(7)
      labels = c("Sunday", "Monday", "Tuesday", "Wednesday", 
                 "Thursday", "Friday", "Saturday")
    )
  )

# Create plot to see the most common days
ggplot(noShow, aes(x = WeekDay, fill = No.show)) +
  geom_bar(position = "dodge") +
  labs(title = "Appointment Distribution by Day of Week",
       subtitle = "Comparing Show vs No-show appointments",
       x = "Day of Week",
       y = "Number of Appointments") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("coral", "turquoise3"))

# Sunday has no appointments and Saturday very few

# Saturday with show/no-show breakdown
noShow %>%
  filter(WeekDay == "Saturday") %>%
  count(No.show)

# 39 Appointments for Saturday. We can safely drop the weekend days.

# Only keep the weekdays:
noShow <- noShow %>%
  filter(WeekDay %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"))


# Create a plot to see distribution
ggplot(noShow, aes(x = WeekDay, fill = No.show)) +
  geom_bar(position = "dodge") +
  labs(title = "Appointment Distribution by Day of Week",
       subtitle = "Comparing Show vs No-show appointments",
       x = "Day of Week",
       y = "Number of Appointments") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("coral", "turquoise3"))

# Seeing how many unique values are in the Neighbourhood variable
n_distinct(noShow$Neighbourhood)

# With 81 distinct values a new, cleaner variable should be created to calculate the risk of no show based on neighbourhood

# Calculate Neighbourhood no show rates
neighborhood_rates <- noShow %>%
  group_by(Neighbourhood) %>%
  summarise(
    noshow_rate = mean(No.show == "Yes"),
    .groups = 'drop'
  )

# Create Neighbourhoodrisk categories based on no show intervals under 33% between 66% and over.
# Using dplyr directly because of the MASS package conflicting
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
  dplyr::select(-noshow_rate)

# Convert the binary variables to factors and give them meaningful labels for the plot
noShow_plot <- noShow %>%
  mutate(
    # Binary variables to factors
    Alcoholism = factor(Alcoholism, levels = c(0, 1), labels = c("No", "Yes")),
    Diabetes = factor(Diabetes, levels = c(0, 1), labels = c("No", "Yes")),
    Hypertension = factor(Hypertension, levels = c(0, 1), labels = c("No", "Yes")),
    SMS_received = factor(SMS_received, levels = c(0, 1), labels = c("No", "Yes")),
    WelfareBenefits = factor(WelfareBenefits, levels = c(0, 1), labels = c("No", "Yes")),
    
    # Multi-level factor
    Disabled = factor(Disabled, levels = c(0,1,2,3,4), 
                      labels = c("None", "Minor", "Moderate", "Major", "Severe")),
)


# Create plots for the continuous variable Age
p1 <- noShow_plot %>%  # Changed from noShow_plot to noShow
  dplyr::select(No.show, Age) %>%
  ggplot() +
  geom_boxplot(aes(x = No.show, y = Age)) +
  labs(title = "Distribution of Age by Show/No-show status") +
  theme_minimal()

# Create plots for categorical variables
p2 <- noShow_plot %>%  # Changed from noShow_plot to noShow
  dplyr::select(No.show, Gender, Alcoholism, Diabetes, Hypertension, 
                SMS_received, WelfareBenefits, Disabled, NeighbourhoodRisk, 
                WeekDay, WaitingDays_cat) %>%  # Fixed variable names
  gather(key = "feature", value = "value", -No.show) %>%
  ggplot() +
  geom_bar(aes(x = value, fill = No.show), position = "dodge") +
  facet_wrap(~feature, scales = "free", ncol = 2) +
  labs(title = "Distribution of categorical variables by Show/No-show status") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display plots
print(p1)
print(p2)

# The Age box plot has an extreme outlier which will need to be investigated.

# For the categorical variables, we see that very few people have Alcoholism, Diabetes or Hypertension.
# Alcoholics have slightly worse attendance, Diabetics and Hypertension both have slightly better attendance.
# All three values can be joined into a new variable called 'HealthConditions' as they make up so few cases and are all health conditions.
# Disabled is very imbalanced as there is very little data as the vast majority are 'none' and will likely be discarded.
# Benefits has slightly higher 



# Create boxplot for Age
ggplot(noShow, aes(x = No.show, y = Age)) +
  geom_boxplot(fill = "lightblue", alpha = 0.5) +
  labs(title = "Distribution of Age by Appointment Status",
       subtitle = "Comparing Show vs No-show appointments",
       x = "Appointment Status",
       y = "Age (years)") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.title = element_text(size = 11),
    axis.text = element_text(size = 10)
  )

# 1. First, let's look at the extreme ages
extreme_ages <- subset(noShow, Age > 100, select = c("Age", "No.show"))
print("Patients over 100:")
print(extreme_ages)

# There are only 7 values and of those only 3 no show

# 2. Create boxplot with and without outliers to compare
# Without outliers
p1 <- ggplot(noShow, aes(x = No.show, y = Age)) +
  geom_boxplot(fill = "lightblue", alpha = 0.5, outlier.shape = NA) +  # Remove outliers
  coord_cartesian(ylim = c(0, 100)) +  # Set y-axis limit to 100
  labs(title = "Age Distribution (Limited to 100 years)",
       x = "Appointment Status",
       y = "Age (years)") +
  theme_minimal()

# With outliers
p2 <- ggplot(noShow, aes(x = No.show, y = Age)) +
  geom_boxplot(fill = "lightblue", alpha = 0.5) +
  labs(title = "Age Distribution (With Outliers)",
       x = "Appointment Status",
       y = "Age (years)") +
  theme_minimal()

# Display both plots
print(p1)
print(p2)


# Remove the extreme ages
noShow <- noShow %>%
  filter(Age <= 100)  # Remove ages over 100


# Create HealthConditions variable and remove individual health conditions
noShow <- noShow %>%
  mutate(
    # Convert back to numeric as they're factors
    Alcoholism_num = as.numeric(as.character(Alcoholism)),
    Diabetes_num = as.numeric(as.character(Diabetes)),
    Hypertension_num = as.numeric(as.character(Hypertension)),
    
    # Create HealthConditions with separate levels
    HealthConditions = case_when(
      Diabetes_num == 1 ~ "Diabetes",
      Hypertension_num == 1 ~ "Hypertension",
      Alcoholism_num == 1 ~ "Alcoholism",
      TRUE ~ "None"
    ),
    
    # Convert to factor with specific order
    HealthConditions = factor(
      HealthConditions,
      levels = c("None", "Diabetes", "Hypertension", "Alcoholism")
    )
  ) %>%
  # Remove temporary and original columns
  dplyr::select(-Alcoholism_num, -Diabetes_num, -Hypertension_num,
                -Alcoholism, -Diabetes, -Hypertension)

# Verify the distribution
table(noShow$HealthConditions)

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

# Verify the distribution
table(noShow$AgeGroup)

# Create a visualization of age groups
ggplot(noShow, aes(x = AgeGroup, fill = No.show)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Age Groups",
       subtitle = "Comparing Show vs No-show appointments",
       x = "Age Group",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("coral", "turquoise3"))

# Check the dataset balance
table(noShow$No.show)
prop.table(table(noShow$No.show))


# Drop original variables that have been categorized
noShow_for_rose <- noShow %>%
  # Remove original variables that have categories
  dplyr::select(-Age,              
                -AppointmentDay,    
                -ScheduledDay,      
                -Neighbourhood,     
                -WaitingDays       
  ) %>%
  # Convert character columns to factors
  mutate(
    Gender = as.factor(Gender),
    # Convert ordered factors to regular factors
    across(where(is.ordered), ~factor(., ordered = FALSE)),
    # Ensure No.show is a regular factor
    No.show = factor(No.show, ordered = FALSE)
  )

# Verify structure
str(noShow_for_rose)

# Apply ROSE
balanced_data <- ROSE(No.show ~ ., data = noShow_for_rose)$data

# Check balance
table(balanced_data$No.show)
prop.table(table(balanced_data$No.show))


# Feature Selection using Stepwise Regression
balanced_data$No.show <- as.factor(balanced_data$No.show)
# Create initial model with all predictors
initial_model <- glm(No.show ~ ., 
                     family = binomial(link = 'logit'), 
                     data = balanced_data)
# Apply stepwise selection
stepwise_model <- stepAIC(initial_model, direction = "both")
# Print summary of final model
print(summary(stepwise_model))

## As we can see, the new Categories and very significant. We should keep all of the data

# Create odds ratios and confidence intervals
coef_data <- data.frame(
  Variable = names(coef(stepwise_model)),
  Estimate = coef(stepwise_model),
  SE = summary(stepwise_model)$coefficients[,"Std. Error"]
) %>%
  mutate(
    OddsRatio = exp(Estimate),
    LowerCI = exp(Estimate - 1.96*SE),
    UpperCI = exp(Estimate + 1.96*SE)
  )

# Remove intercept for better visualization
coef_data <- coef_data[-1,]

# Create forest plot
ggplot(coef_data, aes(x = reorder(Variable, OddsRatio), y = OddsRatio)) +
  geom_point() +
  geom_errorbar(aes(ymin = LowerCI, ymax = UpperCI), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(
    title = "Odds Ratios for No-Show Predictors",
    subtitle = "Values > 1 indicate increased no-show risk",
    x = "Variable",
    y = "Odds Ratio (log scale)"
  ) +
  scale_y_log10() +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 8),
    plot.title = element_text(size = 12, face = "bold")
  )

# We can see that Waiting time is the strongest predictor
# Age is a strong factor, with younger people having higher risk
# SMS reminders help reduce no-shows
# Multiple health conditions increase risk on no show

# 2. Data Split

# Select only the recommended variables based on stepwise regression
balanced_data_clean <- balanced_data %>%
  dplyr::select(
    # Target variable
    No.show,
    
    # Significant predictors from stepwise regression
    Gender,                # Significant (p < 0.001)
    WelfareBenefits,      # Significant (p < 0.001)
    Disabled,             # Significant (p < 0.01)
    SMS_received,         # Significant (p < 0.001)
    WaitingDays_cat,      # Very significant (all levels p < 0.001)
    WeekDay,              # Some days significant
    NeighbourhoodRisk,    # Significant (p < 0.001)
    HealthConditions,     # Some levels significant
    AgeGroup             # Very significant (most levels p < 0.001)
  ) %>%
  # Convert all variables to proper types
  mutate(
    # Convert character columns to factors
    across(where(is.character), as.factor),
    
    # Convert ordered factors to regular factors
    across(where(is.ordered), ~factor(., ordered = FALSE)),
    
    # Ensure No.show is properly formatted
    No.show = factor(No.show, levels = c("No", "Yes"), labels = c("Show", "NoShow")),
    
    # Convert numeric variables to numeric
    across(c(WelfareBenefits,  SMS_received), as.numeric)
  )

# Verify the structure
str(balanced_data_clean)

# Proceed with model training using the clean, selected variables
set.seed(123)
train_index <- createDataPartition(balanced_data_clean$No.show, p = 0.7, list = FALSE)
train_data <- balanced_data_clean[train_index, ]
test_data <- balanced_data_clean[-train_index, ]

# Update control settings for numeric target
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)



# Control settings
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)



# XGBoost model
xgb_model <- train(
  No.show ~ .,  # No need to convert to factor here anymore
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = expand.grid(
    nrounds = 100,
    max_depth = 6,
    eta = 0.3,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
)

# Random Forest Model
rf_model <- train(
  No.show ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  ntree = 100,  # Reduce from default 500
  tuneLength = 3  # Fewer tuning parameters
)



# Logistic Regression
glm_model <- train(
  factor(No.show) ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)


# Add Neural Network
nnet_model <- train(
  No.show ~ .,
  data = train_data,
  method = "nnet",
  trControl = ctrl,
  metric = "ROC"
)

# 4. Make predictions
rf_pred <- predict(rf_model, test_data, type = "prob")[,2]
xgb_pred <- predict(xgb_model, test_data, type = "prob")[,2]
glm_pred <- predict(glm_model, test_data, type = "prob")[,2]
nnet_pred <- predict(nnet_model, test_data, type = "prob")[,2]  # Changed from nnet_model to nnet_pred

# 5. Create ensemble predictions (weighted average) - fixed weights to sum to 1
ensemble_pred <- (rf_pred * 0.3) + (xgb_pred * 0.3) + (glm_pred * 0.2) + (nnet_pred * 0.2)  # Changed nnet_model to nnet_pred

# 6. Convert to binary predictions (removed duplicate ensemble_pred calculation)
ensemble_binary <- ifelse(ensemble_pred > 0.5, "NoShow", "Show")
ensemble_factor <- factor(ensemble_binary, levels = levels(test_data$No.show))

# Convert to factors with matching levels
ensemble_factor <- factor(ensemble_binary, levels = levels(test_data$No.show))

# Now create confusion matrix
conf_matrix <- confusionMatrix(ensemble_factor, test_data$No.show)
print(conf_matrix)

# ROC curve
roc_obj <- roc(test_data$No.show, ensemble_pred)
auc_value <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))

# 8. Model comparison
results <- resamples(list(
  RF = rf_model,
  XGB = xgb_model,
  GLM = glm_model,
  Neural = nnet_model
))

# Plot model comparison
bwplot(results)

# 9. Print performance metrics
print("Individual Model Performance:")
print(rf_model)
print(xgb_model)
print(glm_model)
print(nnet_model)

print("Ensemble Performance:")
print(conf_matrix)
print(paste("AUC:", round(auc_value, 3)))

# 10. Feature importance
# Random Forest importance
rf_importance <- varImp(rf_model)
plot(rf_importance, main = "Random Forest Feature Importance")

# XGBoost importance
xgb_importance <- varImp(xgb_model)
plot(xgb_importance, main = "XGBoost Feature Importance")

# Neural Network importance
nnet_importance <- varImp(nnet_model)
plot(nnet_importance, main = "Neural Network Feature Importance")

# 4. Make predictions
rf_pred <- predict(rf_model, test_data, type = "prob")[,2]
xgb_pred <- predict(xgb_model, test_data, type = "prob")[,2]
glm_pred <- predict(glm_model, test_data, type = "prob")[,2]
nnet_pred <- predict(nnet_model, test_data, type = "prob")[,2]

# 5. Create ensemble predictions
ensemble_pred <- (rf_pred * 0.3) + (xgb_pred * 0.3) + 
  (glm_pred * 0.2) + (nnet_pred * 0.2)

# 6. Convert to binary predictions
ensemble_binary <- ifelse(ensemble_pred > 0.5, "NoShow", "Show")
ensemble_factor <- factor(ensemble_binary, levels = levels(test_data$No.show))

# 7. Create confusion matrix and ROC curve
conf_matrix <- confusionMatrix(ensemble_factor, test_data$No.show)
roc_obj <- roc(test_data$No.show, ensemble_pred)
auc_value <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))

# 8. Model comparison
results <- resamples(list(
  RF = rf_model,
  XGB = xgb_model,
  GLM = glm_model,
  Neural = nnet_model
))

# Plot model comparison
bwplot(results)

# 9. Print performance metrics
print("Individual Model Performance:")
print(rf_model)
print(xgb_model)
print(glm_model)
print(nnet_model)

print("Ensemble Performance:")
print(conf_matrix)
print(paste("AUC:", round(auc_value, 3)))
