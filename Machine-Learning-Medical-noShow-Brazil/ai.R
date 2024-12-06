## Lab2: Binary Classification Machine Learning Project for Medical No-Shows

# Import packages
library(dplyr)
library(tidyr)
library(corrgram)
library(ggplot2)
library(ggthemes)
library(cluster)
library(caret)
library(MASS)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
library(rpart.plot)
library(ROSE)
library(lubridate)
library(skimr)
library(corrplot)

# Read and prepare dataset
noShow <- read.csv("KaggleV2-May-2016.csv")

# Basic dataset exploration
dim(noShow)
sapply(noShow, class)
head(noShow, n=20)
str(noShow)
summary(noShow)

# Detailed summary using skimr
skim(noShow)

# Remove PatientId and AppointmentID explicitly because of MASS package error
noShow <- noShow %>% dplyr::select(-PatientId, -AppointmentID)


# Correlation Analysis
numeric_vars <- noShow[sapply(noShow, is.numeric)]
corrplot(cor(numeric_vars), 
         type="lower", 
         tl.srt = 90,
         main = "Correlation Matrix of Numeric Variables")


noShow_clean <- noShow %>%
  # Rename variables for clarity
  dplyr::rename(
    Hypertension = Hipertension,
    Disabled = Handcap,
    WelfareBenefits = Scholarship
  ) %>%
  # First calculate no-show rates per neighborhood
  group_by(Neighbourhood) %>%
  mutate(
    noshow_rate = mean(No.show == "Yes")
  ) %>%
  ungroup() %>%
  # Remove weekend appointments
  filter(!wday(AppointmentDay) %in% c(1, 7)) %>%  # Remove Sunday (1) and Saturday (7)
  # Create features and convert variables
  mutate(
    # Convert dates
    ScheduledDay = ymd_hms(ScheduledDay),
    AppointmentDay = ymd_hms(AppointmentDay),
    
    # Create waiting days variable
    WaitingDays = as.numeric(difftime(AppointmentDay, ScheduledDay, units = "days")),
    
    # Create waiting days categories
    WaitingDays_cat = factor(
      cut(WaitingDays,
          breaks = c(-Inf, 0, 1, 7, 30, Inf),
          labels = c("Same day", "Next day", "Within week",
                     "Within month", "Over month")
      ),
      ordered = TRUE
    ),
    
    # Create weekday (as factor) - now only Monday to Friday
    WeekDay = factor(
      wday(AppointmentDay),
      levels = 2:6,  # Monday to Friday
      labels = c("Monday", "Tuesday", "Wednesday", 
                 "Thursday", "Friday")
    ),
    
    # Create NHS age groups
    AgeGroup = factor(
      cut(Age, 
          breaks = c(-Inf, 15, 24, 44, 64, 84, Inf),
          labels = c("Children", "Young People", "Working Age", 
                     "Middle Age", "Senior", "Elderly")
      ),
      ordered = TRUE
    ),
    
    # Convert Gender to factor
    Gender = factor(Gender),
    
    # Create neighborhood risk categories
    Neighbourhood = factor(
      case_when(
        noshow_rate <= quantile(noshow_rate, 0.33) ~ "Low Risk",
        noshow_rate <= quantile(noshow_rate, 0.66) ~ "Medium Risk",
        TRUE ~ "High Risk"
      ),
      levels = c("Low Risk", "Medium Risk", "High Risk"),
      ordered = TRUE
    ),
    
    # Convert target to factor
    No.show = factor(No.show == "Yes", 
                     levels = c(FALSE, TRUE),
                     labels = c("0", "1")),
    
    # Convert binary variables to factors
    WelfareBenefits = factor(WelfareBenefits,
                             levels = c(0, 1),
                             labels = c("No", "Yes")),
    SMS_received = factor(SMS_received,
                          levels = c(0, 1),
                          labels = c("No", "Yes")),
    
    # Create health conditions count
    HealthConditions = Hypertension + Diabetes + Alcoholism + (Disabled > 0)
  ) %>%
  # Select final variables
  dplyr::select(
    No.show,
    AgeGroup,
    Gender,
    WaitingDays_cat,
    WeekDay,
    Neighbourhood,
    SMS_received,
    HealthConditions,
    WelfareBenefits
  )

# clean and preprocess data numeric
noShow_clean <- noShow %>%
  # Rename variables for clarity
  dplyr::rename(
    Hypertension = Hipertension,
    Disabled = Handcap,
    WelfareBenefits = Scholarship
  ) %>%
  # First calculate no-show rates per neighborhood
  group_by(Neighbourhood) %>%
  mutate(
    noshow_rate = mean(No.show == "Yes")
  ) %>%
  ungroup() %>%
  # Create features and convert variables
  mutate(
    # Convert dates
    ScheduledDay = ymd_hms(ScheduledDay),
    AppointmentDay = ymd_hms(AppointmentDay),
    
    # Create waiting days variable (numeric)
    WaitingDays = as.numeric(difftime(AppointmentDay, ScheduledDay, units = "days")),
    
    # Create waiting days categories (numeric 1-5)
    WaitingDays_cat = as.numeric(cut(WaitingDays,
                                     breaks = c(-Inf, 0, 1, 7, 30, Inf),
                                     labels = c("Same day", "Next day", "Within week",
                                                "Within month", "Over month"))),
    
    # Create weekday (numeric 1-7)
    WeekDay = wday(AppointmentDay),
    
    # Create NHS age groups (numeric 1-6)
    AgeGroup = as.numeric(cut(Age, 
                              breaks = c(-Inf, 15, 24, 44, 64, 84, Inf),
                              labels = c("Children", "Young People", "Working Age", 
                                         "Middle Age", "Senior", "Elderly"))),
    
    # Convert Gender to numeric
    Gender = as.numeric(factor(Gender)) - 1,  # -1 to make it 0/1
    
    # Create neighborhood risk categories (numeric 1-3)
    Neighbourhood = case_when(
      noshow_rate <= quantile(noshow_rate, 0.33) ~ 1,  # Low risk
      noshow_rate <= quantile(noshow_rate, 0.66) ~ 2,  # Medium risk
      TRUE ~ 3  # High risk
    ),
    
    # Convert target variable to numeric
    No.show = as.numeric(No.show == "Yes"),  # 0 = Show, 1 = No-show
    
    # Convert binary variables to numeric
    WelfareBenefits = as.numeric(WelfareBenefits),
    SMS_received = as.numeric(SMS_received),
    
    # Create health conditions count
    HealthConditions = Hypertension + Diabetes + Alcoholism + (Disabled > 0)
  ) %>%
  # Select final variables
  dplyr::select(
    No.show,
    #Age,
    AgeGroup,
    Gender,
    #WaitingDays,
    WaitingDays_cat,
    WeekDay,
    Neighbourhood,
    SMS_received,
    HealthConditions,
    WelfareBenefits
  )

# Document the numeric encodings
encodings <- list(
  "No.show" = c("0 = Show", "1 = No-show"),
  "AgeGroup" = c("1 = Children", "2 = Young People", "3 = Working Age", 
                 "4 = Middle Age", "5 = Senior", "6 = Elderly"),
  "WaitingDays_cat" = c("1 = Same day", "2 = Next day", "3 = Within week",
                        "4 = Within month", "5 = Over month"),
  "WeekDay" = c("1-7 = Sunday to Saturday"),
  "Neighbourhood" = c("1 = Low Risk", "2 = Medium Risk", "3 = High Risk"),
  "Gender" = c("0 = F", "1 = M"),
  "Other variables" = c("0 = No", "1 = Yes")
)






# Define continuous numeric columns for scaling
numeric_cols <- c("HealthConditions")  # Only truly continuous variable left

# Scale numerical features
preproc <- preProcess(noShow_clean[numeric_cols], method = c("center", "scale"))
noShow_clean[numeric_cols] <- predict(preproc, noShow_clean[numeric_cols])

# Check class balance
table(noShow_clean$No.show)
prop.table(table(noShow_clean$No.show))

# Convert ordered factors to regular factors before ROSE
noShow_clean_for_rose <- noShow_clean %>%
  mutate(
    across(where(is.ordered), ~factor(., ordered = FALSE)),
    # Ensure No.show is a regular factor
    No.show = factor(No.show, ordered = FALSE)
  )

# Verify structure
str(noShow_clean_for_rose)

# Apply ROSE
balanced_data <- ROSE(No.show ~ ., data = noShow_clean_for_rose)$data

# Convert back to ordered factors if needed
balanced_data <- balanced_data %>%
  mutate(
    WaitingDays_cat = factor(WaitingDays_cat, 
                             levels = c("Same day", "Next day", "Within week",
                                        "Within month", "Over month"),
                             ordered = TRUE),
    AgeGroup = factor(AgeGroup,
                      levels = c("Children", "Young People", "Working Age", 
                                 "Middle Age", "Senior", "Elderly"),
                      ordered = TRUE),
    Neighbourhood = factor(Neighbourhood,
                           levels = c("Low Risk", "Medium Risk", "High Risk"),
                           ordered = TRUE)
  )

# Verify the balance
table(balanced_data$No.show)
prop.table(table(balanced_data$No.show))





# Feature Selection using Stepwise Regression
balanced_data$No.show <- as.factor(balanced_data$No.show)
initial_model <- glm(No.show ~ ., family=binomial(link='logit'), data=noShow_clean)
stepwise_model <- stepAIC(initial_model, direction="both")
print(summary(stepwise_model))

# drop Age etc in favour of new categories



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


# Boxplot for numeric variables
numeric_vars <- balanced_data %>% select_if(is.numeric)

# Create long format data for plotting
long_data <- numeric_vars %>%
  gather(key = "Variable", value = "Value")

# Create boxplot
ggplot(long_data, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.alpha = 0.5) +
  theme_minimal() +
  labs(
    title = "Distribution of Numeric Variables",
    x = "Variables",
    y = "Values"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )


# Boxplot for numeric variables
numeric_vars <- balanced_data %>% select_if(is.numeric)

# Create long format data for plotting
long_data <- numeric_vars %>%
  gather(key = "Variable", value = "Value")

# Create boxplot
ggplot(long_data, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.alpha = 0.5) +
  theme_minimal() +
  labs(
    title = "Distribution of Numeric Variables",
    x = "Variables",
    y = "Values"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Convert balanced_data to have proper factor levels
model_data <- balanced_data %>%
  mutate(
    # Keep categorical variables as factors with valid R names
    WaitingDays_cat = factor(WaitingDays_cat,
                             labels = make.names(levels(WaitingDays_cat))),
    AgeGroup = factor(AgeGroup,
                      labels = make.names(levels(AgeGroup))),
    Neighbourhood = factor(Neighbourhood,
                           labels = make.names(levels(Neighbourhood))),
    WeekDay = factor(WeekDay,
                     labels = make.names(levels(WeekDay))),
    Gender = factor(Gender,
                    labels = make.names(levels(Gender))),
    SMS_received = factor(SMS_received,
                          labels = c("No", "Yes")),
    WelfareBenefits = factor(WelfareBenefits,
                             labels = c("No", "Yes")),
    
    # Keep numeric variables as numeric
    HealthConditions = as.numeric(HealthConditions),
    
    # Target variable as factor with valid R names
    No.show = factor(ifelse(No.show == "NoShow", 1, 0), 
                     levels = c(0, 1),
                     labels = c("Show", "NoShow"))
  )

# Verify the conversion
str(model_data)

# Verify the levels
print("Factor levels:")
print(levels(model_data$No.show))
print("Distribution:")
print(table(model_data$No.show))

# Verify all variables are numeric
str(model_data)
sapply(model_data, class)  # Should all be "numeric"

# Document the numeric encodings for interpretation
numeric_encodings <- list(
  "No.show" = c("0 = Show", "1 = No-show"),
  "AgeGroup" = c("1 = Children", "2 = Young People", "3 = Working Age", 
                 "4 = Middle Age", "5 = Senior", "6 = Elderly"),
  "WeekDay" = c("1 = Monday", "2 = Tuesday", "3 = Wednesday", 
                "4 = Thursday", "5 = Friday"),
  "WaitingDays_cat" = c("1 = Same day", "2 = Next day", "3 = Within week",
                        "4 = Within month", "5 = Over month"),
  "Neighbourhood" = c("1 = Low Risk", "2 = Medium Risk", "3 = High Risk"),
  "Gender" = c("0 = F", "1 = M"),
  "Other variables" = c("0 = No", "1 = Yes")  # For SMS_received and WelfareBenefits
)

# Correlation Analysis
numeric_vars <- model_data[sapply(model_data, is.numeric)]
corrplot(cor(numeric_vars), 
         type="lower", 
         tl.srt = 90,
         main = "Correlation Matrix of Numeric Variables")



# 2. Data Split
set.seed(123)  # for reproducibility
train_index <- createDataPartition(balanced_data$No.show, p = 0.7, list = FALSE)
train_data <- balanced_data[train_index, ]
test_data <- balanced_data[-train_index, ]

# Update control settings for numeric target
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)


# Convert No.show to factor for modeling
train_data$No.show <- factor(train_data$No.show, 
                             levels = c(0, 1),
                             labels = c("Show", "NoShow"))
test_data$No.show <- factor(test_data$No.show,
                            levels = c(0, 1),
                            labels = c("Show", "NoShow"))

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
set.seed(234)
rf_model <- train(
  No.show ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  metric = "ROC"
)

xgb_model <- train(
  No.show ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  metric = "RMSE",  # Changed because we're using numeric
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

# Logistic Regression
glm_model <- train(
  factor(No.show) ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

# 4. Make predictions
rf_pred <- predict(rf_model, test_data, type = "prob")[,2]
xgb_pred <- predict(xgb_model, test_data, type = "prob")[,2]
glm_pred <- predict(glm_model, test_data, type = "prob")[,2]

# 5. Create ensemble predictions (weighted average)
ensemble_pred <- (rf_pred * 0.4) + (xgb_pred * 0.4) + (glm_pred * 0.2)

# 6. Convert to binary predictions
ensemble_binary <- ifelse(ensemble_pred > 0.5, 1, 0)

# 7. Evaluate ensemble
# Create confusion matrix
conf_matrix <- confusionMatrix(
  factor(ensemble_binary),
  factor(test_data$No.show)
)

# ROC curve
roc_obj <- roc(test_data$No.show, ensemble_pred)
auc_value <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))

# 8. Model comparison
results <- resamples(list(
  RF = rf_model,
  XGB = xgb_model,
  GLM = glm_model
))

# Plot model comparison
bwplot(results)

# 9. Print performance metrics
print("Individual Model Performance:")
print(rf_model)
print(xgb_model)
print(glm_model)

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