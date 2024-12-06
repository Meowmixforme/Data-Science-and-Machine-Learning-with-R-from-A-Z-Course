library(ggplot2)
library(dplyr)

# Set script current directory as working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read and prepare data
Train_df <- read.csv("exoTrain.csv")

# Count NAs by column
colSums(is.na(Train_df))

# Check how many labels are present in the train df
unique(Train_df$LABEL)

# Extract the index for the stars labelled as 2
indices <- which(Train_df$LABEL == 2)
print(indices)

# Plotted data balance
ggplot(Train_df, aes(x = factor(LABEL))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Labels",
       x = "Label",
       y = "Count") +
  theme_minimal() +
  geom_text(stat = 'count', 
            aes(label = paste0(..count.., "\n(", 
                               round(..count../nrow(Train_df)*100, 1), "%)")),
            vjust = -0.5)

# Replacing labels
Train_df$LABEL <- replace(Train_df$LABEL, Train_df$LABEL == 2, 1)
Train_df$LABEL <- replace(Train_df$LABEL, Train_df$LABEL == 1, 0)

# BoxPlots of outliers
par(mfrow = c(1, 3))
for(i in 1:3) {
  col_name <- paste0("FLUX.", i)
  boxplot(Train_df[[col_name]] ~ Train_df$LABEL,
          main = paste("Boxplot of", col_name),
          xlab = "LABEL",
          ylab = col_name)
}

# Drop extreme outliers
Train_df <- Train_df %>%
  filter(FLUX.2 <= 0.25e6)

