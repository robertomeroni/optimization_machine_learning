# Load necessary libraries
library(ggplot2)
library(dplyr)
library(readr)
library(GGally)
library(corrplot)
library(gridExtra)
library(car)

# Load the data with explicit delimiter
data <- read_delim('BC-Data-Set.csv', delim = ';', col_types = cols())

# Display the first few rows of the data to check for correct import
print(head(data))

# Print a summary of the data
summary(data)

# Calculate correlation matrix for numeric columns
cor_matrix <- cor(data %>% select_if(is.numeric), use = "complete.obs")

# Print the correlation matrix
print("Correlation Matrix:")
print(cor_matrix)

# Flatten the correlation matrix to identify the most relevant correlations
flatten_correlation_matrix <- function(cor_matrix) {
  ut <- upper.tri(cor_matrix)
  data.frame(
    row = rownames(cor_matrix)[row(cor_matrix)[ut]],
    column = rownames(cor_matrix)[col(cor_matrix)[ut]],
    cor  =(cor_matrix)[ut]
  )
}

flat_cor_matrix <- flatten_correlation_matrix(cor_matrix)

# Print the most relevant correlations (top and bottom)
most_relevant <- flat_cor_matrix %>%
  arrange(desc(abs(cor))) %>%
  head(10)

print("Most Relevant Correlations:")
print(most_relevant)

# Plot the correlation matrix
plot_corr_matrix <- corrplot(cor_matrix, method = "circle")

# Create scatter plots for each combination of variables
# Ensure there are no NULL values and data types are correct for ggpairs
numeric_data <- data %>% select_if(is.numeric)

# Generate scatter plots for each pair of variables with enhanced readability
plot_pairs <- ggpairs(numeric_data,
                      upper = list(continuous = wrap("cor", size = 6)),
                      lower = list(continuous = wrap("points", alpha = 0.15, size=0.2, color = "blue")),
                      diag = list(continuous = wrap("barDiag", bins = 20)))

# Print the plot to the console
print(plot_pairs)

# Save the scatter plots and correlation matrix
# ggsave("fig/scatter_plots.png", plot = plot_pairs, width = 20, height = 20)

# Ensure the data has been loaded and date column converted to POSIXct
data$date <- as.POSIXct(data$date, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

# Create individual plots for each pollutant
plot_bc <- ggplot(data, aes(x = date, y = BC)) +
  geom_line(color = "blue", size = 0.3) +
  labs(title = "Temporal Trend of BC (µgr/m3)",
       x = "Date",
       y = "BC (µgr/m3)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12))

plot_nox <- ggplot(data, aes(x = date, y = NOX)) +
  geom_line(color = "green", size = 0.3) +
  labs(title = "Temporal Trend of NOX (µgr/m3)",
       x = "Date",
       y = "NOX (µgr/m3)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12))

plot_O3 <- ggplot(data, aes(x = date, y = O3)) +
  geom_line(color = "red", size = 0.3) +
  labs(title = "Temporal Trend of O3 (µgr/m3)",
       x = "Date",
       y = "O3 (µgr/m3)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12))



# Arrange the plots in a grid layout
grid_temporal_trends <- grid.arrange(plot_bc, plot_nox, plot_O3, ncol = 1)

# Save the individual plots to a single file
# ggsave("fig/temporal_trends_individual.png", grid_temporal_trends, width = 12, height = 16)

# Create a new column for the day of the week
data$day_of_week <- weekdays(data$date)

# Calculate the average concentration for each day of the week
average_by_day <- data %>%
  group_by(day_of_week) %>%
  summarise(across(c(BC, NOX, N_CPC, CO), ~mean(.x, na.rm = TRUE)))

# Print the average concentrations by day of the week
print("Average Concentrations by Day of the Week:")
print(average_by_day)

# Reorder the days of the week for plotting
average_by_day$day_of_week <- factor(average_by_day$day_of_week,
                                     levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

# Normalize the concentrations
normalize <- function(x) {
  return ((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

normalized_data <- average_by_day %>%
  mutate(across(c(BC, NOX, N_CPC, CO), normalize))

# Plot the normalized average concentrations by day of the week
plot_avg_by_day <- ggplot(normalized_data, aes(x = day_of_week, group = 1)) +
  geom_line(aes(y = BC, color = "BC"), linewidth = 1) +
  geom_line(aes(y = NOX, color = "NOX"), linewidth = 1) +
  geom_line(aes(y = N_CPC, color = "N_CPC"), linewidth = 1) +
  geom_line(aes(y = CO, color = "CO"), linewidth = 1) +
  labs(title = "Concentrations of Various Air Pollutants by Day of the Week",
       x = "Day of the Week",
       y = "Normalized Average Concentration",
       color = "Pollutant") +
  scale_color_manual(values = c("BC" = "blue",
                                "NOX" = "green",
                                "N_CPC" = "red",
                                "CO" = "yellow")) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

# Print the plot to the console
print(plot_avg_by_day)

# Save the normalized average concentrations plot
# ggsave("fig/avg_by_day_concentrations.png", plot=plot_avg_by_day, width = 12, height = 8)

# Perform ANOVA for each pollutant
anova_results <- list()
for (pollutant in c("BC", "NOX", "N_CPC", "CO")) {
  formula <- as.formula(paste(pollutant, "~ day_of_week"))
  anova_result <- aov(formula, data = data)
  anova_results[[pollutant]] <- summary(anova_result)
  print(paste("ANOVA results for", pollutant, ":"))
  print(summary(anova_result))
}

# If ANOVA is significant, perform post-hoc tests
post_hoc_results <- list()
for (pollutant in c("BC", "NOX", "N_CPC", "CO")) {
  anova_result <- aov(as.formula(paste(pollutant, "~ day_of_week")), data = data)
  p_value <- summary(anova_result)[[1]][["Pr(>F)"]][1]
  
  if (p_value < 0.05) {
    post_hoc_result <- TukeyHSD(anova_result)
    post_hoc_results[[pollutant]] <- post_hoc_result
    print(paste("Post-hoc test results for", pollutant, ":"))
    print(post_hoc_result)
  }
}

# Function to determine the season based on the date
get_season <- function(date) {
  month <- as.numeric(format(date, "%m"))
  day <- as.numeric(format(date, "%d"))
  
  if ((month == 12 && day >= 21) || (month == 1) || (month == 2) || (month == 3 && day < 20)) {
    return("Winter")
  } else if ((month == 3 && day >= 20) || (month == 4) || (month == 5) || (month == 6 && day < 21)) {
    return("Spring")
  } else if ((month == 6 && day >= 21) || (month == 7) || (month == 8) || (month == 9 && day < 22)) {
    return("Summer")
  } else if ((month == 9 && day >= 22) || (month == 10) || (month == 11) || (month == 12 && day < 21)) {
    return("Fall")
  }
}

# Add a new column for the season
data$season <- sapply(data$date, get_season)

# Calculate the average concentration for each season
average_by_season <- data %>%
  group_by(season) %>%
  summarise(across(c(BC, N_CPC, `PM-10`, `PM-1.0`, O3, CO, NOX), ~mean(.x, na.rm = TRUE)))

# Print the average concentrations by season
print("Average Concentrations by Season:")
print(average_by_season)

# Perform ANOVA for each pollutant
anova_results <- list()
for (pollutant in c("BC", "N_CPC", "PM-10", "PM-1.0", "O3", "CO", "NOX")) {
  if (pollutant %in% c("PM-10", "PM-1.0")) {
    formula <- as.formula(paste("`", pollutant, "` ~ season", sep = ""))
  } else {
    formula <- as.formula(paste(pollutant, "~ season"))
  }
  anova_result <- aov(formula, data = data)
  anova_results[[pollutant]] <- anova_result
  print(paste("ANOVA results for", pollutant, ":"))
  print(summary(anova_result))
}

# If ANOVA is significant, perform post-hoc tests
post_hoc_results <- list()
for (pollutant in c("BC", "N_CPC", "PM-10", "PM-1.0", "O3", "CO", "NOX")) {
  anova_result <- anova_results[[pollutant]]
  p_value <- summary(anova_result)[[1]][["Pr(>F)"]][1]
  
  if (p_value < 0.05) {
    post_hoc_result <- TukeyHSD(anova_result)
    post_hoc_results[[pollutant]] <- post_hoc_result
    print(paste("Post-hoc test results for", pollutant, ":"))
    print(post_hoc_result)
  }
}