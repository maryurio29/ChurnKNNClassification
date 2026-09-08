library(readr) #To load CSV data file
library(visdat) #Install visdat package to find missing values with vis_miss [In-Text Citation: (Tierney & Hughes, 2023)]
library(tidyverse) #To use pivot_longer, dummy_variables functions and essential data science collection
library(ggplot2) #To use plots
library(caTools) #For splitting data
library(caret) #Predictive models package
library(pROC) #Performance curve/ ROC/ AUC
library(tidytable) #Confusion matrix table

#Loading data
churn_data_df <- read_csv("churn_clean.csv")
glimpse(churn_data_df) 



#Duplicates Detection: No duplicates found, no treatment necessary
duplicated(churn_data_df)
sum(duplicated(churn_data_df))



#Missing Values Detection
colSums(is.na(churn_data_df))
vis_miss(churn_data_df)

#Treatment of missing values - Imputation since deletion stats conditions were not met
#Children
hist(churn_data_df$Children)#Skewed right distribution calls for median univariate imputation
churn_data_df$Children[is.na(churn_data_df$Children)] <- median(churn_data_df$Children, na.rm=TRUE)
hist(churn_data_df$Children)#To verify distribution is still skewed to the right
#Age
hist(churn_data_df$Age)#Normal/ Uniform distribution calls for mean univariate imputation
churn_data_df$Age[is.na(churn_data_df$Age)] <- mean(churn_data_df$Age, na.rm=TRUE)
hist(churn_data_df$Age)#To verify distribution is still normal/ uniform, spiked
summary(churn_data_df$Age)#Confirmed mean/ stats remained relatively close to original values
#Income
hist(churn_data_df$Income)#Skewed right distribution calls for median univariate imputation
churn_data_df$Income[is.na(churn_data_df$Income)] <- median(churn_data_df$Income, na.rm=TRUE)
hist(churn_data_df$Income)#To verify distribution is still skewed to the right
#Techie - Categorical variable calls for mode univariate imputation
churn_data_df$Techie[is.na(churn_data_df$Techie)] <- (names(which.max(table(churn_data_df$Techie))))
#Phone - Categorical variable calls for mode univariate imputation
churn_data_df$Phone[is.na(churn_data_df$Phone)] <- (names(which.max(table(churn_data_df$Phone))))
#TechSupport - Categorical/ Boolean variable calls for mode univariate imputation
churn_data_df$TechSupport[is.na(churn_data_df$TechSupport)] <- (names(which.max(table(churn_data_df$TechSupport))))
#Tenure
hist(churn_data_df$Tenure)#Bi-modal distribution can use median or mode univariate imputation, using median
churn_data_df$Tenure[is.na(churn_data_df$Tenure)] <- median(churn_data_df$Tenure, na.rm=TRUE)
hist(churn_data_df$Tenure)#To verify distribution is still bi-modal
summary(churn_data_df$Tenure)#Confirmed mean/ stats remained relatively close to original values
#Bandwidth_GB_Year
hist(churn_data_df$Bandwidth_GB_Year)##Bi-modal distribution can use median or mode univariate imputation, using median
churn_data_df$Bandwidth_GB_Year[is.na(churn_data_df$Bandwidth_GB_Year)] <- median(churn_data_df$Bandwidth_GB_Year, na.rm=TRUE)
hist(churn_data_df$Bandwidth_GB_Year)#To verify distribution is still bi-modal
summary(churn_data_df$Bandwidth_GB_Year)#Confirmed mean/ stats remained relatively close to original values

#Confirm no missing values present
colSums(is.na(churn_data_df))



#Outliers Detection Using Boxplots
#Count of outliers: 158, range: min whisker= 25.53021, max whisker= 49.28436
Lat_boxplot <- boxplot(churn_data_df$Lat)
#Count of outliers: 273, range: min whisker=-122.56348, max whisker=-65.66785
Lng_boxplot <- boxplot(churn_data_df$Lng)
#Count of outliers: 937, range: min whisker= 0, max whisker= 31795
Population_boxplot <- boxplot(churn_data_df$Population)
#Count of outliers: 302, range: min whisker= 0, max whisker= 7
Children_boxplot <- boxplot(churn_data_df$Children)
#Count of outliers: 0, range: min whisker= 18, max whisker= 89
Age_boxplot <- boxplot(churn_data_df$Age)
#Count of outliers: 758, range: min whisker= 740.66, max whisker= 78272.96
Income_boxplot <- boxplot(churn_data_df$Income)
#Count of outliers: 539, range: min whisker= 1.446910, max whisker= 19.056411
Outage_sec_perweek_boxplot <- boxplot(churn_data_df$Outage_sec_perweek)
#Count of outliers: 38, range: min whisker= 4, max whisker= 20
Email_boxplot <- boxplot(churn_data_df$Email)
#Count of outliers: 8, range: min whisker= 0, max whisker= 5
Contacts_boxplot <- boxplot(churn_data_df$Contacts)
#Count of outliers: 94, range: min whisker= 0, max whisker= 2
Yearly_equip_failure_boxplot <- boxplot(churn_data_df$Yearly_equip_failure)
#Count of outliers: 0, range: min whisker= 1.000259, max whisker= 71.999280
Tenure_boxplot <- boxplot(churn_data_df$Tenure)
#Count of outliers: 5, range: min whisker= 77.50523, max whisker= 297.31580
MonthlyCharge_boxplot <- boxplot(churn_data_df$MonthlyCharge)
#Count of outliers: 0, range: min whisker= 155.5067, max whisker= 7158.9820
Bandwidth_GB_Year_boxplot <- boxplot(churn_data_df$Bandwidth_GB_Year)
#Count of outliers: 442, range: min whisker= 2, max whisker= 5
item1_boxplot <- boxplot(churn_data_df$Item1)
#Count of outliers: 445, range: min whisker= 2, max whisker= 5
item2_boxplot <- boxplot(churn_data_df$Item2)
#Count of outliers: 418, range: min whisker= 2, max whisker= 5
item3_boxplot <- boxplot(churn_data_df$Item3)
#Count of outliers: 433, range: min whisker= 2, max whisker= 5
item4_boxplot <- boxplot(churn_data_df$Item4)
#Count of outliers: 422, range: min whisker= 2, max whisker= 5
item5_boxplot <- boxplot(churn_data_df$Item5)
#Count of outliers: 413, range: min whisker= 2, max whisker= 5
item6_boxplot <- boxplot(churn_data_df$Item6)
#Count of outliers: 454, range: min whisker= 2, max whisker= 5
item7_boxplot <- boxplot(churn_data_df$Item7)
#Count of outliers: 426, range: min whisker= 2, max whisker= 5
item8_boxplot <- boxplot(churn_data_df$Item8)

#Treatment of outliers not retained
#Children column
churn_data_df$Children[churn_data_df$Children > 6] <- NA #Replacing outliers with NA
colSums(is.na(churn_data_df)) #Check NA values
churn_data_df$Children[is.na(churn_data_df$Children)] <- median(churn_data_df$Children, na.rm = TRUE)
colSums(is.na(churn_data_df)) #Check NA values
summary(churn_data_df$Children)#Median and mean stayed relatively the same 
#Income
outliers <- churn_data_df[which(churn_data_df$Income > 78300.00), ]#Excluding to outliers to df
str(outliers)
churn_data_df <- churn_data_df[!(churn_data_df$Income > 78300), ]#Drop outliers from analysis df
str(churn_data_df)
#Outage_sec_perweek
churn_data_df$Outage_sec_perweek[churn_data_df$Outage_sec_perweek > 19] <- NA #Replacing upper bound outliers with NA
churn_data_df$Outage_sec_perweek[churn_data_df$Outage_sec_perweek < 1.5] <- NA #Replacing lower bound outliers with NA
colSums(is.na(churn_data_df)) #Check NA values
churn_data_df$Outage_sec_perweek[is.na(churn_data_df$Outage_sec_perweek)] <- median(churn_data_df$Outage_sec_perweek, na.rm = TRUE)
colSums(is.na(churn_data_df)) #Check NA values
summary(churn_data_df$Outage_sec_perweek)#Median and mean stayed relatively the same
#Email
outliers_Email_Upper <- churn_data_df[which(churn_data_df$Email > 20), ]#Excluding upper bound outliers to df
str(outliers_Email_Upper)
outliers_Email_Lower <- churn_data_df[which(churn_data_df$Email < 4), ]#Excluding lower bound outliers to df
str(outliers_Email_Lower)
churn_data_df <- churn_data_df[!(churn_data_df$Email > 20), ]#Drop upper bound outliers from analysis df
churn_data_df <- churn_data_df[!(churn_data_df$Email < 4), ]#Drop lower bound outliers from analysis df
str(churn_data_df)
#Yearly_equip_failures
churn_data_df$Yearly_equip_failure[churn_data_df$Yearly_equip_failure > 2] <- NA #Replacing outliers with NA
colSums(is.na(churn_data_df)) #Check NA values
churn_data_df$Yearly_equip_failure[is.na(churn_data_df$Yearly_equip_failure)] <- median(churn_data_df$Yearly_equip_failure, na.rm = TRUE)
colSums(is.na(churn_data_df)) #Check NA values
summary(churn_data_df$Yearly_equip_failure)#Median and mean stayed relatively the same



#Data Exploration of chosen variables
variables_subset <- select(churn_data_df, Area, Income, Marital, Gender, Churn, Contract,
                            PaymentMethod, Tenure, MonthlyCharge, Bandwidth_GB_Year)
str(variables_subset)



#Data Wrangling
#Transforming each category of Nominal variables Area, Marital, Gender, Churn, Contract, and Payment Method into numeric
all_dummies <- variables_subset %>%
  get_dummies()
glimpse(all_dummies) #Was view(all_dummies) - lowercase view() is an interactive-console convenience, not a portable function


#Removing Churn_No because there are only 2 categories for Churn and to prevent any complete separation/ collinearity
dummy_variables_subset <- select(all_dummies, Income, Tenure, MonthlyCharge, Bandwidth_GB_Year,
                                  Area_Rural, Area_Suburban, Area_Urban,
                                  Marital_Divorced, Marital_Married, `Marital_Never Married`,
                                  Marital_Separated, Marital_Widowed,
                                  Gender_Female, Gender_Male, Gender_Nonbinary,
                                  Churn_Yes,
                                  `Contract_Month-to-month`, `Contract_One year`, `Contract_Two Year`,
                                  `PaymentMethod_Bank Transfer(automatic)`, `PaymentMethod_Credit Card (automatic)`,
                                  `PaymentMethod_Electronic Check`, `PaymentMethod_Mailed Check`)
glimpse(dummy_variables_subset) #Was view(dummy_variables_subset)


#Scaling the dataset so all elements contribute equally by defining the function for min-max normalization (DataCamp: Supervised Learning in R: Classification)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

dummy_variables_subset$Income <- normalize(dummy_variables_subset$Income)
dummy_variables_subset$Tenure <- normalize(dummy_variables_subset$Tenure)
dummy_variables_subset$MonthlyCharge <- normalize(dummy_variables_subset$MonthlyCharge)
dummy_variables_subset$Bandwidth_GB_Year <- normalize(dummy_variables_subset$Bandwidth_GB_Year)
glimpse(dummy_variables_subset) #Was view(dummy_variables_subset)



# Exporting prepared dataset
prepared_dataset <- dummy_variables_subset
write.csv(prepared_dataset, "PreparedDatasetTask1.csv", row.names = FALSE) #Relative path (was a hardcoded local user path)



#Train and Test Split
set.seed(420)

trainIndex <- createDataPartition(dummy_variables_subset$Churn_Yes, times = 1, p = 0.80, list = FALSE)
train <- dummy_variables_subset[trainIndex, ]
glimpse(train) #Was view(train)
#Exporting training set
write.csv(train, "TrainDatasetTask1.csv", row.names = FALSE) #Relative path (was a hardcoded local user path)
test <- dummy_variables_subset[-trainIndex, ]
#Exporting test set
write.csv(test, "TestDatasetTask1.csv", row.names = FALSE) #Relative path (was a hardcoded local user path)
glimpse(test) #Was view(test)


#The recommended good initial value for k is the square root of the number of data points sqrt(8993) ~ 94.8
k <- sqrt(8993)
#Training and predicting using the best k rounded to the nearest whole number, k = 95
test_pred <- knn(
                  train = train,
                  test = test,
                  cl = train$Churn_Yes,
                  k= k
                  )



#Model Evaluation using confusion matrix
actual <- test$Churn_Yes

cm <- table(actual, test_pred)
cm

#Calculating the accuracy
TP = cm[2,2]
TN = cm[1]
FP = cm[1,2]
FN = cm[2,1]
accuracy = ((TP + TN) / (TP + FP + TN + FN))*100
sprintf("Accuracy is %.2f%%", accuracy)



#Calculating the AUC, Bobbitt 2021
model <- knn(
             train = train,
             test = test,
             cl = train$Churn_Yes,
             k=95,
             prob = TRUE
            )

roc(test$Churn_Yes, attributes(model)$prob)

#With an AUC of 0.93 which is very close to 1 we can confidently say our model performs better than a random guess.
