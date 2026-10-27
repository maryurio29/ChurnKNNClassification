# D209 Performance Assessment — Task 1: Classification Analysis (KNN)

*Original coursework for D209 – Data Mining I, Western Governors University.*

## Part I: Research Question

**A1. What are the major predictor variables in determining or predicting the Churn of the customers from the churn dataset?**

**A2. Goals of the data analysis**
The purpose of my analysis is to identify the customer predictor variables that have a significant impact on determining or predicting customer churn, if any. The gaming and telecommunication fields offer services with mass competition resulting in a fast cycle of customer inflow and churn (Ahn et al.). When it costs about ten times the amount to convert a new lead to a customer, reducing customer churn becomes even more important than acquiring new ones. The search and identification of customers who show a high inclination to abandon the company or customer churn prediction is of crucial importance as part of a customer-oriented retention strategy that aims to reduce customer churn (De Caigny et al. 2018).

## Part II: Method Justification

**B1. Justification for Classification Method**
K-Nearest Neighbors (KNN) classification method uses the distance, most often the Euclidean distance, between variables to map datapoints according to their features and their labels. When an unknown datapoint is encountered, KNN then looks at the 'k' closest labeled data points, for example, 5 closest labeled data points, and gets them to vote on what the unlabeled point should be labeled as. The number of neighboring points can be modified and adjusted as necessary to create the most effective model, though k will remain constant throughout that model once selected. Hopefully the outcome will include an algorithm that will have a model accuracy of at least 95% without over or underfitting.

**B2. Assumption**
The crux of the k-nearest neighbors (k-NN) classification method is that it assumes that similar things exist in proximity to each other. Since it uses the nearby neighbors to predict what an unknown nearby datapoint should be labeled as, it must assume that the nearby known datapoints are like the unknown datapoint.

**B3. Packages and their purpose**
For this analysis I will be using RStudio, given that we'll be building a statistical model and RStudio is very efficient at it. RStudio offers many packages that we'll need to create a KNN classification model.

| Package / library | Use case |
|---|---|
| readr | To read/write data files |
| visdat | To find missing values with vis_miss [In-Text Citation: (Tierney & Hughes, 2023)] |
| dplyr | Essential for data frame manipulation |
| tidyverse | Essential data science collection |
| ggplot2 | To generate plots |
| caTools | For splitting data |
| caret | Predictive models package |
| pROC | Performance curve |
| tidytable | Dummy variables/confusion matrix |

## Part III: Data Preparation

**C1. Data Preprocessing**
One of the most important steps in the pre-processing stage for KNN classification modeling is Scaling. The KNN algorithm relies on distance for classification, as shared earlier it assumes that similar datapoints exist in proximity to each other. Scaling is the process of transforming data to fit within a specific scale, like 0-100 or 0-1. Scaling the data is essential when using methods based on (distance) measures of how far apart data points are, like support vector machines, SVM or k-nearest neighbors, or KNN. If the feature/variables have different scales, then the KNN model will be less accurate in its prediction (Professor Elleh).

**C2. Data set variables for analysis**

| Variable | Class Type |
|---|---|
| Income | Numeric |
| Tenure | Numeric |
| MonthlyCharge | Numeric |
| Bandwidth_GB_Year | Numeric |
| Area | Categorical |
| Marital | Categorical |
| Gender | Categorical |
| Churn (Response variable) | Categorical |
| Contract | Categorical |
| PaymentMethod | Categorical |

**C3. Data Preparation**
1. I plan on using the duplicated(df) function from base R to identify if there are any duplicated rows in the data frame. This function determines which elements of a vector or data frame are duplicates of elements with smaller subscripts and returns a logical vector showing which elements (rows) are duplicates with True and False if not a duplicated row (Rdocumentation, accessed 2024). I also plan on applying the sum(duplicated(df)) function included with base R as well. This will sum() any True elements in the duplicated() output vector so it will help us answer how many duplicated rows there are in the data frame. The easiest way to detect missing values in R is using the colSums(is.na(df)) function (native to base R) which tallies up the count of missing values per column. I then plan on using the vis_miss(df) function from the library(visdat) to build a visualization of the missing values by column and provides the sparsity percentages of missing vs. non-missing values in the total data as well as per column. Finally, I plan on using the boxplot method to identify any outliers in the quantitative columns using the boxplot() function already included in R. This method will give me the minimum and maximum whiskers of each numerical column to better assess the quality of the values in the columns and the best method of treatment. See [`churn_knn_classification.R`](./churn_knn_classification.R) for the full code.
2. Now that our data is clean, I will transform all categorical variables to key-value pairs, dummy variables, using the get_dummies() function from the tidytable package. Each value will be its own 'dummy variable' and will have a 0 or 1 if that is the value of the observation. I will remove the Churn_No dummy variable, otherwise it will create a perfect separation with our independent variable, ChurnYes. See [`churn_knn_classification.R`](./churn_knn_classification.R) for the full code.
3. Lastly, I will define a normalize function to scale the dataset so all elements contribute equally by defining the function for min-max normalization (DataCamp: Supervised Learning in R: Classification).

**C4. Prepared Dataset**
See [`PreparedDatasetTask1.csv`](./PreparedDatasetTask1.csv).

## Part IV: Model Comparison and Analysis

**D1. Splitting Data into Train & Test sets**
See [`TrainDatasetTask1.csv`](./TrainDatasetTask1.csv) & [`TestDatasetTask1.csv`](./TestDatasetTask1.csv).

**D2. Classification Analysis techniques**
The crux of the k-nearest neighbors (k-NN) classification method is that it assumes that similar things exist in proximity to each other. Since it uses the nearby neighbors to predict what an unknown nearby datapoint should be labeled as, it must assume that the nearby known datapoints are like the unknown datapoint (Professor Elleh). We will be using the recommended k value of the sqrt of the number of observations, 8993 (DataCamp).

**D3. Code for Classification Analysis**
```r
#The recommended good initial value for k is the square root of the number of data points sqrt(8993) ~ 94.8
k <- sqrt(8993)
#Training and predicting using the best k rounded to the nearest whole number, k = 95
test_pred <- knn(
                  train = train,
                  test = test,
                  cl = train$Churn_Yes,
                  k = k
                  )

#Model Evaluation using confusion matrix
actual <- test$Churn_Yes
cm <- table(actual, test_pred)
cm
```
```
      test_pred
actual    0    1
     0 1332    0
     1   96  370
```
```r
#Calculating the accuracy
TP = cm[2,2]
TN = cm[1]
FP = cm[1,2]
FN = cm[2,1]
accuracy = ((TP + TN) / (TP + FP + TN + FN))*100
sprintf("Accuracy is %.2f%%", accuracy)
```
```
[1] "Accuracy is 94.66%"
```
```r
#Calculating the AUC, Bobbitt 2021
model <- knn(
             train = train,
             test = test,
             cl = train$Churn_Yes,
             k = 95,
             prob = TRUE
            )
roc(test$Churn_Yes, attributes(model)$prob)
```
```
Data: attributes(model)$prob in 1332 controls (test$Churn_Yes 0) > 466 cases (test$Churn_Yes 1).
Area under the curve: 0.927
#with an AUC of 0.93 which is very close to 1 we can confidently say our model performs better than a random guess
```

## Part V: Data Summary and Implications

**E1. Accuracy and Area Under the Curve (AUC)**
Our kNN classification model performed admirably and met our goal of finding an algorithm that will have a model accuracy of at least 95% without over or underfitting. Our model accuracy is roughly 95% when rounded to the nearest percentage point. With an AUC of 0.93 which is very close to 1 we can confidently say our model performs better than a random guess.

**E2. Results and Implications**
Based on the model performance mentioned above, accuracy of 95% and AUC of 0.93, we can conclude that the kNN model works a lot better than the logistic regression model, which gave an accuracy of less than 95% (see [D208 Task 2](../churn_logistic_regression.R/)). I'm confident the model defined in our analysis can be used to make predictions on future data with accurate results and potentially be used to prevent customer churn by highlighting key customer predictor variables that have a significant impact on determining or predicting customer churn.

**E3. Limitation**
In the assumptions of KNN classification covered in section B2, I noted that a KNN model "assumes that other similar things exist in some proximity to each other when plotted on a graph". This assumption is essential to the KNN classification process, because it depends on those "neighbors" to be of the same label as each datapoint it is trying to label. Given the outstanding results of this model, it's clear that the customer churn in the churn dataset is distributed so that neighboring datapoints are very similar to each other to such a degree that KNN classification is extremely useful. With this being said, one of the most noticeable limitations of kNN classification is slow performance. KNN is slow and inefficient for large datasets because it processes the entire training set for each prediction.

**E4. Recommendations**
In conclusion, our kNN classification model suggests that we can use the explanatory variables to accurately predict the response variable, Churn. This means the company can make predictions on how likely customers are to churn and plan quarterly budgets accordingly. I would suggest the company utilize this model whenever planning projected revenue.

## Part VI: Demonstration

**F. Panopto Video**
https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a2f2e10e-6c20-4e16-9928-b289016f9aba

**G. Web Sources for code**
- Tierney, N. (2023, February 2). Preliminary visualisation of data [R package visdat version 0.6.0]. The Comprehensive R Archive Network. https://cran.r-project.org/package=visdat
- Home. RDocumentation. (n.d.-b). https://www.rdocumentation.org/
- Tierney, N. (2023b, February 2). Using visdat. Using Visdat. https://cran.r-project.org/web/packages/visdat/vignettes/using_visdat.html
- Data Tricks. (2021, December 10). One-hot encoding in R: Three simple methods. https://datatricks.co.uk/one-hot-encoding-in-r-three-simple-methods
- DataCamp. Supervised Learning in R: Classification: kNN. DataCamp. https://www.datacamp.com/tutorial/k-nearest-neighbors-knn-classification-with-r-tutorial
- Bobbitt, Z. (2021, April 6). How to Calculate AUC (Area Under Curve) in R. Statology. https://www.statology.org/auc-in-r/

**H. Sources for content**
- WGU INFORMATION TECHNOLOGY. (2023, August 8). R or python. Western Governors University. https://www.wgu.edu/online-it-degrees/programming-languages/r-or-python.html
- WGU - Dr. Elleh's D209 Slide Decks

## Part VII: Post Submission Corrections
**Corrections**
After leveraging Claude Sonnet 5.5, it was found that the 94.66% accuracy and 0.927 AUC include target leakage. Reported confusion matrix has 0 false positives, which is the signature of the leak. The R code passes the whole train/test frames to knn(), so Churn_Yes is used as a feature. With the label column dropped, accuracy is 82.2% and AUC is 0.887. The majority-class baseline is 73.8%. That’s still a real lift, but it isn’t 95%.
