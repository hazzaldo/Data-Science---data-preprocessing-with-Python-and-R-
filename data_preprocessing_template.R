# Data Preprocessing

#Importing the dataset
dataset = read.csv('Data.csv')

# Creating 'Matrix Of Features' and 'Dependant Variable Vector'.
#===============================================================
# Unlike Python, in R langauge, we don't need to make a 
# a distinction between the two matrices (i.e. Matrix Of Features' and 'Dependant Variable Vector').

# Taking care of missing data
#==============================
# the $ sign allows us to reference the column in the matrix
# ifelse() function takes 1st param which is the condition, 2nd param the value if the 
# condition is true, and 3rd param is the value if the condition is false. 
# is.na is a function that checks if the value in the passed to the function is missing or not.
# so we're checking if all the values in the column Age is missing. So it returns true
# if a value in the column is missing, otherwise false if there's no missing value. 
# ave is the average function in R. First param is dataset$Age column. Second param is
# where we create a new function where in the function body we use the mean R function
# passing x (the matrix i.e. dataset$Age) and na.rm = TRUE means include the missing 
# value(s) when calculating the mean of the column.
# Third param is otherwise if false (no missing values) then return the column as it is.
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
# Take care of missing data for Salary column 
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Encoding categoral data 
#=========================
# In R we factors to encode category data. 
# the factor takes param 1 as the column, param 2 as all the values in the column,
# and param 3, the encode label for each value respectively. 
# Unlike Python, R does not associate 1,2,3 as some sort of order data. So Machine Learning
# models can treat them as encoded categories and not ordered data of ascending order.
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased, 
                         levels = c('No', 'Yes'),
                         labels = c(0, 1))

# Splitting the dataset into the Training set and Test set
#===========================================================
# the line below is how to install a new package in R that is not included in the existing
# list of packages. One installed, you can comment out the line as we won't need to 
# exectue it again.
# install.packages('caTools')
# the line below is how to include a library in your code. There's 2 ways. Either tick it
# from the list of packages in the Packages tab. Or you can use the line below instead.
library(caTools)
# the line below is the same as the random_set function in python.
set.seed(123)
# splitting the data into training and test set:
# Unlike in python, in R we only need to input the Y (or dependant variable) matrix to
# the split function.
# Unlike python, here in R the SplitRatio of dataset is aimed at the Training set (not the test set). 
# So in this case we specify 80% (0.8). 
# The return value will be either true or false for each entry (or observation). 
# True if the observation is chosen to go to the Training set and false if it's chosen
# for the Test set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# create the training set. As the split is already created with TRUE for training set,
# and FALSE for test set, we simply assign these 2 datasets to variable as per below.
training_set = subset(dataset, split == TRUE)
# create the test set
test_set = subset(dataset, split == FALSE)


# Feature Scaling 
# Note Country and Purchased columns are factored in R, which is not the same as
# one-hot-encoding applied to categorical data in python. Even though in the 
# dataset it shows numbers in those columns, but in fact a Factor in R
# is not a numerical data. Therefore to scale the dataset directly in R in our case
# will throw an error, because not all data are numerical. In this case we need
# to exclude categories from the feature scaling. Therefore we simply include 
# the columns with the numerical values (i.e. Age and Salary, and exclued Country and Purchased).
# We do this simply by specifying training_set[, 2:3], which is column 2 and 3. 
# Remember R index count start from 1 (unlike python which starts at 0). Same applies
# to the test_set.
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3]) 