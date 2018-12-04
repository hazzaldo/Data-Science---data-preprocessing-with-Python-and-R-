# Data Preprocessing

#Importing the dataset
dataset = read.csv('Data.csv')

# Creating 'Matrix Of Features' and 'Dependant Variable Vector'.
# Unlike Python, in R langauge, we don't need to make a 
# a distinction between the two matrices (i.e. Matrix Of Features' and 'Dependant Variable Vector').

# Taking care of missing data
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