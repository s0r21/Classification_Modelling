from Packages import pd, np, RandomForestClassifier, metrics
from sklearn import datasets

full_iris_dataset = datasets.load_iris()
# print(full_iris_dataset)
    # Note: dataset has two arrays under "data" and "target"
# print(full_iris_dataset.feature_names, full_iris_dataset.data[0:7]) # Data = Exogenous variable
# print(full_iris_dataset.target_names, full_iris_dataset.target[0:7]) # Target = Endogenous variable

# NOTE: data is set up all in cm values.
# Transforming the dataset into a dataframe
exogenous_variables = pd.DataFrame({
    'sepal length (cm)': full_iris_dataset.data[:,0],
    'sepal width (cm)': full_iris_dataset.data[:,1],
    'petal length (cm)': full_iris_dataset.data[:,2],
    'petal width (cm)': full_iris_dataset.data[:,3],
})

endogenous_variables = pd.DataFrame({
    'species': full_iris_dataset.target
})

# Preparing the values for the model (training & test set)
training_set_value = 0.75 # percentage of the df you want in the training set
training_set_value_actual = round((len(exogenous_variables)*training_set_value)) # The actual number to stop at
x_training = exogenous_variables[0:training_set_value_actual]
y_training = endogenous_variables[0:training_set_value_actual]
x_test = exogenous_variables[training_set_value_actual:len(exogenous_variables)]
y_test = endogenous_variables[training_set_value_actual:len(endogenous_variables)]
