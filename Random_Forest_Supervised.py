from Classification_dataset import x_training, y_training, x_test, y_test
from Packages import RandomForestClassifier, metrics

# Building the model
model_classifier = RandomForestClassifier(n_estimators=100,
                                          criterion='gini',
                                          random_state=True)
# Fitting the model onto the datasets
model_fit = model_classifier.fit(x_training, y_training)

# Predicting the results
prediction = model_fit.predict(x_test)

# Figuring out the accuracy of the model
accuracy_of_model = metrics.accuracy_score(y_test, prediction)
print("Results of the model are the following:", accuracy_of_model)