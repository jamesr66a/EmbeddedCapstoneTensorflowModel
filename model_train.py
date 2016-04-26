import pandas
import skflow
import sys

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

data = pandas.read_csv('out.txt')

y, X = data['x'], data[data.columns[2:-1]].fillna(-100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if len(sys.argv) > 1:
  print sys.argv[1]
  regressor = skflow.TensorFlowDNNRegressor.restore(sys.argv[1])
else:
  regressor = skflow.TensorFlowDNNRegressor(
      hidden_units=[50, 50, 50],
      batch_size=128,
      steps=500000,
      learning_rate=0.005)
  try:
    regressor.fit(X_train, y_train, logdir='training')
  except KeyboardInterrupt:
    pass

y_hat = regressor.predict(X_test)
print 'mean absolute error: ', mean_absolute_error(y_hat, y_test)
print 'r^2: ', r2_score(y_hat, y_test)

regressor.save('3hidden50_gaussian3')
