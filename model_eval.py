import skflow
import sys

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

if len(sys.argv) > 1:
  X = []
  Y = []

  with open('apr25_1.csv') as f:
      lines = f.readlines()
      features = set()
      for line in lines:
          toks = line.split(',')
          for ft in toks[3::2]:
            if ft != '\n':
              features.add(ft)
                                         
      feat_list = list(features)

      X = []
      Y = []
      for line in lines:
        toks = line.split(',')
        x_vec = [-100]*len(features)
        for key, value in zip(toks[3::2], toks[4::2]):
          if key != '\n':
            x_vec[feat_list.index(key)] = int(value)
        X += [x_vec]
        if float(toks[0]) > 500:
          x = -float(toks[0])
        else:
          x = float(toks[0])
        y = float(toks[1])
        y_vec = [x, y]

        Y += [y_vec]

  X_np = np.array(X, dtype=np.int32)
  Y_np = np.array(Y, dtype=np.double)

  X_train, X_test, y_train, y_test = train_test_split(X_np, Y_np, test_size=0.2, random_state=42)

  regressor = skflow.TensorFlowDNNRegressor.restore(sys.argv[1])

  y_hat = regressor.predict(X_test)
  print 'mean absolute error: ', mean_absolute_error(y_hat, y_test)
  print 'r^2: ', r2_score(y_hat, y_test)

  residual = np.abs(y_hat - y_test)
  print 'stdev:', np.std(residual)

  import matplotlib.pyplot as plt

  plt.subplot(2, 1, 1)
  plt.rc('text', usetex=True)
  predicted = plt.scatter(y_hat[:,0], y_hat[:,1], c='b')
  actual = plt.scatter(y_test[:,0], y_test[:,1], c='r')
  plt.legend([predicted, actual], ['Predicted test set data points', 'Actual test set data points'], loc=4)
  plt.title('RSSI Location Predictions')
  plt.xlabel('X coordinate [cm]')
  plt.ylabel('Y coordinate [cm]')
  plt.subplot(2, 1, 2)
  vals = np.sqrt(np.power(residual[:,0], 2) + np.power(residual[:,0], 2)) 
  weights = np.ones_like(vals)/float(len(vals))
  plt.hist(vals, bins=40, weights=weights,normed=False)
  plt.title('Probability Histogram for RSSI Predicton Residual')
  plt.xlabel('$L_2$ residual value')
  plt.ylabel('Probability')
  plt.show()
