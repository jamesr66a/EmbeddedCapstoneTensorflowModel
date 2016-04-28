import os
import skflow
from skflow.addons.config_addon import ConfigAddon
import sys

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

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
      y_vec = [float(i) for i in toks[0:2]]
      Y += [y_vec]

X_np = np.array(X, dtype=np.int32)
Y_np = np.array(Y, dtype=np.double)

X_train, X_test, y_train, y_test = train_test_split(X_np, Y_np, test_size=0.2, random_state=42)

if os.path.exists(sys.argv[1]):
  print sys.argv[1]
  regressor = skflow.TensorFlowDNNRegressor.restore(sys.argv[1])
else:
  regressor = skflow.TensorFlowDNNRegressor(
      hidden_units=[37, 37, 37],
      batch_size=1000,
      steps=1000,
      learning_rate=0.0005,
      continue_training=True,
      config_addon = ConfigAddon(num_cores=9))

if not os.path.exists(sys.argv[1]):
  os.makedirs(sys.argv[1])

with open(sys.argv[1] + "/logfile.txt", "a") as f:
  while True:
    regressor.fit(X_train, y_train, logdir='training')
    regressor.save(sys.argv[1])

    y_hat = regressor.predict(X_test)
    mae = mean_absolute_error(y_hat, y_test)
    r2 = r2_score(y_hat, y_test) 
    print 'mean absolute error: ', mean_absolute_error(y_hat, y_test)
    print 'r^2: ', r2_score(y_hat, y_test)
    f.write('{},{}\n'.format(mae, r2))
    f.flush()

