import skflow
import sys
import json

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

regressor = skflow.TensorFlowDNNRegressor.restore("../actual_fixed_corpus")

# Get feature vector

with open('../apr25_1.csv') as f:
    lines = f.readlines()
    features = set()
    for line in lines:
        toks = line.split(',')
        for ft in toks[3::2]:
          if ft != '\n':
            features.add(ft)

    feat_list = list(features)


from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

from bokeh.client import push_session
from bokeh.plotting import figure, curdoc, vplot

p = figure(x_range=(-1000, 1000), y_range=(-1000, 1000), toolbar_location=None)

r = p.scatter(x=[], y=[], radius=10, fill_color='#000000', fill_alpha=0.5)
ds = r.data_source

class TFModelServer(Resource):
  def put(self):
    json_data = request.get_json(force=True)
    x = json_data['x']
    y = json_data['y']
    yaw = json_data['yaw']
    features = json_data['features']

    print x, y, yaw,
    feat_vec = [-100]*len(feat_list)
    for i, feat_name in enumerate(feat_list):
      feat_vec[i] = features.get(feat_name, -100)

    prediction = regressor.predict(np.array([feat_vec]))
    print prediction
    ds.data['x'].append(prediction[0, 0])
    ds.data['y'].append(prediction[0, 1])
    ds.trigger('data', ds.data, ds.data)

curdoc().add_root(vplot(p))

import thread

session = push_session(curdoc())

def bokeh_thread():
  session.loop_until_closed()

thread.start_new_thread(bokeh_thread, ())
session.show()

api.add_resource(TFModelServer, '/vector') 
app.run(debug=True, host='192.168.2.101')

