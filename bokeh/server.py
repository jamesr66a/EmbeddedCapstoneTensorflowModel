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

from bokeh.plotting import Figure
from bokeh.client import push_session
from bokeh.models import ColumnDataSource, HBox
from bokeh.plotting import figure, curdoc, show

p = Figure(plot_height=600, plot_width=800, title="")
source = ColumnDataSource(data=dict(x=[], y=[]))
p.circle(x="x", y="y", source=source)

plot_xs = []
plot_ys = []

def update():
  source.data = dict(x=plot_xs, y=plot_ys)

session = push_session(curdoc())

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
    plot_xs.append(prediction[0, 0])
    plot_ys.append(prediction[0, 0])
    update(None, None, None)

curdoc().add_root(HBox(None, p, width=1100))
curdoc().add_periodic_callback(update, 50)

api.add_resource(TFModelServer, '/vector')

import thread

def bokeh_thread():
  session.show()
  session.loop_until_closed()
 
try:
  thread.start_new_thread(bokeh_thread, ())
except:
  print "Unable to start bokeh thread"
  
app.run(debug=True, host='192.168.2.101')

