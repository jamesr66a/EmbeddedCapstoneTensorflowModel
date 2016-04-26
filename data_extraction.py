from generated.feature_pb2 import Feature
from generated.example_pb2 import Example

with open('apr25_1.csv') as f:
    lines = f.readlines()
    for line in lines:
        toks = line.split(',')
        features = zip(toks[3::2], toks[4::2])
        x = Example()
        x.features.feature['x'].float_list.value.append(float(toks[0]))
        x.features.feature['y'].float_list.value.append(float(toks[1]))
        x.features.feature['yaw'].float_list.value.append(float(toks[2]))
        for key, value in features:
          x.features.feature[key].int64_list.value.append(int(value))
        print x
        break
