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
      y_vec = [float(i) for i in toks[0:3]]
      Y += [y_vec]

with open('outfile', 'w') as f:
  for row in X:
    for col in row:
      f.write('%s,' % col)
    f.write('\n')
