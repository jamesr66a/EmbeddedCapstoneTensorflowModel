from numpy.random import random

from bokeh.plotting import figure, show, output_file

def mscatter(p, x, y, typestr):
  p.scatter(x, y, marker=typestr, line_color="#6666ee", fill_color="#ee6666", fill_alpha=0.5, size=12)

def mtext(p, x, y, textstr):
  p.text(x, y, text=[textstr], text_color="#449944", text_align="center", text_font_size="10pt")

output_file("markers.html")

p = figure(title="markers.py example")

N = 10

mscatter(p, random(N)+2, random(N)+1, "circle")
mscatter(p, random(N)+4, random(N)+1, "square")

show(p)
