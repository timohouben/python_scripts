#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import math
import random
import ternary


def generate_random_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator

    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j)] = random.random()
    return d


scale = 20
d = generate_random_heatmap_data(scale)
print(d)
figure, tax = ternary.figure(scale=scale)
tax.heatmap(d, style="h")
tax.boundary()
tax.set_title("Heatmap Test: Hexagonal")
tax.show()
