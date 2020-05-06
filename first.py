import numpy as np
from funcs import func, der_func, der_der_func, norm2
import math
from tabulate import tabulate


def first_order(f, df, x_0, eps, alpha):
    x = x_0
    iterations = 0
    x_next = x - alpha * df(x)
    while norm2(df(x)) ** 2 >= eps:
        x, x_next = x_next, x - alpha * df(x)
        iterations += 1
    return [x, iterations]


xs = [-5.785685835824241, -4.322697383792522, -2.240651381500233, -2.1931138685586804, -0.6799281866848323,
      0.6442180763909864, 2.302006495708799, 2.420223444760598, 4.388096788407111, 6.1256254032859285]
ys = [2.5591253949120842, 1.57416568843803, 0.5502991528690719, -0.46431826282763566, -0.14010337286254337,
      -1.012376059460888, -1.3241578701223444, -2.718572313201977, -2.4726769096483445, -3.662889903151375]


r = np.linalg.norm(der_der_func([-np.pi / 2, 0]), ord=2)
o_eps = 0.1
o_alpha = (1 - o_eps) / r

points = []
print()
for k in range(len(xs)):
    i = xs[k]
    j = ys[k]
    if -math.pi <= i + 2 * j <= 0:
        o_x_0 = np.array([i, j])
        res = first_order(func, der_func, o_x_0, o_eps, o_alpha)
        # print("[{:.3f}, {:.3f}]--[{:.8f}, {:.8f}]--{}".format(i, j, res[0][0], res[0][1], res[1]))
        points.append([i, j, res[1], res[0][0], res[0][1]])

print('eps: {}'.format(o_eps))
columns = ['x_0', 'y_0', 'iterations', 'x_opt', 'y_opt']
print(tabulate(points, headers=columns, floatfmt=".4f", tablefmt='github', showindex="always"))
