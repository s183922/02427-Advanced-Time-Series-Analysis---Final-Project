import numpy as np

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def measure(x_vals, y_vals, z_grid, p):
    x = p[0]
    y = p[1]
    i_x = np.searchsorted(x_vals, x)
    i_y = np.searchsorted(y_vals, y)

    points = (
        (
            x_vals[i_x - 1],
            y_vals[i_y - 1],
            z_grid[i_x - 1, i_y - 1]
        ),
        (
            x_vals[i_x],
            y_vals[i_y - 1],
            z_grid[i_x, i_y - 1]
        ),
        (
            x_vals[i_x - 1],
            y_vals[i_y],
            z_grid[i_x - 1, i_y]
        ),
        (
            x_vals[i_x],
            y_vals[i_y],
            z_grid[i_x, i_y]
        )
    )
    return bilinear_interpolation(x, y, points)