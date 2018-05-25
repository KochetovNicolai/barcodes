import sys

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def rect_from_center(self, height, width):
        r = Rect()
        r.left = self.x - 0.5 * width
        r.right = self.x + 0.5 * width
        r.top = self.y - 0.5 * height
        r.bottom = self.y + 0.5 * height
        return r


class Rect:
    def __init__(self, left=0, top=0, right=0, bottom=0):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def dump(self, out=sys.stdout):
        out.write(reduce(lambda x, y : x + y, map(str, ('l=', self.left, ',t=', self.top, ',r=', self.right, ',b=', self.bottom, '\n'))))

    def copy(self):
        r = Rect()
        r.left = self.left
        r.right = self.right
        r.top = self.top
        r.bottom = self.bottom
        return r

    def left_top(self):
        return Point(self.left, self.top)

    def left_bottom(self):
        return Point(self.left, self.bottom)

    def right_top(self):
        return Point(self.right, self.top)

    def right_bottom(self):
        return Point(self.right, self.bottom)

    def contains(self, point):
        assert isinstance(point, Point)
        return self.left <= point.x and point.x < self.right and self.top <= point.y and point.y < self.bottom

    def move(self, dy, dx):
        self.left += dx
        self.right += dx
        self.top += dy
        self.bottom += dy

    def scale(self):
        return float(self.width()) / self.height()

    def stretch_h(self, scale):
        self.top *= scale
        self.bottom *= scale

    def stretch_w(self, scale):
        self.left *= scale
        self.right *= scale

    def stretch(self, scale_h, scale_w):
        self.stretch_h(scale_h)
        self.stretch_w(scale_w)

    def integerify(self):
        self.top = int(self.top)
        self.bottom = int(self.bottom)
        self.left = int(self.left)
        self.right = int(self.right)

    def height(self):
        return self.bottom - self.top

    def width(self):
        return self.right - self.left

    def area(self):
        return self.height() * self.width()

    def center(self):
        p = Point()
        p.y = self.top + self.height() * 0.5
        p.x = self.left + self.width() * 0.5
        return p

    def intersection(self, other):
        i = Rect()
        i.left = max(self.left, other.left)
        i.right = min(self.right, other.right)
        i.top = max(self.top, other.top)
        i.bottom = min(self.bottom, other.bottom)
        return i

    def union(self, other):
        o = Rect()
        o.left = min(self.left, other.left)
        o.right = max(self.right, other.right)
        o.top = min(self.top, other.top)
        o.bottom = max(self.bottom, other.bottom)
        return o

    def valid(self):
        return self.height() >= 0 and self.width() >= 0

    def __repr__(self):
        return Rect.__name__+repr((self.left, self.top, self.right, self.bottom))
