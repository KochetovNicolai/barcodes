import sys

class Point:
    def __init__(self):
        self.x = 0
        self.y = 0

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

    def stretch_h(self, scale):
        self.top *= scale
        self.bottom *= scale

    def stretch_w(self, scale):
        self.left *= scale
        self.right *= scale

    def stretch(self, scale_h, scale_w):
        self.stretch_h(scale_h)
        self.stretch_w(scale_w)

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
