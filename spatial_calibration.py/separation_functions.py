
import math





class Sep:

    def f(self,x):
        pass

class ExpSep(Sep):


    def __init__(self, min_sep, max_sep):

        self.min_sep = min_sep
        self.max_sep = max_sep


    def f(self, x):

        if x <= 0:
            print("[ERROR] function undefined")
            return None

        theta = math.exp(-x - 1)
        return self.max_sep*theta + (1-theta)*self.min_sep


class PieceLinSep(Sep):

    def __init__(self,min_sep,max_sep, bound):

        self.min_sep = min_sep
        self.max_sep = max_sep
        self.bound = bound
        self.slope = (self.min_sep - self.max_sep)/self.bound

    def f(self,x):

        if x <= 0:
            print("[ERROR] function undefined")
            return None
        elif x <= self.bound + 1:

            return self.slope * (x - 1) + self.max_sep
        else:
            return self.min_sep
