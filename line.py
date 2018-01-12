import numpy as np
from collections import deque


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # # was the line detected in the last iteration?
        # self.detected = False
        # x value of the last fit of the line
        self.xfit = [np.array([False])]
        # x values of the last n fits of the line
        self.last_n = 10
        self.recent_xfitted = deque(maxlen=self.last_n)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.recent_polyfit = deque(maxlen=self.last_n)
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # # distance in meters of vehicle center from the line
        # self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # # x values for detected line pixels
        # self.allx = None
        # # y values for detected line pixelsnp.
        # self.ally = None

    def set_xfitted(self, xfit):
        self.xfit = xfit
        self.recent_xfitted.append(xfit)
        self.bestx = np.mean(self.recent_xfitted, axis=0)

    def set_polyfit(self, fit):
        if self.fit is not None:
            self.diffs = self.fit - fit
        self.fit = fit
        self.recent_polyfit.append(fit)
        self.best_fit = np.mean(self.recent_polyfit, axis=0)
