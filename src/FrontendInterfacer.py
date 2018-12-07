# import numpy as np
#
# class FrontendInterfacer:
#
#     def __init__(self, num_leds):
#         """
#         Wraps Frontend display controllers. This class updates leds efficiently using numpy arrays
#         @param num_leds: total leds being controlled
#         @type num_leds: int
#         """
#         self._num_leds = num_leds
#         self._leds = np.zeros((num_leds, 4))  # the main numpy array containing led data
#
#     def set_leds(self, leds):
#         """
#         Setter for leds
#         @param leds: array of leds of the form [ [red, blue, green, brightness], ... ]
#         @type leds: np.array((self._num_leds, 4), dtype=int)
#         """
#         if leds.shape() != self._leds.shape():
#             raise ValueError("shape of /'leds/' param (%d, %d) must match shape (%d, %d)"
#                              % (leds.shape()[0], leds.shape[1], self._num_leds, 4))
#         self._leds = leds
#
#
#     def get_leds(self, leds):
#         """
#         Getter for leds
#         @return: array of leds of the form [ [red, blue, green, brightness], ... ]
#         @rtype: np.array((self._num_leds, 4), dtype=int)
#         """
#         return self._leds
#
#     def show(self):
#         """
#         Updates the front end display controller with the led data stored in self._leds
#         """
#         # TODO: call show method of display controller