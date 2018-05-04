import numpy as np

class History:
    def __init__(self,config):
        history_length , screen_height, screen_width = config.history_length, config.screen_height, config.screen_width

        self.history = np.zeros([history_length, screen_height, screen_width], dtype = np.float32)

    def get(self):
      #  return self.history
        return np.transpose(self.history, (2,1,0))

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen
