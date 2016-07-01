class DqnParams(object):
    def __init__(self):
        self._gamma = None
        self._initial_epsilon = None
        self._final_epsilon = None
        self._replay_size = None
        self._batch_size = None

    @property
    def gamma(self):
        """Discount factor for target Q"""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property
    def initial_epsilon(self):
        """Starting value of epsilon"""
        return self._initial_epsilon

    @initial_epsilon.setter
    def initial_epsilon(self, value):
        self._initial_epsilon = value

    @property
    def final_epsilon(self):
        """Final value of epsilon"""
        return self._final_epsilon

    @final_epsilon.setter
    def final_epsilon(self, value):
        self._final_epsilon = value

    @property
    def replay_size(self):
        """Size of the experience replay buffer"""
        return self._replay_size

    @replay_size.setter
    def replay_size(self, value):
        self._replay_size = value

    @property
    def batch_size(self):
        """Size of minibatch"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    def __str__(self):
        return "%0.2f\t%0.2f\t%0.2f\t%06d\t%04d" % (self.gamma, self.initial_epsilon, self.final_epsilon, self.replay_size, self.batch_size)

    def random_initialization(self, seed=None):
        import random
        if seed != None:
            random.seed(seed)
        else:
            from datetime import datetime
            random.seed(datetime.now())

        self.gamma = round(random.random(), 2)
        self.initial_epsilon = round(random.random(), 2)
        self.final_epsilon = max(0, round(self.initial_epsilon - random.random()))
        self.replay_size = random.randint(1000, 100000)
        self.batch_size = 2 ** random.randint(1, 9)


