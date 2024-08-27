<<<<<<< HEAD
from braincog.base.brainarea import BrainArea
from braincog.model_zoo.base_module import BaseLinearModule, BaseModule


class PFC():
=======
import math
import random
import matplotlib
# matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from pygame.locals import *
import pandas as pd
import time
from braincog.base.brainarea import BrainArea

class PFC2():
>>>>>>> refs/remotes/origin/main
    """
    PFC
    """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x):
        """

        :return:x
        """

        return x

    def reset(self):
        """

        :return:x
        """

        pass

class dlPFC(BaseModule, PFC):
    """
    SNNLinear
    """
    def __init__(self,
                 step,
                 encode_type,
                 in_features:int,
                 out_features:int,
                 bias,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.fc = self._create_fc()
        self.c = self._rest_c()

    def _rest_c(self):
        c = torch.rand((self.out_features, self.in_features)) # eligibility trace
        return c

    def _create_fc(self):
        """
        the connection of the SNN linear
        @return: nn.Linear
        """
        fc = nn.Linear(in_features=self.in_features,
                  out_features=self.out_features, bias=self.bias)
        return fc



