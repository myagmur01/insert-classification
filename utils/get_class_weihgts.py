
import numpy as np
from sklearn.utils import class_weight

def calculate_class_weights(trainY):

    """
    Loop over all classes and calculate the class weight
    :param trainY:
    :return: class_weights
    """

    # for i in range(0, len(classTotals)):
    # 	classWeight[i] = classTotals.max() / classTotals[i]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(trainY),
                                                      trainY)
    class_weights = {l: c for l, c in zip(np.unique(trainY), class_weights)}
    return class_weights