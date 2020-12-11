import numpy as np

def gettruedartboards(name):
    """
    Get ground truth bounding boxes for dartboards in test images
    based on name. Returns false for images not in test set.
    """
    if (name == "dart0"):
        return np.asarray([[426, 0, 193, 217]])
    elif (name == "dart1"):
        return np.asarray([[168, 106, 246, 242]])
    elif (name == "dart2"):
        return np.asarray([[90, 88, 100, 106]])
    elif (name == "dart3"):
        return np.asarray([[314, 139, 82, 86]])
    elif (name == "dart4"):
        return np.asarray([[155, 65, 260, 260]])
    elif (name == "dart5"):
        return np.asarray([[412, 125, 143, 135]])
    elif (name == "dart6"):
        return np.asarray([[205, 108, 73, 78]])
    elif (name == "dart7"):
        return np.asarray([[235, 150, 183, 182]])
    elif (name == "dart8"):
        return np.asarray([[65, 240, 70, 110],
	                       [829, 201, 142, 208]])
    elif (name == "dart9"):
        return np.asarray([[170, 13, 300, 300]])
    elif (name == "dart10"):
        return np.asarray([[78, 91, 117, 134]
	                       [578, 118, 64, 101]
	                       [913, 142, 40, 79]])
    elif (name == "dart11"):
        return np.asarray([[169, 93, 71, 96]])
    elif (name == "dart12"):
        return np.asarray([[153, 60, 71, 169]])
    elif (name == "dart13"):
        return np.asarray([[257, 103, 161, 164]])
    elif (name == "dart14"):
        return np.asarray([[103, 84, 157, 159],
	                       [971, 84, 154, 241]])
    elif (name == "dart15"):
        return np.asarray([[129, 35, 171, 173]])
    return False


def gettruefaces(name):    
    """
    Get ground truth bounding boxes for dartboards in test images
    based on name. Returns false for images not in test set.
    """
    if (name == "dart0"):
        return np.asarray([[191, 198, 71, 94]])
    elif (name == "dart1"):
        return np.asarray([])
    elif (name == "dart2"):
        return np.asarray([])
    elif (name == "dart3"):
        return np.asarray([])
    elif (name == "dart4"):
        return np.asarray([[343, 100, 131, 174]])
    elif (name == "dart5"):
        return np.asarray([[ 64, 132, 60, 78],
                           [ 57, 240, 61, 81],
                           [191, 204, 62, 78],
                           [250, 153, 55, 81],
                           [294, 232, 54, 78],
                           [374, 173, 71, 74],
                           [434, 227, 51, 74],
                           [516, 170, 57, 66],
                           [564, 236, 55, 79],
                           [649, 181, 65, 59],
                           [681, 243, 48, 67]])
    elif (name == "dart6"):
        return np.asarray([[290, 113, 35, 45]])
    elif (name == "dart7"):
        return np.asarray([[346, 183, 84, 101]])
    elif (name == "dart8"):
        return np.asarray([[186, 294, 51, 50],
                           [607, 254, 45, 62]])
    elif (name == "dart9"):
        return np.asarray([[ 94, 195, 105, 146]])
    elif (name == "dart10"):
        return np.asarray([])
    elif (name == "dart11"):
        return np.asarray([[327,  72, 53, 75]])
    elif (name == "dart12"):
        return np.asarray([])
    elif (name == "dart13"):
        return np.asarray([[414, 109, 105, 159]])
    elif (name == "dart14"):
        return np.asarray([[464, 204, 84, 125],
                           [719, 180, 117, 109]])
    elif (name == "dart15"):
        return np.asarray([[ 68, 128, 56, 84],
                           [544, 125, 53, 85],
                           [376, 104, 42, 81]])
    return False
