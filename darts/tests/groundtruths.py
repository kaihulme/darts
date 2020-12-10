def gettruedartboards(name):
    """
    Get ground truth bounding boxes for dartboards in test images
    based on name. Returns false for images not in test set.
    """
    if (name == "dart0"):
        return [[426, 0, 193, 217]]
    elif (name == "dart1"):
        return [[168, 106, 246, 242]]
    elif (name == "dart2"):
        return [[90, 88, 100, 106]]
    elif (name == "dart3"):
        return [[314, 139, 82, 86]]
    elif (name == "dart4"):
        return [[155, 65, 260, 260]]
    elif (name == "dart5"):
        return [[412, 125, 143, 135]]
    elif (name == "dart6"):
        return [[205, 108, 73, 78]]
    elif (name == "dart7"):
        return [[235, 150, 183, 182]]
    elif (name == "dart8"):
        return [[65, 240, 70, 110],
	            [829, 201, 142, 208]]
    elif (name == "dart9"):
        return [[170, 13, 300, 300]]
    elif (name == "dart10"):
        return [[78, 91, 117, 134]
	            [578, 118, 64, 101]
	            [913, 142, 40, 79]]
    elif (name == "dart11"):
        return [[169, 93, 71, 96]]
    elif (name == "dart12"):
        return [[153, 60, 71, 169]]
    elif (name == "dart13"):
        return [[257, 103, 161, 164]]
    elif (name == "dart14"):
        return [[103, 84, 157, 159]
	            [971, 84, 154, 241]]
    elif (name == "dart15"):
        return [[129, 35, 171, 173]]
    return False


def gettruefaces(name):    
    """
    Get ground truth bounding boxes for dartboards in test images
    based on name. Returns false for images not in test set.
    """
    if (name == "dart0"):
        return[[191, 198, 262, 292]]
    elif (name == "dart1"):
        return[[]]
    elif (name == "dart2"):
        return[[]]
    elif (name == "dart3"):
        return[[]]
    elif (name == "dart4"):
        return[[343, 100, 474, 274]]
    elif (name == "dart5"):
        return[[ 64, 132, 124, 210],
               [ 57, 240, 118, 321],
               [191, 204, 253, 282],
               [250, 153, 305, 234],
               [294, 232, 348, 310],
               [374, 173, 445, 247],
               [434, 227, 485, 300],
               [516, 170, 573, 236],
               [564, 236, 619, 315],
               [649, 181, 714, 240],
               [681, 243, 729, 310]]
    elif (name == "dart6"):
        return[[290, 113, 325, 158]]
    elif (name == "dart7"):
        return[[346, 183, 429, 284]]
    elif (name == "dart8"):
        return[[186, 294, 237, 344],
               [607, 254, 652, 316]]
    elif (name == "dart9"):
        return[[ 94, 195, 199, 341]]
    elif (name == "dart10"):
        return[[]]
    elif (name == "dart11"):
        return[[327,  72, 380, 147]],
    elif (name == "dart12"):
        return[[]],
    elif (name == "dart13"):
        return[[414, 109, 519, 268]],
    elif (name == "dart14"):
        return[[464, 204, 548, 329],
               [719, 180, 836, 289]],
    elif (name == "dart15"):
        return[[ 68, 128, 124, 212],
               [544, 125, 597, 210],
               [376, 104, 418, 185]]
    return False
