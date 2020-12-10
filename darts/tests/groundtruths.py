def testrueboxes(name):
    """
    Get ground truth bounding boxes for test images 
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

# GROUND TRUTH BOUNDING BOXES
# 0	 [[426, 0, 193, 217]]
# 1	 [[168, 106, 246, 242]]
# 2	 [[90, 88, 100, 106]]
# 3	 [[314, 139, 82, 86]]
# 4	 [[155, 65, 260, 260]]
# 5	 [[412, 125, 143, 135]]
# 6	 [[205, 108, 73, 78]]
# 7	 [[235, 150, 183, 182]]
# 8	 [[65, 240, 70, 110],
# 	 [829, 201, 142, 208]]
# 9	 [[170, 13, 300, 300]]
# 10 [[78, 91, 117, 134]
# 	 [578, 118, 64, 101]
# 	 [913, 142, 40, 79]]
# 11 [[169, 93, 71, 96]]
# 12 [[153, 60, 71, 169]]
# 13 [[257, 103, 161, 164]]
# 14 [[103, 84, 157, 159]
# 	 [971, 84, 154, 241]]
# 15 [[129, 35, 171, 173]]
