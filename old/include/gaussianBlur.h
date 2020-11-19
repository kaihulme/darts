#include <string>
#include <iostream>

#include <stdio.h>
#include <opencv/cv.h>        
#include <opencv/cxcore.h> 
#include <opencv/highgui.h>   

using namespace cv;

// apply gaussian blur
void GaussianBlur(const cv::Mat &input, 			// input image to blur
				  const int 	size,				// gaussian kernel size
	              cv::Mat 		&output);			// blurred output