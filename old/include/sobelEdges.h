#include <string>
#include <iostream>

#include <stdio.h>
#include <opencv/cv.h>        
#include <opencv/cxcore.h>
#include <opencv/highgui.h>   

using namespace cv;

// apply sobel edge detection
void SobelEdgeDetector(const cv::Mat &input, 				// image to apply edge detection to
					   const int     size, 					// size of sobel kernel
					   cv::Mat       &dfdx_output, 			// horizontal gradients output
					   cv::Mat       &dfdy_output,			// vertical gradients output
					   cv::Mat       &magnitude_output, 	// magnitude of gradients output
					   cv::Mat       &direction_output);	// direction of gradients output