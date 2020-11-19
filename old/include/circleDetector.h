#include <string>
#include <iostream>

#include <stdio.h>
#include <opencv/cv.h>        
#include <opencv/cxcore.h>
#include <opencv/highgui.h> 

#include <include/utils.h>
#include <include/houghCircles.h>

using namespace cv;

// returns circle positions and radii in image
std::vector<circle_t> FindCircles(std::vector<cv::Mat> &input,      // hough spaces for each radii
                                  cv::Mat &hough_space_sum,         // summed hough spaces
								  int r_size,                       // number of radii
                                  int min_r,                        // min radius
                                  int r_step);                      // step size between radii
	
// returns circle locations as local maxima in image
std::vector<pos_t> GetCircleLocs(cv::Mat input);                    // hough space (circles)