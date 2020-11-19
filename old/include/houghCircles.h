#include <string>
#include <iostream>

#include <stdio.h>
#include <opencv/cv.h>        
#include <opencv/cxcore.h>
#include <opencv/highgui.h>   

#include <include/utils.h>

using namespace cv;

// gets hough space circles for r radii
void GetHoughSpaceCircles(const cv::Mat 	   &input, 			// input image
						  int  				   r_size,			// number of radii
				    	  int  				   min_r, 			// min radius
						  int  				   max_r, 			// max radius
						  int  				   r_step,			// radius step size
						  int  				   t_step,			// theta step size
				     	  std::vector<cv::Mat> &output);		// hough spaces for r radii

// sums each hough space into one summed hough space
void SumHoughSpaceCircles(std::vector<cv::Mat> &input,			// hough spaces
						  cv::Mat 			   &output);		// summed hough space

// threshold each hough space
void ThresholdHoughSpace(std::vector<cv::Mat> &input, 			// hough spaces
					 	 int 				  threshold,		// threshold value
						 std::vector<cv::Mat> &output);			// thresholded hough spaces