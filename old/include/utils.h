#include <string>
#include <iostream>

#include <stdio.h>
#include <opencv/cv.h>        
#include <opencv/cxcore.h>
#include <opencv/highgui.h>   

using namespace cv;

#ifndef UTILS_H
#define UTILS_H

// position type
struct pos_t {
    int x;          // row position
    int y;          // col position
};

// weighted position type
struct w_pos_t {
    pos_t  pos;     // postiion
    double weight;  // weight (accumulator)
};

// circle type
struct circle_t {
    pos_t pos;      // circle position
    int   radius;   // circle radius
};

#endif /* UTILS_H */

// convolve point (i,j) in matrix with kernel
double Convolution(cv::Mat   &input,                    // input matrix
                   cv::Mat   &kernel,                   // convolution kernel
                   int       i,                         // col in input to convolve
                   int       j,                         // row in input to convolve
                   const int r_i,                       // kernel horizontal radius
                   const int r_j);                      // kernel veritacal radius

// set value below threshold in matrix to 0
void Threshold(const cv::Mat &input,                    // input to threshold
               const double  threshold_val,             // threshold value
			   cv::Mat       &output);                   // thresholded output

// find max in specified region size around point
pos_t LocalMax(cv::Mat &input,                          // hough space
               int     i,                               // row for point to search around
               int     j,                               // col for point to serach around
			   int     region_size);                    // size of region to search for local max

// draw a set of circles on input image
void DrawCircles(cv::Mat               &input,          // image to draw over
                 std::vector<circle_t> &circles,        // poistion and radius of circles
                 std::string           image_name);     // image name

// normalise pixels between 0-255
void NormalisePixels(cv::Mat &input,                    // input to normalise
                     cv::Mat &output);                  // normalised output

// helper function to normalise then write image
void NormaliseWrite(cv::Mat input,                      // image to write
                    string  img_name,                   // image name
                    string  out_type,                   // image type (for naming)
                    int     arg);                       // extra argument (for naming)
                    
// convert matrix of radian angles to degrees
void RadToDeg(cv::Mat &input,                           // input matrix of radians
              cv::Mat &output);                         // output matrix of degrees