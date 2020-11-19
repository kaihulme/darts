#include <string>
#include <fstream>

#include <stdio.h>
#include <opencv/cv.h>        
#include <opencv/cxcore.h> 
#include <opencv/highgui.h>   

#include <include/utils.h>

// handles command line arguments
int ArgsHandler(int     argc,               // no. of arguments
                char    *argv[],            // arguments
                cv::Mat &image,             // image
                string  &image_name,        // image name
                bool    &a_sobel,           // whether or not to apply sobel edge detection
                bool    &a_hough_transform, // whether or not to apply hough transform (circles)
                bool    &a_m_threshold,     // whether or not to apply magnitude thresholding
                bool    &a_gaussian,        // whether or not to apply gaussian smoothing
	            int     &gaussian_size,     // gaussian kernel size
                int     &m_threshold,       // threshold for sobel gradient magnitude
                int     &h_threshold,       // hough space threshold
	            int     &r_min,             // min radius for hough circles
                int     &r_max,             // max radius for hough circles
	            int     &r_step,            // radius stepping for hough circles
                int     &t_step);           // theta stepping for hough circles
                
// writes contents of args_help.txt
void ArgsHelper();