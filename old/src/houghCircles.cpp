#include <include/houghCircles.h>

// gets hough space circles for r radii
void GetHoughSpaceCircles(const cv::Mat 	   &input, 			// input image
						  int  				   r_size,			// number of radii
				    	  int  				   min_r, 			// min radius
						  int  				   max_r, 			// max radius
						  int  				   r_step,			// radius step size
						  int  				   t_step,			// theta step size
				     	  std::vector<cv::Mat> &output) {		// hough spaces for r radii

	// initalise hough_circles output
	cv::Mat hough_space;
	hough_space.create(input.size(), cv::DataType<double>::type);

	// set current radius to min radius
	int radius = min_r;

	// for each radii
	for (int r=0; r<r_size; r++) {

		// reset hough space for current radius
		for (int y=0; y<input.rows; y++) {	
			for(int x=0; x<input.cols; x++) {
				hough_space.at<double>(y, x) = 0.0;
			}
		}

		// for each pixel in image
		for (int y=0; y<input.rows; y++) {	
			for(int x=0; x<input.cols; x++) {

				// get gradient magnitude at (i,j)
				double pixel_val = input.at<double>(y, x);
				
				// if gradient magnitude is not 0
				if (pixel_val > 0) {
				
					// for circle around point
					for (int theta=0; theta<360; theta+=t_step){
				
						// get circle coordinates at and angle theta
						double x0 = x - (radius * cos(theta*CV_PI/180));
						double y0 = y - (radius * sin(theta*CV_PI/180));
				
						// increment value in hough space (unless out of image)
						if (x0>=0 && y0>=0 && x0<input.cols && y0<input.rows) {
							hough_space.at<double>(y0, x0)++;
						}

					}

				}

			}
		}

		// add hough space for r and increase radius
		output.push_back(hough_space.clone());		
		radius += r_step;

	}

	std::cout << "\nHough transform circles complete!" << std::endl;
}

// sums each hough space into one summed hough space
void SumHoughSpaceCircles(std::vector<cv::Mat> &input,			// hough spaces
						  cv::Mat 			   &output) {		// summed hough space

    // set summed space as 
    output.create(input[0].size(), cv::DataType<double>::type);

    // set hough space sum to 0
    for (int y=0; y<output.rows; y++) {	
        for(int x=0; x<output.cols; x++) {
            output.at<double>(y, x) = 0.0;
        }
    }

    // for each hough space add value to summed hough space
    for (auto space : input) {
        for (int y=0; y<output.rows; y++) {
            for (int x=0; x<output.cols; x++) {
                output.at<double>(y, x) += space.at<double>(y, x);
            }
        }
    }

}

// threshold each hough space
void ThresholdHoughSpace(std::vector<cv::Mat> &input, 			// hough spaces
					 	 int 				  threshold,		// threshold value
						 std::vector<cv::Mat> &output) {		// thresholded hough spaces


	// for each hough space / radii
	int input_size = input.size();
	for (int d=0; d<input_size; d++) {

		// threshold each hough space
		cv::Mat thresholded_input;
		Threshold(input[d], threshold, thresholded_input);
		output.push_back(thresholded_input);

	}	

}


