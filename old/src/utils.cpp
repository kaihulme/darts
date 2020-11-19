#include <include/utils.h>

// convolve point (i,j) in matrix with kernel
double Convolution(cv::Mat   &input,                    // input matrix
                   cv::Mat   &kernel,                   // convolution kernel
                   int       i,                         // col in input to convolve
                   int       j,                         // row in input to convolve
                   const int r_i,                       // kernel horizontal radius
                   const int r_j) {                     // kernel veritacal radius

    // sum for matrix multiplication
    double result = 0.0;

    // for each kernel value
    for (int m=-r_i; m<=r_i; m++) {
		for (int n=-r_j; n<=r_j; n++ ) { 

            // correct image and kernel indices
            int x   = i + m + r_i;
            int y   = j + n + r_j;
            int k_x = m + r_i;
            int k_y = n + r_j;

            // get input and kernel values
            double input_val  = input.at<double>(y, x);
			double kernel_val = kernel.at<double>(k_y, k_x);

            // add product to current sum
            result += input_val * kernel_val;

        }
    }

    return result;
}

// set value below threshold in matrix to 0
void Threshold(const cv::Mat &input,                    // input to threshold
               const double  threshold_val,             // threshold value
			   cv::Mat       &output) {                  // thresholded output
	
	// intialise the output using the input
	output.create(input.size(), input.type());
	
	// for each pixel in input
	for (int y=0; y<input.rows; y++) {	
		for (int x=0; x<input.cols; x++) {

			// get pixel and set output pixel to 0
			double pixel = input.at<double>(y, x);
			double thresholded_pixel = 0;

			// if pixel exceeds threshold set to 255
			if (pixel > threshold_val) {
				thresholded_pixel = pixel;
			}

			// set value in output image
			output.at<double>(y, x) = thresholded_pixel;

		}
	}

}

// find max in specified region size around point
pos_t LocalMax(cv::Mat &input,                          // hough space
               int     i,                               // row for point to search around
               int     j,                               // col for point to serach around
			   int     region_size) {                    // size of region to search for local max

	// outofbounds checks
	int x_max = input.cols;
	int y_max = input.rows;

	// current max val / location
	double max 	   = 0;
	pos_t  max_pos = {0,0};

	// for a 2r*2r region around (i,j)
	for (int y=j-region_size; y<j+region_size; y++) {	
		for(int x=i-region_size; x<i+region_size; x++) {

			// check if in space
			if (x>=0 && x<x_max && y>=0 && y<y_max) {

				// get val at (i,j)
				double val = input.at<double>(y, x);

				// if highest set new max
				if (val > max) { 
					max     = val; 
					max_pos = {x,y};
				}

			}

			// remove region to reduce multiple detections
			input.at<double>(y, x) = 0;

		}
	}

	return max_pos;
}

// draw a set of circles on input image
void DrawCircles(cv::Mat               &input,          // image to draw over
                 std::vector<circle_t> &circles,        // poistion and radius of circles
                 std::string           image_name) {    // image name

	// for each circle
	for (auto circle : circles) {

		// get circle radius and centre
		int radius  = circle.radius;
		cv::Point point = cv::Point(circle.pos.x, circle.pos.y);		

		// draw circle and centre over original image
		cv::circle(input, point, radius, cvScalar(124, 200, 73), 2);
		cv::circle(input, point, 4, cvScalar(50, 50, 240), -1);

	}

	// string file_name = "/out/" + image_name + "_circles.jpg";
	// std::cout << "writing to " << file_name << std::endl;

	// write image with circles and centres
	imwrite("out/CIRCLES_coins1.jpg", input);


}

// normalise pixels between 0-255
void NormalisePixels(cv::Mat &input,                    // input to normalise
                     cv::Mat &output) {                 // normalised output

	// intialise the output using the input
	output.create(input.size(), cv::DataType<double>::type);

	// get min and max values in input
	double minVal, maxVal; 
	Point minLoc, maxLoc;
	minMaxLoc(input, &minVal, &maxVal, &minLoc, &maxLoc);

	// for each pixel in input
	for (int y=0; y<input.rows; y++) {	
		for(int x=0; x<input.cols; x++) {

			// get current pixel
			double pixel = input.at<double>(y, x);

			// normalise pixel between range 0-255
			double normalised_pixel = 255*((pixel-minVal) / (maxVal-minVal));
			output.at<double>(y, x) = normalised_pixel;

		}
	}

}

// helper function to normalise then write image
void NormaliseWrite(cv::Mat input,                      // image to write
                    string  img_name,                   // image name
                    string  out_type,                   // image type (for naming)
                    int     arg) {                      // extra argument (for naming)

	// normalised output image
	cv::Mat normalised_input;
	NormalisePixels(input, normalised_input);

	// create file name based of image input and output type
	string file_name = "out/" + img_name + "_" + out_type;
	file_name += (arg==0 ? ".jpg" : "_" + std::to_string(arg) + ".jpg");

	// write normalised image
	cv::imwrite(file_name, normalised_input);

} 

// convert matrix of radian angles to degrees
void RadToDeg(cv::Mat &input,                           // input matrix of radians
              cv::Mat &output) {                        // output matrix of degrees

	// intialise the output using the input
	output.create(input.size(), cv::DataType<double>::type);

	// convert each radian to degrees
	for (int y=0; y<input.rows; y++) {	
		for(int x=0; x<input.cols; x++) {
			double rad = input.at<double>(y, x);
			double deg = (rad >= 0 ? rad : (2*CV_PI + rad)) * 360 / (2*CV_PI);
			output.at<double>(y, x) = deg;
		}
	}

}