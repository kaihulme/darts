#include <include/utils.h>
#include <include/gaussianBlur.h>
#include <include/sobelEdges.h>
#include <include/houghCircles.h>
#include <include/circleDetector.h>
#include <include/argsHandler.h>

int main(int argc, char* argv[]) {

	// image from file
	cv::Mat img_input;
	string  image_name;

	// variables for arguments
	bool a_sobel; 
	bool a_hough; 
    bool a_m_threshold;
	bool a_gaussian;
	int  gaussian_size; 
	int  m_threshold;
	int  h_threshold;
	int  r_min; 
	int  r_max;
	int  r_step; 
	int  t_step;

	// get image, set argument variables and return -1 if invalid
	int pass = ArgsHandler(argc, argv, img_input, image_name,
                           a_sobel, a_hough, a_m_threshold, a_gaussian,
	                       gaussian_size, m_threshold, h_threshold,
	                       r_min, r_max, r_step, t_step);
	if (pass == -1) return -1;

	// keep original copy
	const string img_loc = "resources/" + image_name + ".png";
	cv::Mat img_original = imread(img_loc, 1);

	// apply gaussian filter
	if (a_gaussian) { 
		cv::Mat img_gaussian;
		GaussianBlur(img_input, gaussian_size, img_gaussian);
		NormaliseWrite(img_gaussian, image_name, "gaussian", 0);
		img_input = img_gaussian;
	}
	
	// sobel edge detector
    if (a_sobel) {
	
		// set kernel size and output matrices
        const int kernel_size = 3;
        cv::Mat img_dfdx, img_dfdy;
		cv::Mat img_magnitude, img_direction;
	
		// apply sobel edge detection
		SobelEdgeDetector(img_input, kernel_size, 
						  img_dfdx, img_dfdy,
						  img_magnitude, img_direction);

		// normalise and write output to images
		NormaliseWrite(img_dfdx, image_name, "dfdx", 0);
		NormaliseWrite(img_dfdy, image_name, "dfdy", 0);
		NormaliseWrite(img_magnitude, image_name, "magnitude", 0);
		
		// convert radians to degrees then write 
		cv::Mat img_direction_deg;
		RadToDeg(img_direction, img_direction_deg);
		NormaliseWrite(img_direction_deg, image_name, "direction", 0);
	
		// if thresholding
		if (a_m_threshold) {

			// threshold output image
			cv::Mat img_thresholded_magnitude;

			// threshold magnitude and write to file
			Threshold(img_magnitude, m_threshold,
					  img_thresholded_magnitude);
			NormaliseWrite(img_thresholded_magnitude, image_name, 
						  "thresholded_magnitude", m_threshold);

			// set magnitude image to thresholded version
			img_magnitude = img_thresholded_magnitude.clone();

		}

		// if hough circle transform
		if (a_hough) {

			// set number of radii to apply
			const int r_size = (r_max - r_min) / r_step + 1;

			// create vector of hough spaces
			std::vector<cv::Mat> hough_space;

			// perform hough circle transform on magnitudes
			GetHoughSpaceCircles(img_magnitude, r_size, r_min, r_max, 
					  			 r_step, t_step, hough_space);

			// // sum hough spaces and write
			cv::Mat hough_space_sum;
			SumHoughSpaceCircles(hough_space, hough_space_sum);		 
			NormaliseWrite(hough_space_sum, image_name, "hough_space_sum", 0);

			double sum_threshold = r_size * h_threshold * 0.6;

			// threshold sum
			cv::Mat thresholded_sum;
			Threshold(hough_space_sum, sum_threshold, thresholded_sum);
			NormaliseWrite(thresholded_sum, image_name, "hough_space_sum_thresholded", 0);

			std::vector<cv::Mat> thresholded_hough_space;
			ThresholdHoughSpace(hough_space, h_threshold,
								thresholded_hough_space);

			std::cout << "\nFinding circles..." << std::endl;

			// find circles
			std::vector<circle_t> circles = FindCircles(thresholded_hough_space, 
														thresholded_sum,
														r_size, r_min, r_step);

			// draw circles over image
			DrawCircles(img_original, circles, image_name);
			
		}
	
	}

	std::cout << "\nComplete!\n" << std::endl;
	return 0;
}