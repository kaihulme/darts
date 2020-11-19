#include <include/circleDetector.h>

// returns circle positions and radii in image
std::vector<circle_t> FindCircles(std::vector<cv::Mat> &input,      // hough spaces for each radii
                                  cv::Mat &hough_space_sum,         // summed hough spaces
								  int r_size,                       // number of radii
                                  int min_r,                        // min radius
                                  int r_step) {                     // step size between radii
	
	// create vector of circles
	std::vector<circle_t> circles_output;

	// get circle centres from summed hough space 
	std::vector<pos_t> circle_locs = GetCircleLocs(hough_space_sum);

	// for each circle
	for (pos_t &circle_loc : circle_locs) {

		// best radius and position counter
		double max_r = 0.0;
		int    r_pos = 0;
		
		// find best radius
		int r = 0;
		for (cv::Mat &space : input) {

			// get current radius
			double curr_r = space.at<double>(circle_loc.y, circle_loc.x);

			// if best radius set as max
			if (curr_r > max_r) {
				max_r = curr_r;
				r_pos = r;
			}

			// increment radius position
			r++;

		}

		// set radius and create circle
		int circle_radius = min_r + r_step*r_pos;
		circle_t circle = {circle_loc, circle_radius};

		// add circle to vector and increment circle position
		circles_output.push_back(circle);

	}

	return circles_output;
}
	
// returns circle locations as local maxima in image
std::vector<pos_t> GetCircleLocs(cv::Mat input) {                    // hough space (circles)

	// create vector of circle locations and weighted locations
	std::vector<pos_t>   circle_locs;
	std::vector<w_pos_t> weighted_locs;

	// circle counter
	int c = 0;

	// for each pixel
	for (int y=0; y<input.rows; y++) {	
		for(int x=0; x<input.cols; x++) {
			
			// get current value
			double curr = input.at<double>(y, x);
			
			// ignore blank
			if (curr > 0) {
			
				// find local max then remove region
				int region_size = 25;
				pos_t circle_pos = LocalMax(input, x, y, region_size);
				
				// get weighted position of circle
				double weight = input.at<double>(y, x);
				w_pos_t weighted_loc = {circle_pos, weight};

				// add weighed location to vector and increment circle counter
				weighted_locs.push_back(weighted_loc);
				c++;
			
			}
			
		}
	}

	std::cout << "\nFound " << c << " circles" << std::endl;
	// REMOVE CLOSE CIRCLES USING WEIGHTS ???
		
	// add circle positions to vector
	for (auto loc : weighted_locs) circle_locs.push_back(loc.pos);
	
	return circle_locs;
}