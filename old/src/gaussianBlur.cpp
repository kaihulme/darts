#include <include/gaussianBlur.h>
#include <include/utils.h>

void GaussianBlur(const cv::Mat &input, 			// input image to blur
				  const int 	size,				// gaussian kernel size
	              cv::Mat 		&output) {  		// blurred output
	
	// intialise the output using the input
	output.create(input.size(), cv::DataType<double>::type);
	
	// create the gaussian kernel
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);
	cv::Mat gaussian_kernel = kX * kY.t();
	
	// get radius of kernel for padding and convolution
	const int r_x = (size-1)/2;
	const int r_y = (size-1)/2;
	
	// replicate edge at border to allow edge convolutions
	cv::Mat padded_input;
	cv::copyMakeBorder(input, padded_input, 
					   r_x, r_x, r_y,r_y,
					   cv::BORDER_REPLICATE);
	
	// apply convolution to each pixel in image
	for (int y=0; y<input.rows; y++) {	
		for(int x=0; x<input.cols; x++) {
			// apply convolution to current pixel
            double gaussian_pixel = Convolution(padded_input, gaussian_kernel, x, y, r_x, r_y);
			// update the image with new pixel value
			output.at<double>(y, x) = gaussian_pixel;
		}
	}

	std::cout << "\nGaussian blur complete!" << std::endl;
}