#include <include/argsHandler.h>

// handles command line arguments
int ArgsHandler(int     argc,               // no. of arguments
                char    *argv[],            // arguments
                cv::Mat &image,             // image
                string  &image_name,        // image name
                bool    &a_sobel,           // whether or not to apply sobel edge detection
                bool    &a_hough,           // whether or not to apply hough transform (circles)
                bool    &a_m_threshold,     // whether or not to apply magnitude thresholding
                bool    &a_gaussian,        // whether or not to apply gaussian smoothing
	            int     &gaussian_size,     // gaussian kernel size
                int     &m_threshold,       // threshold for sobel gradient magnitude
                int     &h_threshold,       // hough space threshold
	            int     &r_min,             // min radius for hough circles
                int     &r_max,             // max radius for hough circles
	            int     &r_step,            // radius stepping for hough circles
                int     &t_step) {          // theta stepping for hough circles

    // set default values
    a_sobel           = false; 
	a_hough           = false;
    a_m_threshold     = false; 
	a_gaussian        = false;
	gaussian_size     = 20; 
	m_threshold       = 30;
	r_min             = 30; 
	r_max             = 50; 
    r_step            = 5; 
	t_step            = 20;
	h_threshold       = 10;

    // check arg count
	if (argc < 2)  { 
        printf("\nError: image not specified!\n\n");
        return -1; 
    }
	
    // if ? show args help
	if (!strcmp(argv[1], "?")) {
        ArgsHelper(); 
        return -1;
    }

    // if only image argument specified
	else if (argc == 2) {
        printf("\nError: operation not specified!\n\n"); 
        return -1;
    }

	// get image location
	image_name = argv[1];
	const string img_loc = "resources/" + image_name + ".png";
	
	// read image data
 	const Mat image_read = imread(img_loc, 1);
 	if(!image_read.data ) {
   		printf("\nError: image not found!\n\n");
   		return -1;
 	} 

	// convert to grey double Matrix
 	cv::Mat img_grey;
 	cvtColor(image_read, img_grey, CV_BGR2GRAY);
	img_grey.convertTo(image, cv::DataType<double>::type);
	
	// handle flags and set bools
	for (int i=2; i<argc; i++) {

		// if -s apply sobel edge detection
		if (!strcmp(argv[i], "-s") && !a_sobel) {
			try { m_threshold = std::stoi(argv[i+1]); a_m_threshold=true; i++; }
			catch (std::exception const &e) {}
			a_sobel = true;
		}
			//sobel = true;

		// if -h [x][y][z] apply hough transform circles with radii x->y step z
		else if (!strcmp(argv[i], "-h") && !a_hough) { 
			try {
				r_min  		= std::stoi(argv[i+1]); i++;
				r_max  		= std::stoi(argv[i+1]); i++;
				r_step 		= std::stoi(argv[i+1]); i++;
				t_step 		= std::stoi(argv[i+1]); i++;
				h_threshold = std::stoi(argv[i+1]); i++;
			}
			catch (std::exception const &e) {}
			a_hough = true;
		}

		// if -g [x] apply gaussian blur with specified kernel size
		else if (!strcmp(argv[i], "-g") && !a_gaussian)  {
			try { gaussian_size = std::stoi(argv[i+1]); i++; }
			catch (std::exception const &e) {}
			a_gaussian = true;
		}
		
		// unrecognised argument
		else { 
			std::cout << "\nError: check your flags! ('?' for help)\n" << std::endl; 
			return -1; }

	}

    // cannot calculate hough space without magnitudes from sobel edge detection
	if (a_hough && !a_sobel) { 
        std::cout << "\nError: hough transform requires sobel edge detection (-s)!\n" << std::endl;     
        return -1; 
    }

    // return pass
    return 1;
}

// writes contents of args_help.txt
void ArgsHelper() {

    // open file
    string file_name = "args_help.txt";
    std::ifstream f(file_name);

    // output file contents
    std::cout<<"\n"<<std::endl;
    if (f.is_open()) std::cout << f.rdbuf();
    std::cout<<"\n"<<std::endl;

}