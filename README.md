# dart-detection

Face and dartboard detection using OpenCV and Python.

## Set up

Please run the provided bash script to set up the Python environment.

```bash
bash setup.sh
```

Once requirements have been install you can activate the environment.

```bash
. .env/bin/activate
```

## Running the program

An image can be processed by specifying its filename, e.g. `dart0.jpg`. Test images are located in `darts/resources/images/test`.

```bash
python -m darts dart0.jpg
```

A convenience script is provided to process all test images. Note this will take a while to complete.

```bash
bash alltests.sh
```

Outputs can be cleaned.

```bash
python -m darts clean
```

## Outputs

Output images are located in `darts/out/images`. Various images are outputted; the final detection with ground truth boudning boxes follows the naming:

```bash
dartX_true_pred_ensemble_dartboards.png
```

For detection results with KMeans colour segmentation see:

```bash
dartX_kmeans_true_pred_ensemble_dartboards.png
```

Output results are located in CSV files in `darts/out/results`.

## Repository structure

- `/darts`: main python module.
  - `/detection`: detection related modules.
    - `circledetector.py`: circle detection from Hough (circles) space.
    - `edgedetector.py`: sobel edge detection.
    - `ensembledetector.py`: detection of dartboards from ensemble of methods.
    - `linedetector.py`: lines detection from Hough (lines) space.
    - `violajones.py`: Viola-Jones detection with given cascade.
  - `/io`: input / output related modules.
    - `draw.py`: module for drawing on images.
    - `read.py`: module for reading from resources.
    - `write.py`: module for writing to output directory.
  - `/manipulation`: image manipulation related modules.
    - `convolution.py`: module for performing convolutions.
    - `gaussian.py`: module for performing gaussian blur.
  - `/out`: output directory.
    - `images`: output images.
    - `results`: output results in CSV files.
  - `/resources`: required resources.
    - `cascades`: Viola-Jones trained cascades.
    - `images`: images for training and testing
    - `materials`: given materials for project.
    - `opencv`: compiled OpenCV files.
    - `face.cpp`: C++ file for face detection using Viola-Jones cascade.
  - `/tests`: testing and evaluation scripts.
    - `evalutate.py`: evaluation of detections.
    - `groundtruths.py`: functions to get ground truth bounding boxes for testing.  
  - `/tools`: miscellaneous tooling modules.
    - `metrics.py`: calculation of performance metrics.
    - `utils.py`: miscellaneous utility functions.
  - `/transformation`: image transformation related modules.
    - `houghcircles.py`: performs Hough (circles) transform.
    - `houghlines.py`: performs Hough (lines) transform.
    - `segment.py`: performs KMeans color segmentation.
  - `__main__.py`: main script to run application.
  - `app.py`: application script for orchestration of processing.
- `/report`: LaTeX report and figures based on COLING2020 template.
- `/submission`: contains submission .zip
- `.gitignore`: Git ignore
- `alltests.sh`: convenience script to process all test images.
- `report.pdf`: final report PDF.
- `README.md`: this README file.
- `requirements.txt`: Python requirements to run Python module.
- `setup.sh`: convenience script to setup Python environment with requirements.

## Supporting technologies

- `OpenCV`: used for various image processing functionality
- `NumPy`: used for linear algebra and vectorisation of functions.
- `Pandas`: used for handling of tabular data and CSV files.
- `SciPy`: 
- `Scikit-Image`: used for an optimised implementation of 3D local maxima detection.
- `TQDM`: used to display progress of slower functions.

## Notes on Python implementation

As I have opted for a Python implementation of the project over C++ the processing time is a little slow as is the nature with interpreted vs. compiled languages. Where possible I have vectorised functions with NumPy so runtime is measured in minutes not hours but there is still a considerable wait. I have used the TQDM package to update progress of the slower functions, notably the Hough transforms.

There are some occurences in the project where I have used other package's implementations of functions in practice whilst having implemented the function myself. Again, I have done so in an effort to reduce processing time. 

Examples of this are using OpenCV's gausian blur function and Scikit-Image's implementation of 3D local maxima detection; both of these have been implemented in Python / NumPy and are functionally the same, but run slow so I opted to use other implementations to speed up testing.
