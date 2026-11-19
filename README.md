Chromatic Aberration-Based Image Forgery Detection

Python-based image processing project | 2020

1. Overview

	This project explores the detection of digital image manipulation using chromatic aberration, a physical phenomenon caused by the different refraction of light wavelengths through a camera lens.

	The underlying idea is that chromatic aberration produces characteristic spatial displacements between the red, green and blue color channels. These displacements can be analyzed to identify inconsistencies that may indicate image manipulation.

	The application uses mutual information to evaluate the relationship between color channels and estimate chromatic aberration parameters.

2. Methodology

	The detection method estimates the optical distortion center and chromatic aberration parameters. It analyzes the displacement directions of the red and blue color channels relative to the green channel.

	Mutual information is used as an information-theoretic measure during parameter estimation.

3. Optimization Methods

	Hill climbing algorithm

	Iterative parameter search

4. Graphical User Interface

	The application features a graphical user interface developed with PyQt5, allowing users to load images, select optimization methods, configure analysis parameters, visualize chromatic aberration corrections and inspect potential manipulation areas.

5. Image Analysis

	The application divides images into smaller regions (ROIs), evaluates local inconsistencies, and visualizes suspicious areas using colored rectangles and displacement vectors.

6. Technologies

	Python

	OpenCV

	NumPy

	scikit-learn

	PyQt5

7. Installation

	Install the required Python packages:

		pip install opencv-python numpy scikit-learn PyQt5

8. How to Run

	The project provides two ways to run the application.

	8.1. Command-Line Version (Without GUI)

		Run the standalone script from the project directory:

			python chromatic_aberration_detection.py

		Input image:

			The script expects a color image named chromatic.jpg.

			Place the image in the same directory as chromatic_aberration_detection.py.
	
			The image filename is specified directly in the source code.

		The script estimates the chromatic aberration parameters, displays the displacement vectors and performs regional image analysis using OpenCV windows.

	8.2. Graphical User Interface

		Start the PyQt5 graphical application:

			python gui.py

		Input image:

			Load an image using the Load button.

			Select the image from your computer through the file dialog.

			No predefined image filename or fixed input path is required.

	The GUI supports image loading, parameter estimation, regional analysis and visualization of the results.

	Input Image Requirements

		Both versions expect a color image with three color channels (BGR format), readable by OpenCV.

9. Execution Environment

	Microsoft Windows

	Python

	Command Prompt (CMD)

10. Author

	SZBR

	Original project developed in 2020.
