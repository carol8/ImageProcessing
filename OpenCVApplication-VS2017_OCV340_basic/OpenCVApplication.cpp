// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <sstream>
#include <random>


void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


//--------------------------------------------------- L1 ---------------------------------------------------
void testAdditiveImage(int factor) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar added = (val + factor) > 255 ? 255 : (val + factor);
				dst.at<uchar>(i, j) = added;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("modified image", dst);
		waitKey();
	}
}

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
	auto size = static_cast<size_t>(size_s);
	std::unique_ptr<char[]> buf(new char[size]);
	std::snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

void testMultiplicativeImage(float factor) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar multiplied = (val * factor) > 255 ? 255 : ((int)(val * factor));
				dst.at<uchar>(i, j) = multiplied;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		std::string head, tail, filename, delimiter;
		filename = fname;
		delimiter = ".";
		head = filename.substr(0, filename.find(delimiter));
		filename.erase(0, filename.find(delimiter) + delimiter.length());
		tail = filename;

		imshow("input image", src);
		imshow("modified image", dst);

		std::string full_path = format("%s%f.%s", head.c_str(), factor, tail.c_str());

		imwrite(full_path, dst);
		waitKey();
	}
}

void createImage() {
	Mat img(256, 256, CV_8UC3);
	int height = img.rows;
	int width = img.cols;
	for (int i = 0; i < height / 2; i++) {
		for (int j = 0; j < width / 2; j++) {
			Vec3b pixel(255, 255, 255);
			img.at<Vec3b>(i, j) = pixel;
		}
	}
	for (int i = 0; i < height / 2; i++) {
		for (int j = width / 2 + 1; j < width; j++) {
			Vec3b pixel(0, 0, 255);
			img.at<Vec3b>(i, j) = pixel;
		}
	}
	for (int i = height / 2 + 1; i < height; i++) {
		for (int j = 0; j < width / 2; j++) {
			Vec3b pixel(0, 255, 0);
			img.at<Vec3b>(i, j) = pixel;
		}
	}
	for (int i = height / 2 + 1; i < height; i++) {
		for (int j = width / 2 + 1; j < width; j++) {
			Vec3b pixel(0, 255, 255);
			img.at<Vec3b>(i, j) = pixel;
		}
	}

	imshow("steag", img);
	waitKey();
}

void displayMatrix(Mat input) {
	int height = input.rows;
	int width = input.cols;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%f ", input.at<float>(i, j));
		}
		printf("\n");
	}
}

float determinant(Mat a, float k)
{
	float s = 1, det = 0;
	Mat b(3, 3, CV_32FC1);
	int i, j, m, n, c;
	if (k == 1)
	{
		return a.at<float>(0, 0);
	}
	else
	{
		det = 0;
		for (c = 0; c < k; c++)
		{
			m = 0;
			n = 0;
			for (i = 0; i < k; i++)
			{
				for (j = 0; j < k; j++)
				{
					b.at<float>(i, j) = 0;
					if (i != 0 && j != c)
					{
						b.at<float>(m, n) = a.at<float>(i, j);
						if (n < (k - 2))
							n++;
						else
						{
							n = 0;
							m++;
						}
					}
				}
			}
			det = det + s * (a.at<float>(0, c) * determinant(b, k - 1));
			s = -1 * s;
		}
	}
	return (det);
}

// function to find the transpose of a matrix
Mat tp(Mat num, Mat fac, float r)
{
	int i, j;
	Mat b(3, 3, CV_32FC1), inverse(3, 3, CV_32FC1);
	float d;

	for (i = 0; i < r; i++)
	{
		for (j = 0; j < r; j++)
		{
			b.at<float>(i, j) = fac.at<float>(j, i);
		}
	}

	d = determinant(num, r);
	inverse = (1 / d) * b;
	return inverse;
}

// function for cofactor calculation
Mat cofactor(Mat num, float f)
{
	Mat b(3, 3, CV_32FC1), fac(3, 3, CV_32FC1);
	int p, q, m, n, i, j;
	for (q = 0; q < f; q++)
	{
		for (p = 0; p < f; p++)
		{
			m = 0;
			n = 0;
			for (i = 0; i < f; i++)
			{
				for (j = 0; j < f; j++)
				{
					if (i != q && j != p)
					{
						b.at<float>(m, n) = num.at<float>(i, j);
						if (n < (f - 2))
							n++;
						else
						{
							n = 0;
							m++;
						}
					}
				}
			}
			fac.at<float>(q, p) = pow(-1, q + p) * determinant(b, f - 1);
		}
	}
	return tp(num, fac, f);
}

void inverseMatrix() {
	Mat normal(3, 3, CV_32FC1);
	normal.at<float>(0, 0) = 1.2;
	normal.at<float>(0, 1) = 2.5;
	normal.at<float>(0, 2) = 3.7;
	normal.at<float>(1, 0) = 0;
	normal.at<float>(1, 1) = 2.9;
	normal.at<float>(1, 2) = 6.66;
	normal.at<float>(2, 0) = 6.9;
	normal.at<float>(2, 1) = 0;
	normal.at<float>(2, 2) = 0;

	Mat normal_copy = normal;

	printf("Normal matrix:\n");
	displayMatrix(normal);
	printf("\nInverse matrix(built-in OpenCV method):\n");
	displayMatrix(normal.inv());
	printf("\nInverse matrix(custom made function)\n");
	displayMatrix(cofactor(normal_copy, 3));
	getchar();
	getchar();
}



//--------------------------------------------------- L2 ---------------------------------------------------
// 2.7.1
void displayChannels() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat b = Mat(height, width, CV_8UC1);
		Mat g = Mat(height, width, CV_8UC1);
		Mat r = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				b.at<uchar>(i, j) = pixel[0];
				g.at<uchar>(i, j) = pixel[1];
				r.at<uchar>(i, j) = pixel[2];
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("b channel", b);
		imshow("g channel", g);
		imshow("r channel", r);
		waitKey();
	}
}

// 2.7.2
void convertBGRToGray() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat gray = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				gray.at<uchar>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("converted image ", gray);
		waitKey();
	}
}

// 2.7.3
void convertGrayToBinary(uchar threshold) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat binarized = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel < threshold) {
					binarized.at<uchar>(i, j) = 0;
				}
				else {
					binarized.at<uchar>(i, j) = 255;
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("binarized image ", binarized);
		waitKey();
	}
}

// 2.7.4
double minVec(std::vector<double> vals) {
	double min = 255;
	int size = vals.size();
	for (int i = 0; i < size; i++) {
		if (min > vals.at(i)) {
			min = vals.at(i);
		}
	}
	return min;
}

double maxVec(std::vector<double> vals) {
	double max = 0;
	int size = vals.size();
	for (int i = 0; i < size; i++) {
		if (max < vals.at(i)) {
			max = vals.at(i);
		}
	}
	return max;
}

void convertBGRToHSVAndDisplay() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat h = Mat(height, width, CV_8UC1);
		Mat s = Mat(height, width, CV_8UC1);
		Mat v = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				double r{ pixel[2] / 255.0 };
				double g{ pixel[1] / 255.0 };
				double b{ pixel[0] / 255.0 };
				std::vector<double> channels{ r, g, b };
				double min{ minVec(channels) };
				double max{ maxVec(channels) };
				double contrast = max - min;

				//V
				double val{ max };
				double sat{ val != 0.0 ? contrast / val : 0.0 };
				double hue{ 0.0 };

				if (contrast != 0) {
					if (max == r) {
						hue = 60 * (g - b) / contrast;
					}
					else if (max == g) {
						hue = 120 + 60 * (b - r) / contrast;
					}
					else {
						hue = 240 + 60 * (r - g) / contrast;
					}
				}
				if (hue < 0) {
					hue += 360;
				}

				h.at<uchar>(i, j) = static_cast<uchar>(hue * 255 / 360);
				s.at<uchar>(i, j) = static_cast<uchar>(sat * 255);
				v.at<uchar>(i, j) = static_cast<uchar>(val * 255);
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("h channel", h);
		imshow("s channel", s);
		imshow("v channel", v);
		waitKey();
	}
}

// 2.7.5
bool isInside(Mat img, int i, int j) {
	return i >= 0 && i < img.rows&& j >= 0 && j < img.cols;
}

void testIsInside(int i, int j) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		printf("Is the coordinate at (i, j) inside the image: %s", isInside(src, i, j) == 1 ? "yes" : "no");

		getchar();
		getchar();
	}
}


// 3.6.1 && 3.6.2 && 3.6.3 && 3.6.4 && 3.6.5 && 3.6.6
std::vector<double> computeFDP(std::vector<int> hist, int M) {
	std::vector<double> fdp;
	for (int i = 0; i < hist.size(); i++) {
		fdp.push_back(((double)hist.at(i)) / M);
		printf("%d: %lf\n", i, fdp.at(i));
	}
	return fdp;
}

std::vector<int> computeThresholds(std::vector<double> fdp) {
	const int WH{ 5 };
	const double TH{ 0.0003 };
	std::vector<int> peaks;
	for (int i = 0 + WH; i < 255 - WH; i++) {
		double avg{ 0.0 };
		double max{ 0.0 };
		for (int j = i - WH; j < i + WH; j++) {
			avg += fdp.at(j);
			if (max < fdp.at(j)) {
				max = fdp.at(j);
			}
		}
		avg /= 2 * WH + 1;
		if (fdp.at(i) > avg + TH && fdp.at(i) == max) {
			peaks.push_back(i);
		}
	}
	peaks.insert(peaks.begin(), 0.0);
	peaks.push_back(255);
	return peaks;
}

bool isInside(int height, int width, int i, int j) {
	return i >= 0 && j >= 0 && i < height&& j < width;
}

void computeHistogram(const int bins) {
	char fname[MAX_PATH];
	std::vector<int> hist(bins, 0);
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				++hist.at(src.at<uchar>(i, j) * bins / 256);
			}
		}
		std::vector<double> fdp = computeFDP(hist, height * width);
		if (bins == 256) {
			std::vector<int> peaks = computeThresholds(fdp);
			std::vector<int> pixelMap(256, 0);
			int index{ 0 };
			for (int i = 0; i < 256; i++) {
				if (index < peaks.size() - 1 && abs(i - peaks.at(index)) > abs(i - peaks.at(index + 1))) {
					++index;
				}
				pixelMap.at(i) = peaks.at(index);
			}
			Mat reduced_image = Mat(height, width, CV_8UC1);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					reduced_image.at<uchar>(i, j) = pixelMap.at(src.at<uchar>(i, j));
				}
			}
			imshow("Reduced image", reduced_image);

			Mat reduced_image_dithered = Mat(height, width, CV_8UC1);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					reduced_image_dithered.at<uchar>(i, j) = pixelMap.at(src.at<uchar>(i, j));
					int error{ src.at<uchar>(i, j) - reduced_image_dithered.at<uchar>(i, j) };
					if (isInside(height, width, i, j + 1)) {
						reduced_image_dithered.at<uchar>(i, j) += 7 * error / 16;
					}
					if (isInside(height, width, i + 1, j - 1)) {
						reduced_image_dithered.at<uchar>(i, j) += 3 * error / 16;
					}
					if (isInside(height, width, i + 1, j)) {
						reduced_image_dithered.at<uchar>(i, j) += 5 * error / 16;
					}
					if (isInside(height, width, i + 1, j + 1)) {
						reduced_image_dithered.at<uchar>(i, j) += error / 16;
					}
				}
			}
			imshow("Reduced image dithered", reduced_image_dithered);
		}
		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		showHistogram("histogram", &hist[0], bins, 200);
		waitKey();
	}
}

// 3.6.7
void reduceHSV() {
	char fname[MAX_PATH];
	std::vector<int> hist(256, 0);
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat hsv = Mat(height, width, CV_8UC3);
		cvtColor(src, hsv, CV_BGR2HSV);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				++hist.at(hsv.at<Vec3b>(i, j)[0]);
			}
		}
		std::vector<double> fdp = computeFDP(hist, height * width);

		std::vector<int> peaks = computeThresholds(fdp);
		std::vector<int> pixelMap(256, 0);
		int index{ 0 };
		for (int i = 0; i < 256; i++) {
			if (index < peaks.size() - 1 && abs(i - peaks.at(index)) > abs(i - peaks.at(index + 1))) {
				++index;
			}
			pixelMap.at(i) = peaks.at(index);
		}
		Mat reduced_image = Mat(height, width, CV_8UC3);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel;
				pixel[0] = pixelMap.at(hsv.at<Vec3b>(i, j)[0]);
				pixel[1] = hsv.at<Vec3b>(i, j)[1];
				pixel[2] = hsv.at<Vec3b>(i, j)[2];
				reduced_image.at<Vec3b>(i, j) = pixel;
			}
		}
		Mat output = Mat(height, width, CV_8UC3);
		cvtColor(reduced_image, output, CV_HSV2BGR);
		imshow("Reduced image", output);

		imshow("input image", src);
		showHistogram("histogram", &hist[0], 256, 200);
		waitKey();
	}
}


// 4.4.1
int computeArea(Mat* src, Vec3b objectPixel) {
	int width{ src->cols };
	int height{ src->rows };
	int area = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ((*src).at<Vec3b>(i, j) == objectPixel) {
				++area;
			}
		}
	}
	return area;
}

Point computeCenterOfMass(Mat* src, Vec3b objectPixel, Mat* modified = nullptr) {
	int width{ src->cols };
	int height{ src->rows };
	int area{ computeArea(src, objectPixel) };
	int rbara{ 0 }, cbara{ 0 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ((*src).at<Vec3b>(i, j) == objectPixel) {
				rbara += i;
				cbara += j;
			}
		}
	}
	rbara /= area;
	cbara /= area;
	if (modified != nullptr) {
		line(*modified, Point(cbara, rbara), Point(cbara, rbara), Scalar(0, 0, 255), 5);
	}
	//modified->at<Vec3b>(rbara, cbara) = Vec3b(0, 0, 255);
	return Point(cbara, rbara);
}

double computeElongationAxisAngle(Mat* src, Vec3b objectPixel, Mat* modified = nullptr) {
	int width{ src->cols };
	int height{ src->rows };
	int area{ computeArea(src, objectPixel) };
	Point centerOfMass{ computeCenterOfMass(src, objectPixel, modified) };
	const int LINE_WIDTH{ 100 };
	int64_t sum1{ 0 }, sum2{ 0 }, sum3{ 0 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ((*src).at<Vec3b>(i, j) == objectPixel) {
				sum1 += (i - centerOfMass.y) * (j - centerOfMass.x);
				sum2 += (j - centerOfMass.x) * (j - centerOfMass.x);
				sum3 += (i - centerOfMass.y) * (i - centerOfMass.y);
			}
		}
	}
	double elongationAxisAngle = atan2(2 * sum1, sum2 - sum3);
	if (modified != nullptr) {
		Point p1((int)(centerOfMass.x - LINE_WIDTH / 2 * cos(elongationAxisAngle / 2)), (int)(centerOfMass.y - LINE_WIDTH / 2 * sin(elongationAxisAngle / 2)));
		Point p2((int)(centerOfMass.x + LINE_WIDTH / 2 * cos(elongationAxisAngle / 2)), (int)(centerOfMass.y + LINE_WIDTH / 2 * sin(elongationAxisAngle / 2)));
		line(*modified, p1, p2, Scalar(255, 255, 0), 1);
	}
	return atan2(2 * sum1, sum2 - sum3);
}

int computePerimeter(Mat* src, Vec3b objectPixel, Mat* modified) {
	int width{ src->cols };
	int height{ src->rows };
	int perimeter{ 0 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ((*src).at<Vec3b>(i, j) == objectPixel) {
				if ((*src).at<Vec3b>(i + 1, j) != objectPixel ||
					(*src).at<Vec3b>(i, j + 1) != objectPixel ||
					(*src).at<Vec3b>(i - 1, j) != objectPixel ||
					(*src).at<Vec3b>(i, j - 1) != objectPixel) {
					++perimeter;
					modified->at<Vec3b>(i, j) = Vec3b(255, 0, 255);
					modified->at<Vec3b>(i, j + 1) = Vec3b(255, 0, 255);
					modified->at<Vec3b>(i, j - 1) = Vec3b(255, 0, 255);
					modified->at<Vec3b>(i + 1, j) = Vec3b(255, 0, 255);
					modified->at<Vec3b>(i - 1, j) = Vec3b(255, 0, 255);
				}
			}
		}
	}
	return perimeter;
}

double computeAspectRatio(Mat* src, Vec3b objectPixel) {
	int width{ src->cols };
	int height{ src->rows };
	int rmin{ height }, rmax{ 0 }, cmin{ width }, cmax{ 0 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ((*src).at<Vec3b>(i, j) == objectPixel) {
				if (i < rmin) {
					rmin = i;
				}
				if (i > rmax) {
					rmax = i;
				}
				if (j < cmin) {
					cmin = j;
				}
				if (j > cmax) {
					cmax = j;
				}
			}
		}
	}
	return ((cmax - cmin) + 1.0) / ((rmax - rmin) + 1.0);
}

void computeProjections(Mat* src, Vec3b objectPixel) {
	int width{ src->cols };
	int height{ src->rows };
	int max_orizontala{ 0 };
	int max_verticala{ 0 };
	std::vector<int> ap_orizontala(width, 0);
	std::vector<int> ap_verticala(height, 0);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src->at<Vec3b>(i, j) == objectPixel) {
				++ap_orizontala.at(j);
				++ap_verticala.at(i);
				if (max_orizontala < ap_orizontala.at(j)) {
					max_orizontala = ap_orizontala.at(j);
				}
				if (max_verticala < ap_verticala.at(i)) {
					max_verticala = ap_verticala.at(i);
				}
			}
		}
	}
	Mat proiectie_orizontala = Mat(max_orizontala, width, CV_8UC3);
	Mat proiectie_verticala = Mat(height, max_verticala, CV_8UC3);
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < ap_orizontala.at(i); j++) {
			proiectie_orizontala.at<Vec3b>(max_orizontala - j - 1, i) = objectPixel;
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < ap_verticala.at(i); j++) {
			proiectie_verticala.at<Vec3b>(i, max_verticala - j - 1) = objectPixel;
		}
	}
	imshow("Proiectie orizontala", proiectie_orizontala);
	imshow("Proiectie verticala", proiectie_verticala);
}

// 4.4.2
void copyAllPixels(Mat src, Mat* dest, Vec3b pixel) {
	int width{ src.cols };
	int height{ src.rows };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<Vec3b>(i, j) == pixel) {
				dest->at<Vec3b>(i, j) = pixel;
			}
		}
	}
}

void filterImage(int min_area, double min_phi, double max_phi) {
	const Vec3b background_pixel(255, 255, 255);
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		int width{ src.cols };
		int height{ src.rows };
		std::vector<Vec3b> labels;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				Vec3b pixel = src.at<Vec3b>(i, j);
				if (pixel != background_pixel) {
					if (std::find(labels.begin(), labels.end(), pixel) == labels.end()) {
						labels.push_back(pixel);
					}
				}
			}
		}

		Mat filtered = Mat(height, width, CV_8UC3, Vec3b(255, 255, 255));
		std::vector<Vec3b>::iterator it;
		for (int i = 0; i < labels.size(); i++) {
			int area = computeArea(&src, labels.at(i));
			double phi = computeElongationAxisAngle(&src, labels.at(i));
			if (area > min_area && phi > min_phi && phi < max_phi) {
				copyAllPixels(src, &filtered, labels.at(i));
			}
		}

		imshow("Original", src);
		imshow("Filtered", filtered);
		waitKey();
	}
}

void mouseCallbackGeometry(int event, int x, int y, int flags, void* param) {
	Mat* src = (Mat*)param;
	Mat modified = src->clone();
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("\nPos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		int width{ src->cols };
		int height{ src->rows };
		Vec3b pixel{ (*src).at<Vec3b>(y, x) };

		//Area
		int area{ computeArea(src, pixel) };
		//Perimeter
		int perimeter{ computePerimeter(src, pixel, &modified) };
		//Elongation axis
		double elongationAxisAngle{ computeElongationAxisAngle(src, pixel, &modified) };
		//Center of mass
		Point centerOfMass{ computeCenterOfMass(src, pixel, &modified) };
		//Thinness ratio
		double thinnessRatio{ 4 * PI * area / (perimeter * perimeter) };
		//Aspect ratio
		double aspectRatio{ computeAspectRatio(src, pixel) };
		//Projections
		computeProjections(src, pixel);

		printf("Area: %d\n", area);
		printf("Center of mass (x, y): %d, %d\n", centerOfMass.x, centerOfMass.y);
		printf("Elongation axis: %lf\n", elongationAxisAngle);
		printf("Perimeter: %d\n", perimeter);
		printf("Thinness ratio: %lf\n", thinnessRatio);
		printf("Aspect ratio: %lf\n", aspectRatio);

		imshow("Modified", modified);
	}
}

void computeGeometry() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
	namedWindow("Geometry", 1);
	setMouseCallback("Geometry", mouseCallbackGeometry, &src);
	imshow("Geometry", src);
	waitKey(0);
}



// 5.5.1 & 5.5.2
struct Adiacent {
	int n{};
	int* di;
	int* dj;
};

enum TipAdicenta {
	N4, 
	N8,
	ANTERIORI,
	CUSTOM
};

void bfs(const Mat& src, const Adiacent& adiacenta, int** visited, int class_index, std::pair<int, int> startingPosition) {
	std::queue<std::pair<int, int>> queue;
	queue.push(startingPosition);
	visited[startingPosition.first][startingPosition.second] = class_index;
	while (!queue.empty()) {
		std::pair<int, int> current = queue.front();
		queue.pop();
		//std::cout << "(" << current.first << ", " << current.second << "), visited: " << visited[current.first][current.second] << ":\n";
		for (int i = 0; i < adiacenta.n; i++) {
			int newRow = current.first + adiacenta.di[i];
			int newCol = current.second + adiacenta.dj[i];
			//std::cout << "	(" << newRow << ", " << newCol << "), visited: " << visited[newRow][newCol] << '\n';
			if (isInside(src, newRow, newCol) && src.at<uchar>(newRow, newCol) == 0 && visited[newRow][newCol] == 0) {
				queue.push(std::pair<int, int>(newRow, newCol));
				visited[newRow][newCol] = class_index;
			}
		}
	}
}

void labelImageBfs(const Adiacent& ad) {
	const Vec3b background_pixel(255, 255, 255);
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src{ imread(fname, CV_LOAD_IMAGE_GRAYSCALE) };
		int width{ src.cols };
		int height{ src.rows };
		
		Adiacent adiacent = ad;

		int** visited = new int*[height];
		for (int i = 0; i < height; i++) {
			visited[i] = new int[width] {};
		}

		int class_index{ 1 };

		std::default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && !visited[i][j]) {
					bfs(src, adiacent, visited, class_index, std::pair<int, int>(i, j));
					++class_index;
				}
			}
		}

		Vec3b* array{ new Vec3b[class_index] };

		for (int i = 0; i < class_index; i++) {
			array[i] = Vec3b(d(gen), d(gen), d(gen));
		}

		Mat labeled{ Mat(height, width, CV_8UC3) };

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (visited[i][j]) {
					labeled.at<Vec3b>(i, j) = array[visited[i][j]];
				}
				else {
					labeled.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
			}
		}

		imshow("Original", src);
		imshow("Labeled", labeled);
		waitKey(0);
	}
}

void labelImageBfs(int tip_adiacent) {
	Adiacent adiacent;
	int di1[]{ 1, 0, -1, 0 };
	int dj1[]{ 0, 1, 0, -1 };
	int di2[]{ 1, 1, 1, 0, 0,-1,-1,-1 };
	int dj2[]{ 1, 0,-1, 1,-1, 1, 0,-1 };
	int di3[]{ 0,-1,-1,-1 };
	int dj3[]{ -1, 1, 0,-1 };
	switch (tip_adiacent) {
	case N4:
		adiacent.n = 4;
		adiacent.di = di1;
		adiacent.dj = dj1;
		break;
	case N8:
		adiacent.n = 8;
		adiacent.di = di2;
		adiacent.dj = dj2;
		break;
	case ANTERIORI:
		adiacent.n = 8;
		adiacent.di = di3;
		adiacent.dj = dj3;
		break;
	}
	labelImageBfs(adiacent);
}

// 5.5.3
class DisjointSet {
private:
	DisjointSet* m_parent{};
	int m_rank{};
	int m_value{};
public:

	DisjointSet(DisjointSet* parent, int rank, int value):
		m_parent{parent}, m_rank{rank}, m_value{value}
	{
	}

	DisjointSet(int value):
		m_parent{this}, m_rank{0}, m_value{value}
	{
	}

	int getValue() {
		return m_value;
	}

	void setValue(int value) {
		m_value = value;
	}

	void unionSet(DisjointSet* ds) {
		this->findSet()->link(ds->findSet());
	}

	void link(DisjointSet* ds) {
		if (m_rank > ds->m_rank) {
			ds->m_parent = this;
		}
		else {
			m_parent = ds;
			if (m_rank == ds->m_rank) {
				ds->m_rank++;
			}
		}
	}

	DisjointSet* findSet() {
		DisjointSet* ds = this;
		while (ds != ds->m_parent) {
			ds = ds->m_parent;
		}

		DisjointSet* ds2 = this;
		while (ds2->m_parent != ds2->m_parent->m_parent) {
			DisjointSet* ds2_old_parent = ds2->m_parent;
			ds2->m_parent = ds;
			ds2 = ds2_old_parent;
		}
		return ds;
	}
};

void labelImage2Pass(const Adiacent& ad) {
	const Vec3b background_pixel(255, 255, 255);
	char fname[MAX_PATH];

	std::ofstream out("out.txt");
	std::streambuf* coutbuf = std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt

	if (openFileDlg(fname))
	{
		Mat src{ imread(fname, CV_LOAD_IMAGE_GRAYSCALE) };
		int width{ src.cols };
		int height{ src.rows };

		Adiacent adiacent = ad;

		DisjointSet*** visited = new DisjointSet** [height];
		for (int i = 0; i < height; i++) {
			visited[i] = new DisjointSet*[width] {};
		}

		std::default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);

		std::vector<DisjointSet*> dsVector;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && !visited[i][j]) {
					bool labelPresent = false;
					DisjointSet* minimum{};
					for (int k = 0; k < adiacent.n; k++) {
						if (isInside(height, width, i + adiacent.di[k], j + adiacent.dj[k]) && visited[i + adiacent.di[k]][j + adiacent.dj[k]]) {
							if (!labelPresent || minimum->findSet()->getValue() > visited[i + adiacent.di[k]][j + adiacent.dj[k]]->findSet()->getValue()) {
								if (minimum) {
									minimum->unionSet(visited[i + adiacent.di[k]][j + adiacent.dj[k]]);
								}
								minimum = visited[i + adiacent.di[k]][j + adiacent.dj[k]];
							}
							if (minimum != visited[i + adiacent.di[k]][j + adiacent.dj[k]]) {
								visited[i + adiacent.di[k]][j + adiacent.dj[k]]->unionSet(minimum);
							}
							labelPresent = true;
						}
					}
					if (labelPresent) {
						visited[i][j] = minimum;
					}
					else {
						DisjointSet* ds = new DisjointSet(dsVector.size() + 1);
						visited[i][j] = ds;
						dsVector.push_back(ds);
					}
				}
			}
		}

		Vec3b* array{ new Vec3b[dsVector.size()] };
		for (int i = 0; i < dsVector.size(); i++) {
			array[i] = Vec3b(d(gen), d(gen), d(gen));
		}

		Mat labeled{ Mat(height, width, CV_8UC3) };

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (visited[i][j]) {
					labeled.at<Vec3b>(i, j) = array[visited[i][j]->findSet()->getValue()];
				}
				else {
					labeled.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
				}
			}
		}

		for (DisjointSet* dsp : dsVector) {
			delete dsp;
		}

		imshow("Original", src);
		imshow("Labeled", labeled);
		waitKey(0);
	}

	std::cout.rdbuf(coutbuf);
}

void labelImage2Pass(int tip_adiacent) {
	Adiacent adiacent;
	int di1[]{ 1, 0, -1, 0 };
	int dj1[]{ 0, 1, 0, -1 };
	int di2[]{ 1, 1, 1, 0, 0,-1,-1,-1 };
	int dj2[]{ 1, 0,-1, 1,-1, 1, 0,-1 };
	int di3[]{ 0,-1,-1,-1 };
	int dj3[]{ -1, 1, 0,-1 };
	switch (tip_adiacent) {
	case N4:
		adiacent.n = 4;
		adiacent.di = di1;
		adiacent.dj = dj1;
		break;
	case N8:
		adiacent.n = 8;
		adiacent.di = di2;
		adiacent.dj = dj2;
		break;
	case ANTERIORI:
		adiacent.n = 8;
		adiacent.di = di3;
		adiacent.dj = dj3;
		break;
	}
	labelImage2Pass(adiacent);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images afrom folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - 1.10.3 Image intensity increase\n");
		printf(" 11 - 1.10.4 Image contrast increase\n");
		printf(" 12 - 1.10.5 Display flag\n");
		printf(" 13 - 1.10.6 Matrix inverse\n");
		printf(" 14 - 2.7.1 (Color to R, G, B channels)\n");
		printf(" 15 - 2.7.2 (Color to Grayscale)\n");
		printf(" 16 - 2.7.3 (Grayscale to binary)\n");
		printf(" 17 - 2.7.4 (Color to H, S, V channels)\n");
		printf(" 18 - 2.7.5 (IsInside test)\n");
		printf(" 19 - 3.6.1 && 3.6.2 && 3.6.3 && 3.6.4 (Histogram (bins <= 256) + FDP)\n");
		printf(" 20 - 3.6.5 && 3.6.6 (Grayscale levels reducing)\n");
		printf(" 21 - 3.6.7 (HSV H reducing)\n");
		printf(" 22 - 4.4.1 (Geometry)\n");
		printf(" 23 - 4.4.2 (Filtering based on area and elongation angle)\n");
		printf(" 24 - 5.5.1 & 5.5.2 (Labeling using BFS)\n");
		printf(" 25 - 5.5.3 (Labeling using 2 passes and Disjoint sets)\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		std::vector<int> hist(256, 0);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			uchar intensity;
			printf("Intensity factor: ");
			scanf("%d", &intensity);
			testAdditiveImage(intensity);
			break;
		case 11:
			uchar contrast;
			printf("Intensity factor: ");
			scanf("%d", &contrast);
			testMultiplicativeImage(contrast);
			break;
		case 12:
			createImage();
			break;
		case 13:
			inverseMatrix();
			break;
		case 14:
			displayChannels();
			break;
		case 15:
			convertBGRToGray();
			break;
		case 16:
			printf("Threshold: ");
			uchar threshold;
			scanf("%d", &threshold);
			convertGrayToBinary(threshold);
			break;
		case 17:
			convertBGRToHSVAndDisplay();
			break;
		case 18:
			printf("i: ");
			int i;
			scanf("%d", &i);
			printf("j: ");
			int j;
			scanf("%d", &j);
			testIsInside(i, j);
			break;
		case 19:
			printf("bins: ");
			int bins;
			scanf("%d", &bins);
			computeHistogram(bins);
			break;
		case 20:
			computeHistogram(256);
			break;
		case 21:
			reduceHSV();
			break;
		case 22:
			computeGeometry();
			break;
		case 23:
			int min_area;
			std::cout << "Minimum area: ";
			std::cin >> min_area;
			double min_phi, max_phi;
			std::cout << "Minimum, maximum phi: ";
			std::cin >> min_phi >> max_phi;
			filterImage(min_area, min_phi, max_phi);
			break;
		case 24:
			int tip_ad_bfs;
			std::cout << "Choose type of vicinity (0 - N4, 1 - N8, 2 - PREVIOUS, 3 - CUSTOM): ";
			std::cin >> tip_ad_bfs;
			if (tip_ad_bfs == 3) {
				Adiacent adiacent;
				std::cout << "Input number of adiacent cells: ";
				std::cin >> adiacent.n;
				adiacent.di = new int[adiacent.n];
				adiacent.dj = new int[adiacent.n];
				std::cout << "Input di: ";
				for (int i = 0; i < adiacent.n; i++) {
					std::cin >> adiacent.di[i];
				}
				std::cout << "Input dj: ";
				for (int i = 0; i < adiacent.n; i++) {
					std::cin >> adiacent.dj[i];
				}
				labelImageBfs(adiacent);
			}
			else {
				labelImageBfs(tip_ad_bfs);
			}
			break;
		case 25:
			int tip_ad_2pass;
			std::cout << "Choose type of vicinity (0 - N4, 1 - N8, 2 - PREVIOUS, 3 - CUSTOM): ";
			std::cin >> tip_ad_2pass;
			if (tip_ad_2pass == 3) {
				Adiacent adiacent;
				std::cout << "Input number of adiacent cells: ";
				std::cin >> adiacent.n;
				adiacent.di = new int[adiacent.n];
				adiacent.dj = new int[adiacent.n];
				std::cout << "Input di: ";
				for (int i = 0; i < adiacent.n; i++) {
					std::cin >> adiacent.di[i];
				}
				std::cout << "Input dj: ";
				for (int i = 0; i < adiacent.n; i++) {
					std::cin >> adiacent.dj[i];
				}
				labelImage2Pass(adiacent);
			}
			else {
				labelImage2Pass(tip_ad_2pass);
			}
			break;
		}
	} while (op != 0);
	return 0;
}