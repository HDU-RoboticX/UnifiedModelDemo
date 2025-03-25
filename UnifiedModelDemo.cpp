#include <iostream>
#include <opencv2/opencv.hpp>//Used for image reading, display, and writing, as well as preprocessing binary images
#include <iostream>//Used for output debugging information
#include <interface.h>


using namespace cv;
using namespace std;



// Global variables (used to store user input)
int toolSize = 3;  // Default tool size
int scale = 0;
int cntRate = 0;     // Default contact rate (1-100)
int delay = 60;
Mat binaryImage;
Mat originImage;

const int BUTTON_WIDTH = 350;
const int BUTTON_HEIGHT = 50;

Rect button1(20, 50, BUTTON_WIDTH, BUTTON_HEIGHT);
Rect button2(20, 120, BUTTON_WIDTH, BUTTON_HEIGHT);


// Function to judge whether two pixel values are equal
bool judge(uchar a, uchar b) {
	if (a == b)
		return true;
	else
		return false;
}

// Edge extraction function
Mat edge_Extraction(Mat img) {
	Mat edge = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));
	int jmpCnt = 0;// Transition count
	uchar temp = 0;// Temporary variable to store the previous pixel value

	for (int row = 0; row < img.rows; row++)// Row loop
	{
		for (int col = 0; col < img.cols; col++)// Column loop
		{
            // Initialize variables
			jmpCnt = 0;
			if (row < 1 || row >= img.rows - 1 || col < 1 || col >= img.cols - 1) {
				continue;
			}
			else {
                // Top-left
				temp = img.at<uchar>(row - 1, col - 1);
				// Top
				if (!judge(temp, img.at<uchar>(row - 1, col))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row - 1, col);
				// Top-right
				if (!judge(temp, img.at<uchar>(row - 1, col + 1))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row - 1, col + 1);
				// Right
				if (!judge(temp, img.at<uchar>(row, col + 1))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row, col + 1);
                // Bottom-right
				if (!judge(temp, img.at<uchar>(row + 1, col + 1))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row + 1, col + 1);
                // Bottom
				if (!judge(temp, img.at<uchar>(row + 1, col))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row + 1, col);
                // Bottom-left
				if (!judge(temp, img.at<uchar>(row + 1, col - 1))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row + 1, col - 1);
                // Left
				if (!judge(temp, img.at<uchar>(row, col - 1))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row, col - 1);
                // Top-left
				if (!judge(temp, img.at<uchar>(row - 1, col - 1))) {
					jmpCnt++;
				}
				temp = img.at<uchar>(row - 1, col - 1);
			}
            // Edge point processing
			if (jmpCnt == 2 && img.at<uchar>(row, col) == 255) {// Use inner points to handle edges to avoid double edges
                // Edge point processing
				edge.at<uchar>(row, col) = 255;
			}
		}
	}
	
	return edge;
}

// Image resizing function
Mat img_Resize(Mat img, float img_Factor) {
    Mat res = img.clone();
    Size tmpSize = Size(res.cols * img_Factor, res.rows * img_Factor);
    resize(img, res, tmpSize, 0, 0, INTER_AREA);// Pixel reconstruction
    return res;
}

// Compute cumulative arc length
vector<double> computeCumulativeArcLength(const vector<Point>& path) {
    vector<double> arcLength;
    if (path.empty()) return arcLength;

    arcLength.push_back(0.0);
    for (size_t i = 1; i < path.size(); i++) {
        double segmentLength = norm(path[i] - path[i - 1]); // Compute Euclidean distance between adjacent points
        arcLength.push_back(arcLength.back() + segmentLength);
    }
    return arcLength;
}

// Resample path at equidistant arc length positions
vector<Point> resamplePath(const vector<Point>& path, int numPoints) {
    vector<Point> newPath;
    if (path.size() < 2 || numPoints < 2) return newPath;

    vector<double> arcLength = computeCumulativeArcLength(path);
    double totalLength = arcLength.back();
    double segmentLength = totalLength / (numPoints - 1);  // Compute uniform intervals

    newPath.push_back(path[0]); // Start point
    size_t currentIndex = 0;

    for (int i = 1; i < numPoints - 1; i++) {
        double targetLength = i * segmentLength;

        while (currentIndex < path.size() - 1 && arcLength[currentIndex] < targetLength) {
            currentIndex++;
        }

        if (currentIndex == 0) continue; // Ensure Index

        double alpha = (targetLength - arcLength[currentIndex - 1]) /
            (arcLength[currentIndex] - arcLength[currentIndex - 1]);
        Point interpolatedPoint = path[currentIndex - 1] * (1 - alpha) + path[currentIndex] * alpha;
        newPath.push_back(interpolatedPoint);
    }

    newPath.push_back(path.back()); // End point
    return newPath;
}

void animateCircleAlongBoundary(const Mat& binaryImage , Mat originImage) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // Use CHAIN_APPROX_NONE to ensure all boundary points are retrieved
    findContours(binaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    if (contours.empty()) {
        cerr << "No white region found in the image!" << endl;
        return;
    }

    // Select the largest contour
    size_t maxContourIdx = 0;
    double maxContourArea = 0.0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxContourArea) {
            maxContourArea = area;
            maxContourIdx = i;
        }
    }

    vector<Point> boundaryPath = contours[maxContourIdx];

    // **Uniform resampling to ensure smooth motion**
    vector<Point> uniformPath = resamplePath(boundaryPath, 500); // 生成500个均匀点

    if (uniformPath.empty()) {
        cerr << "Failed to extract a valid boundary path!" << endl;
        return;
    }

    namedWindow("Animation", WINDOW_AUTOSIZE);
    
    // Move along the smoothed and evenly distributed path
    for (size_t i = 0; i < uniformPath.size(); i++) {
        Mat frame = originImage.clone();
        //cvtColor(frame, frame, COLOR_GRAY2BGR); // RGB image trans

        // Draw a circle
        circle(frame, uniformPath[i], scale, Scalar(0, 0, 255), -1);

        imshow("Animation", frame);
        if (waitKey(delay) == 27) {
            cv::destroyWindow("Animation");
            break;
        } // Press ESC to exit
    }

}

Mat QuickConvertBin(Mat img, int throld) {
    for (int i = 0; i < img.rows; i++)//rows
    {
        for (int j = 0; j < img.cols; j++)//cols
        {
            if (img.at<uchar>(i, j) > throld) {
                img.at<uchar>(i, j) = 255;
            }
            else {
                img.at<uchar>(i, j) = 0;
            }
        }
    }
    return img;
}

void EAB_Gen_Demo() {
    originImage = imread("D:\\VCProj\\UnifiedModelDemo\\resource\\leaf.png");
    if (originImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return;
    }
    cvtColor(originImage, originImage, CV_BGR2GRAY);//gray image
    originImage = QuickConvertBin(originImage, 128);
    Interface instance;
    Mat result = instance.EAB_IAB_Extraction(originImage, 20, 255);// para:(binaryImageOfTarget,toolScale,contactResolution)
    imshow("Main Menu", result);
    waitKey(0);
}

Mat pixFilter(Mat img, int threshold) {
    Mat res(img.rows, img.cols, CV_8UC1, Scalar(0));
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; j++)
		{
            if (img.at<uchar>(i, j) >= threshold) {
                res.at<uchar>(i, j) = 255;
            }
		}
	}
    return res;
}

// Callback function for trackbar adjustment
void onTrackbarChange(int, void*) {
    // Update the display when the user adjusts the trackbar
    //cout << "Parameter Saved " << endl;
}

// Display the UI and obtain user input
void showUserInterface() {
    // Create a window
    namedWindow("UnifiedModelDisplay", WINDOW_AUTOSIZE);


    // Create trackbars (values ranging from 1 to 100)
    createTrackbar("CntRate", "UnifiedModelDisplay", &cntRate, 100, onTrackbarChange);

    createTrackbar("ToolSize", "UnifiedModelDisplay", &toolSize, 5, onTrackbarChange);

    createTrackbar("Speed", "UnifiedModelDisplay", &delay, 100, onTrackbarChange);

    // Display the interface
    Mat interface(200, 400, CV_8UC3, Scalar(220, 220, 220));
    putText(interface, "CntRate:0%-100%", Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(interface, "ToolSize:1-5", Point(20, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(interface, "Speed:1%-100%", Point(20, 120), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(interface, "Press 'Enter' to Display , Press 'Esc' to Exit", Point(20, 160), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    imshow("UnifiedModelDisplay", interface);
    int Thr;
    string src = "./resource/res";
    
    // Wait for the user to press Enter
    while (true) {
        int key = waitKey(10);
        if (key == 13 || key == 10) { 
            Thr = (int)((float)cntRate * 2.55);
            if (Thr < 1) {
                Thr = 1;
            }
            if (delay > 99) {
                delay = 99;
            }
            switch (toolSize) {
                case 1:scale = 5; break;
                case 2:scale = 10; break;
                case 3:scale = 20; break;
                case 4:scale = 30; break;
                case 5:scale = 40; break;
                default:scale = 5; break;
            }
            src = "./resource/res";
            src.append(to_string(scale));
            src.append(".hdu");
            binaryImage = imread(src, IMREAD_GRAYSCALE);
            if (binaryImage.empty()) {
                cerr << "Error: Could not load image!" << endl;
                return;
            }
            delay = 100 - delay;
            Mat filted = pixFilter(binaryImage, Thr);
            Mat edge = edge_Extraction(filted);
            animateCircleAlongBoundary(edge, originImage);
        }
    }
    cv::destroyWindow("UnifiedModelDisplay");
}

// callback function
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        if (button1.contains(Point(x, y))) {
            cout << "Button 1 Clicked: Running showUserInterface()" << endl;
            showUserInterface();
        }
        else if (button2.contains(Point(x, y))) {
            cout << "Button 2 Clicked: Running showContactRate()" << endl;
            showUserInterface();
        }
    }
}

// background
void drawGradientBackground(Mat& img) {
    for (int i = 0; i < img.rows; i++) {
        float alpha = (float)i / img.rows;
        Vec3b topColor(200, 230, 255);  
        Vec3b bottomColor(100, 150, 255);  
        Vec3b color = topColor * (1 - alpha) + bottomColor * alpha;
        img.row(i).setTo(color);
    }
}

// draw rectBtn
void drawButtons(Mat& img) {
    Scalar btnColor(50, 150, 255);  // btn
    Scalar borderColor(0, 0, 0);  // bd
    Scalar textColor(255, 255, 255);  // txt

    rectangle(img, button1, borderColor, 2);
    rectangle(img, button1.tl() + Point(1, 1), button1.br() - Point(1, 1), btnColor, FILLED);
    putText(img, "Unified Model Display", button1.tl() + Point(20, 38), FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2);

    rectangle(img, button2, borderColor, 2);
    rectangle(img, button2.tl() + Point(1, 1), button2.br() - Point(1, 1), btnColor, FILLED);
    putText(img, "Real-time Contact Rate", button2.tl() + Point(30, 38), FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2);
}

void Display() {
    // Load the original image
    originImage = imread("D:\\VCProj\\UnifiedModelDemo\\resource\\origin.hdu");
    if (originImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return;
    }

    //showUserInterface();

    Mat menu(350, 400, CV_8UC3);
    drawGradientBackground(menu);  // set background 
    drawButtons(menu);  // set btn
    namedWindow("Main Menu", WINDOW_AUTOSIZE);
    setMouseCallback("Main Menu", onMouse, nullptr);
    imshow("Main Menu", menu);
    while (true) {
        int k = waitKey(20);
        if (k == 27)
            break;
    }
}



int main() {
    
    Display();
    
    EAB_Gen_Demo();

    return 0;
}