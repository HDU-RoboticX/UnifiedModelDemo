#include <iostream>
#include <opencv2/opencv.hpp>//Used for image reading, display, and writing, as well as preprocessing binary images
#include <iostream>//Used for output debugging information
#include <interface.h>

#define TOOLSCALE 20 //toolscale of Model Generator
#define CNTRESOLUTION 255//Contact resolution of Model Generator

using namespace cv;
using namespace std;




// Global variables (used to store user input)
int toolSize = 3;  // Default tool size
int scale = 0;
int cntProgress = 0;     // Default contact Progress (1-100)
int delay = 60;
Mat binaryImage;
Mat originImage;
Mat displayImg;
Mat maskImage;

const int BUTTON_WIDTH = 300;
const int BUTTON_HEIGHT = 50;

Rect button1(20, 50, BUTTON_WIDTH, BUTTON_HEIGHT);
Rect button2(20, 120, BUTTON_WIDTH, BUTTON_HEIGHT);
Rect button3(20, 190, BUTTON_WIDTH, BUTTON_HEIGHT);

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

Mat ConvertColorResFromBin(Mat originImg) {

    Mat res = Mat(originImg.rows, originImg.cols, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 0; i < res.rows; i++) {
        for (int j = 0; j < res.cols; j++) {
            if (originImg.at<uchar>(i, j) >= 0) {
                res.at<Vec3b>(i, j)[0] = 180 - originImg.at<uchar>(i, j) * 180 / 255;
                res.at<Vec3b>(i, j)[1] = 255;
                res.at<Vec3b>(i, j)[2] = 255;
            }
            /*if (edge.at<uchar>(i, j) == 255) {
                res.at<Vec3b>(i, j)[0] = 0;
                res.at<Vec3b>(i, j)[1] = 0;
                res.at<Vec3b>(i, j)[2] = 0;
            }*/

        }
    }

    cvtColor(res, res, COLOR_HSV2BGR_FULL);
    return res;

}

//demo of EAB_IAB generation
void EAB_Gen(int scale,int resolution) {
    maskImage = imread(".\\ModelGenerate\\binaryResource.png");
    if (maskImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return;
    }
    cvtColor(maskImage, maskImage, CV_BGR2GRAY);//gray image
    maskImage = QuickConvertBin(maskImage, 128);
    Interface instance;
    Mat result = instance.EAB_IAB_Extraction(maskImage, scale, resolution);// para:(binaryImageOfTarget,toolScale,contactResolution)
    imwrite(".\\ModelGenerate\\mapping\\map1.png", result);
    Mat colorRes = ConvertColorResFromBin(result);
    imwrite(".\\ModelGenerate\\heat_map\\heat_map1.png", colorRes);

    maskImage = imread(".\\ModelGenerate\\binaryResource2.png");
    if (maskImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return;
    }
    cvtColor(maskImage, maskImage, CV_BGR2GRAY);//gray image
    maskImage = QuickConvertBin(maskImage, 128);
    result = instance.EAB_IAB_Extraction(maskImage, scale, resolution);// para:(binaryImageOfTarget,toolScale,contactResolution)
    imwrite(".\\ModelGenerate\\mapping\\map2.png", result);
    colorRes = ConvertColorResFromBin(result);
    imwrite(".\\ModelGenerate\\heat_map\\heat_map2.png", colorRes);

    maskImage = imread(".\\ModelGenerate\\binaryResource3.png");
    if (maskImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return;
    }
    cvtColor(maskImage, maskImage, CV_BGR2GRAY);//gray image
    maskImage = QuickConvertBin(maskImage, 128);
    result = instance.EAB_IAB_Extraction(maskImage, scale, resolution);// para:(binaryImageOfTarget,toolScale,contactResolution)
    imwrite(".\\ModelGenerate\\mapping\\map3.png", result);
    colorRes = ConvertColorResFromBin(result);
    imwrite(".\\ModelGenerate\\heat_map\\heat_map3.png", colorRes);

#ifdef _WIN32
    system("msg * The contact result is generated in .\\ModelGenerate\\");
#elif __APPLE__
    system("osascript -e 'display dialog \"The contact result is generated in .\\ModelGenerate\\\" buttons {\"OK\"}'");
#elif __linux__
    system("zenity --info --text='The contact result is generated in .\\ModelGenerate\\'");
#endif
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

    originImage = imread("D:\\VCProj\\UnifiedModelDemo\\resource\\origin.hdu");
    if (originImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return;
    }

    // Create a window
    namedWindow("UnifiedModelDisplay", WINDOW_AUTOSIZE);


    // Create trackbars (values ranging from 1 to 100)
    createTrackbar("Contact", "UnifiedModelDisplay", &cntProgress, 100, onTrackbarChange);

    createTrackbar("ToolSize", "UnifiedModelDisplay", &toolSize, 5, onTrackbarChange);

    createTrackbar("Speed", "UnifiedModelDisplay", &delay, 100, onTrackbarChange);

    // Display the interface
    Mat interface(200, 400, CV_8UC3, Scalar(220, 220, 220));
    putText(interface, "CntProgress:0%-100%", Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
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
            Thr = (int)((float)cntProgress * 2.55);
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

// mouse callback of realtime display
void onMouse2(int event, int x, int y, int, void*) {
    if (displayImg.empty()) return;

    Mat displayImage = displayImg.clone();  // clone resource
    Scalar redColor(0, 0, 255); 
    Scalar textColor(0, 0, 255);  

    if (event == EVENT_MOUSEMOVE) {
  
        circle(displayImage, Point(x, y), scale, redColor, 1);

        string pixelValue;
        int pix = (int)binaryImage.at<uchar>(y, x);
        float scale = 255;
        float cntRate = ((float)pix)/scale;
        pixelValue = "Contact Progress: " + to_string((int)(cntRate*100)) + "%";

        // showContactProgress
        putText(displayImage, pixelValue, Point(x + 25, y - 10), FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);

        imshow("Contact Progress", displayImage);
    }
}

void showContactRate() {

    displayImg = imread("./resource/origin.hdu");

    if (displayImg.empty()) {
        cout << "Error: Unable to load image!" << endl;
        return;
    }

    namedWindow("Contact Progress", WINDOW_AUTOSIZE);
    setMouseCallback("Contact Progress", onMouse2, nullptr);
    imshow("Contact Progress", displayImg);

    while (true) {
        int key = waitKey(10);
        if (key == 27) {
            destroyWindow("Contact Progress");
            break;
        }
    }
    
}

void showUserInterfaceofRealtime() {
    // Create a window
    namedWindow("RealtimeDisplay", WINDOW_AUTOSIZE);

    createTrackbar("ToolSize", "RealtimeDisplay", &toolSize, 5, onTrackbarChange);

    // Display the interface
    Mat interface(200, 400, CV_8UC3, Scalar(220, 220, 220));
    putText(interface, "ToolSize:1-5", Point(20, 80), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(interface, "Press 'Enter' to Display , Press 'Esc' to Exit", Point(20, 160), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    imshow("RealtimeDisplay", interface);
    int Thr;
    string src = "./resource/res";

    // Wait for the user to press Enter
    while (true) {
        int key = waitKey(10);
        if (key == 13 || key == 10) {
            Thr = (int)((float)cntProgress * 2.55);
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

            showContactRate();
        }
    }
    cv::destroyWindow("UnifiedModelDisplay");
}

// callback function
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        if (button1.contains(Point(x, y))) {
            showUserInterface();
        }
        else if (button2.contains(Point(x, y))) {
            showUserInterfaceofRealtime();
        }else if (button3.contains(Point(x, y))) {
            EAB_Gen(TOOLSCALE, CNTRESOLUTION);
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
    putText(img, "Real-time Display", button2.tl() + Point(30, 38), FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2);

    rectangle(img, button3, borderColor, 2);
    rectangle(img, button3.tl() + Point(1, 1), button3.br() - Point(1, 1), btnColor, FILLED);
    putText(img, "Model Generator", button3.tl() + Point(30, 38), FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2);


}

void Display() {
    // Load the original image
    originImage = imread("D:\\VCProj\\UnifiedModelDemo\\resource\\origin.hdu");
    if (originImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return;
    }

    Mat menu(300, 350, CV_8UC3);
    drawGradientBackground(menu);  // set background 
    drawButtons(menu);  // set btn
    namedWindow("HDU-Unified Model", WINDOW_AUTOSIZE);
    setMouseCallback("HDU-Unified Model", onMouse, nullptr);
    imshow("HDU-Unified Model", menu);
    while (true) {
        int k = waitKey(20);
        if (k == 27)
            break;
    }
}



int main() {
    
    Display();
    
    //EAB_Gen(TOOLSCALE,CNTRESOLUTION);

    return 0;
}