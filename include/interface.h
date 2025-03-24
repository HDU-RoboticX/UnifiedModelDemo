#pragma once
#include <opencv2/opencv.hpp>//用于图像读取和显示及写入，二值图的预处理
#include <iostream>//用于输出调试信息



#define MATHLIBRARY_API __declspec(dllexport)

class MATHLIBRARY_API Interface
{
public:
	cv::Mat EAB_IAB_Extraction(cv::Mat img, int size, int grade);
}; 