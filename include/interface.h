#pragma once
#include <opencv2/opencv.hpp>//����ͼ���ȡ����ʾ��д�룬��ֵͼ��Ԥ����
#include <iostream>//�������������Ϣ



#define MATHLIBRARY_API __declspec(dllexport)

class MATHLIBRARY_API Interface
{
public:
	cv::Mat EAB_IAB_Extraction(cv::Mat img, int size, int grade);
}; 