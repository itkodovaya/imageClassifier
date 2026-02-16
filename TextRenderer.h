#ifndef TEXT_RENDERER_H
#define TEXT_RENDERER_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

// Функция для получения размера русского текста
Size getTextSizeRussian(const string& text, double fontScale, int thickness);

// Функция для отображения русского текста с использованием Windows API
void putTextRussian(Mat& img, const string& text, Point pos, int fontFace, 
                   double fontScale, Scalar color, int thickness = 1);

#endif

