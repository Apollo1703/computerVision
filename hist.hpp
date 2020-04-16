#include <iostream>
#include <opencv2/core/core_c.h>
#include <opencv2/opencv.hpp>
// #include <hist.hpp>

void drawHist(cv::Mat frameGray)
{
    int histSize = 256;
    float range[257] = {0, 255};
    const float *histRange = {range};
    bool uniform = true, accumulate = false;

    cv::Mat grayHist;

    cv::calcHist(&frameGray, 1, 0, cv::Mat(), grayHist, 1, &histSize, &histRange, uniform, accumulate);
    int histW = 600, histH = 600;
    int bin_w = cvRound((double)histW / histH);
    cv::Mat histImage(histH, histW, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(grayHist, grayHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point(bin_w * (i - 1), histH - cvRound(grayHist.at<float>(i - 1))), cv::Point(bin_w * (i), histH - cvRound(grayHist.at<float>(i))), cv::Scalar(255, 255, 255), 2, 8, 0);
    }
    cv::imshow("Histogram", histImage);
}