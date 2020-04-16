#include <iostream>
#include <opencv2/core/core_c.h>
#include <opencv2/opencv.hpp>
// #include <hist.hpp
using namespace cv;
class hist
{
private:
    /* data */
public:
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
    void drawHistRGB(cv::Mat frame)
    {
        cv::Mat dst;
        /// Separate the image in 3 places ( B, G and R )
        std::vector<Mat> bgr_planes;
        split(frame, bgr_planes);

        /// Establish the number of bins
        int histSize = 256;

        /// Set the ranges ( for B,G,R) )
        float range[] = {0, 256};
        const float *histRange = {range};

        bool uniform = true;
        bool accumulate = false;

        Mat b_hist, g_hist, r_hist;

        /// Compute the histograms:
        cv::calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
        cv::calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

        // Draw the histograms for B, G and R
        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound((double)hist_w / histSize);

        cv::Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

        /// Normalize the result to [ 0, histImage.rows ]
        cv::normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        cv::normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
        cv::normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

        /// Draw for each channel
        for (int i = 1; i < histSize; i++)
        {
            line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                 Scalar(255, 0, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                 Scalar(0, 255, 0), 2, 8, 0);
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                 Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                 Scalar(0, 0, 255), 2, 8, 0);
        }

        /// Display
        namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
        imshow("calcHist Demo", histImage);

        waitKey(0);
    }
};

class edgeDetection
{
private:
    /* data */
public:
    edgeDetection(/* args */);
    ~edgeDetection();
    void prewitt(cv::Mat frame);
    void gradient(cv::Mat frame);
    void robert(cv::Mat frame);
};

edgeDetection::edgeDetection(/* args */)
{
}

edgeDetection::~edgeDetection()
{
}

void edgeDetection::prewitt(cv::Mat frame)
{
    cv::Mat dstX, dstY, dstXY;
    cv::Mat kernelX = (cv::Mat_<int>(3, 3) << 1, 0, -1, 1, 0, -1, 1, 0, -1);
    cv::Mat kernelY = (cv::Mat_<int>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
    cv::filter2D(frame, dstX, -1, kernelX);
    cv::filter2D(frame, dstY, -1, kernelY);
    cv::bitwise_or(dstX, dstY, dstXY);
    cv::imshow("Prewitt Combine", dstXY);
}

void edgeDetection::robert(cv::Mat frame)
{
    cv::Mat dstX, dstY, dstXY;
    cv::Mat kernelX = (cv::Mat_<int>(2, 2) << 1, 0, 0, -1);
    cv::Mat kernelY = (cv::Mat_<int>(2, 2) << 0, 1, -1, 0);
    cv::filter2D(frame, dstX, -1, kernelX);
    cv::filter2D(frame, dstY, -1, kernelY);
    cv::bitwise_or(dstX, dstY, dstXY);
    cv::imshow("Robert Combine", dstXY);
}

void edgeDetection::gradient(cv::Mat frame)
{
    cv::Mat dstX, dstY, dstXY;
    cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<int>(3, 1) << -1, 0, 1);
    cv::filter2D(frame, dstX, -1, kernelX);
    cv::filter2D(frame, dstY, -1, kernelY);
    cv::bitwise_or(dstX, dstY, dstXY);
    cv::imshow("Gradient Method", dstXY);
}
