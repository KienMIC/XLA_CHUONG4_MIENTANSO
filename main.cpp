//IDEAL LOW PASS FILTER
#include <opencv2/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//IDEAL LOW PASS FILTER - CREATE KERNEL
enum FILTERTYPE { IDEAL_LP = 0,BUTTERWORTH_LP = 1,GAUSSIAN_LP=2 };
void createLowPassFilter(int m, int n, int D0, int degree, Mat& filter, FILTERTYPE ft) 
{
    Point center = Point(n / 2, m / 2);
    filter = Mat::zeros(m, n, CV_32F);
    float D02 = D0 * D0;
    switch (ft)
    {
    case IDEAL_LP:
        circle(filter, center, D0, Scalar(1), -1, 0, 0);
        break;
    case BUTTERWORTH_LP:
        for(int u=0;u<m;u++)
            for (int v = 0; v < n; v++) {
                float du = u - m / 2.0;
                float dv = v - n / 2.0;
                float Duv2 = du * du + dv * dv;
                filter.at<float>(u, v) = 1.0f / (1 + pow(Duv2 / D02, degree));
            }
        break;
    case GAUSSIAN_LP:

        break;
    default:
        break;
    }
}
void ShiftDFT(Mat &fImage) {
    Mat tmp, q0, q1, q2, q3;
    fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));
    int cx = fImage.cols / 2;
    int cy = fImage.rows / 2;

    q0 = fImage(Rect(0, 0, cx, cy));
    q1 = fImage(Rect(cx, 0, cx, cy));
    q2 = fImage(Rect(0, cy, cx, cy));
    q3 = fImage(Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

 }
void DFTFilter(Mat& input, Mat& filter, Mat& result) {
    Mat padded, filter2;
    Mat mergefilter[] = { filter,filter };
    cv::merge(mergefilter, 2, filter2);
    int m = getOptimalDFTSize(input.rows);
    int n = getOptimalDFTSize(input.cols);
    copyMakeBorder(input, padded,0, m - input.rows, n - input.cols, BORDER_CONSTANT, 0);
    Mat planes[] = {Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F)};
    Mat complexI;
    cv::merge(planes, 2, complexI);
    dft(complexI, complexI);
    ShiftDFT(complexI);
    mulSpectrums(complexI, filter2, complexI,0);
    ShiftDFT(complexI);
    idft(complexI, complexI, DFT_SCALE);
    split(complexI, planes);
    result = planes[0](Rect(0, 0, input.cols, input.rows));
    normalize(result, result, 0, 1, NORM_MINMAX);

}
int main(int argc, char* argv[])
{
    Mat im1 = imread("D://A10.jpg",0);
    Mat im_gauss;
    if (im1.data == nullptr)
    {
        cout << "Loi khi doc file " << argv[1];
        return 1;
    }
    int m = getOptimalDFTSize(im1.rows);
    int n = getOptimalDFTSize(im1.cols);
    Mat filter, result,filter1,result1;
    int D0 = 200;
    int deg = 2;
    imshow("Input Image", im1);
    //LOW PASS FILTER

    //IDEAL
    createLowPassFilter(m, n, D0, deg, filter, IDEAL_LP);
    imshow("Ideal Lowpass Kernel", filter);
    DFTFilter(im1, filter, result);
    imshow("Ideal Lowpass Filter Result", result);
    imwrite("IdealLowPass.jpg", result);
    //BUTTERWORTH
    createLowPassFilter(m, n, D0, deg, filter1, BUTTERWORTH_LP);
    imshow("Butterworth Lowpass Kernel", filter1);
    DFTFilter(im1, filter1, result1);
    imshow("Butterworth Lowpass Filter Result", result1);
    imwrite("ButtterWorthLowPass.jpg", result1);
    //GAUSSIAN
    GaussianBlur(im1, im_gauss, Size(3, 3), 1.0);
    imshow("Gaussian Lowpass Filter Result", im_gauss);
    imwrite("GaussianLowPass.jpg", im_gauss);

    //HIGH PASS FILTER
    imshow("Ideal Highpass Filter Result", 1-result);
    imwrite("IdealHighPass.jpg", 1 - result);

    imshow("Butterworth Highpass Filter Result", 1 - result1);
    imwrite("ButterworthHighPass.jpg", 1 - result1);

    imshow("Gaussian Highpass Filter Result", 1-im_gauss);
    imwrite("GaussianHighPass.jpg", 1-im_gauss);

    waitKey(0);
    return 0;
}

