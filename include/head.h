#ifndef HEAD_H_
#define HEAD_H_

#include <cv.h>
#include <highgui.h>
#include <cmath>

#include <fstream>
#include <iostream>
#include <string>

#define M_PI 3.14159265358979323846
// #define max(a,b) (((a)>(b)) ? (a):(b))
// #define min(a,b) (((a)<(b)) ? (a):(b))

#ifndef coordinates_
#define coordinates_
struct coordinates {
    int row;
    int col;
};
#endif

struct superContour4COutput {
    cv::Mat pb2;
    cv::Mat V;
    cv::Mat H;
    std::vector<coordinates> loc;
};

struct FitContourOutput {
    cv::Mat vertices;
    cv::Mat edges;
    std::vector<cv::Mat> edge_x_coords;
    std::vector<cv::Mat> edge_y_coords;
    cv::Mat is_completion;
};

struct RegionpropsObject {
    std::vector<coordinates> PixelList;
};

void Imsave(std::string filename, cv::Mat& matrix);
void ImsaveScale(std::string filename, cv::Mat& matrix);
cv::Mat Contours2Ucm(std::string pieceFilePat, std::vector<cv::Mat>& pb_oriented, std::string fmt);

/* fit_contour.cpp */
FitContourOutput fit_contour(cv::Mat&);
void MexContourSides(FitContourOutput& contours, cv::Mat& nmax);

/* MATLAB_functions.cpp */
std::vector<RegionpropsObject> regionprops(cv::Mat, std::string);
cv::Mat bwmorph(cv::Mat&, std::string, int);
void bwlabelCore(int, int, int, cv::Mat&, int);
cv::Mat bwlabel(cv::Mat&, int);
int bwlabel(cv::Mat& labels, cv::Mat& BW, int numNeighbor);

cv::Mat UcmMeanPb(const cv::Mat& ws_wt2, const cv::Mat& labels);

/* watershed.cpp */
cv::Mat Matrix2Image(cv::Mat& Pb_8U);
std::vector<coordinates> find_8U(cv::Mat& matrix, int level);
cv::Mat watershed_yupeng(cv::Mat& image, int conn);

#endif
