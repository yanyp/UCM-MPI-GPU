#include <cv.h>
#include <highgui.h>
#include <limits>
#include "otherheaders.h"

using namespace cv;

Mat* nonmax_channels(std::vector<Mat*> mPb_all, double nonmax_ori_tol) {
    Size orig_size = mPb_all.at(0)->size();     /// TODO: size or size()
    Mat* a = new Mat(orig_size, CV_64FC1);      /// TODO: C1 or not??
    Mat* ind = new Mat(orig_size, CV_64FC1);
    int n_ori = mPb_all.size();
    vector<double> oris(n_ori);
    for (int i = 0; i < n_ori; i++) oris.at(i) = (double)i * PI / (double)n_ori;

    double data;
    double maxVal = std::numeric_limits<double>::min();  /// TODO: check whether 0 will suffice
    int maxValIndex = 0;

    /// TODO: change this below to binary search instead of linear search   // not required
    for (int r = 0; r < a->rows; r++) {
        for (int c = 0; c < a->cols; c++) {
            for (int i = 0; i < n_ori; i++) {
                data = mPb_all.at(i)->at<double>(r, c);
                if (data > maxVal) {
                    maxVal = data;
                    maxValIndex = i;
                }
            }
            if (maxVal < 0) maxVal = 0;
            a->at<double>(r, c) = maxVal;
            ind->at<double>(r, c) = oris.at(maxValIndex);
            maxVal = std::numeric_limits<double>::min();
        }
    }

    std::string imageName;
    double alpha, beta, minXi, maxXi;
    Point minLoc, maxLoc;
    Mat destIm(a->size(), CV_64FC1);

    /*
    for (int i = 0; i < n_ori; i++) {
        imageName = "image_mPb1_" + std::to_string(i) + ".png";
        minMaxLoc(*mPb_all.at(i), &minXi, &maxXi);
        alpha = 255.0 / (maxXi - minXi);
        beta = -minXi * 255.0 / (maxXi - minXi);
        mPb_all.at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);
    }
    */

    /// FIXME: generating this image is probably unneccesary and commented for now or get filename as parameter
    /*
    imageName = "image_max_a.png";
    minMaxLoc(*a, &minXi, &maxXi);
    alpha = 255.0 / (maxXi - minXi);
    beta = -minXi * 255.0 / (maxXi - minXi);
    a->convertTo(destIm, CV_8U, alpha, beta);
    imwrite(imageName, destIm);
    */

    Mat* nmax = NULL;
    nonmax_oriented(nmax, a, ind, nonmax_ori_tol);

    /*
    generating this image is probably unneccesary and commented for now or get filename as parameter
    imageName = "image_nmax.png" ;
    minMaxLoc(*nmax, &minXi, &maxXi);
    alpha = 255.0 / (maxXi - minXi);
    beta = -minXi * 255.0 / (maxXi - minXi);
    nmax->convertTo(destIm, CV_8U, alpha, beta);
    imwrite(imageName, destIm);
    */

    delete a;
    delete ind;

    return nmax;
}
