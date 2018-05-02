#include <cv.h>
#include <highgui.h>
#include "otherheaders.h"

using namespace cv;
using namespace std;

Mat* fitparab(cv::Mat* z, double ra, double rb, double theta, cv::Mat* filt) {
    Mat* a = NULL;
    Size orig_size = z->size();         /// TODO: check size() vs size
    a = new Mat(orig_size, CV_64FC1);   /// TODO: check whether to putC1 or not and unify
    // filter2D(*z, *a, -1, *filt, (-1,-1), 0, BORDER_DEFAULT);
    filter2D(*z, *a, -1, *filt);

    /*
    string imageName;
    double alpha, beta, minVal, maxVal;
    Point minLoc, maxLoc;
    Mat destIm(a->size(), CV_64FC1);

    imageName = "image_a_" + to_string(ra) + "_" + to_string(rb) + "_" + to_string(theta) + ".png";
    minMaxLoc(*a, &minVal, &maxVal, &minLoc, &maxLoc);
    alpha = 255.0 / (maxVal - minVal);
    beta = -minVal * 255.0 / (maxVal - minVal);
    a->convertTo(destIm, CV_8U, alpha, beta);
    imwrite(imageName, destIm);
    */

    Mat* b = new Mat(orig_size, CV_64FC1);
    savgol_border(b, a, z, ra, rb, theta);

    delete a;
    a = NULL;

    return b;
}
