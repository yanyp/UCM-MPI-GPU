#include <cv.h>
#include <highgui.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include "otherheaders.h"

using namespace cv;
using namespace std;
using namespace std::placeholders;

double applyFx(double x, double sigma) { return exp(-pow(x, 2) / (2 * pow(sigma, 2))); }

double applyFy_case1(double x, double sigma) { return exp(-pow(x, 2) / (2 * pow(sigma, 2))) * (-x / pow(sigma, 2)); }

double applyFy_case2(double x, double sigma) {
    return exp(-pow(x, 2) / (2 * pow(sigma, 2))) * (-pow(x, 2) / (pow(sigma, 2) - 1));
}

Mat oeFilter(double sigma, int support, double theta, int deriv, int hil, int vis) {
    /* calculate filter size */
    int hsz = ceil(support * sigma);
    int sz = 2 * hsz + 1;

    /* sampling limits */
    int maxSamples = 1000;
    int maxRate = 10;
    int frate = 10;

    /* calculate sampling rate and number of samples */
    int rate = min(maxRate, max(1, maxSamples / sz));
    int samples = sz * rate;

    /* The 2D sampling grid */
    double r = sz / 2.0 + 0.5 * (1 - 1.0 / rate);

    vector<double> dom;
    dom.reserve(samples);
    double current = -r;
    double inc = 2.0 * r / static_cast<double>(samples);
    /// FIXME: check mutable above
    generate_n(back_inserter(dom), samples, [current, inc]() mutable { return current += inc; });
    /*
    Mat sx(dom.size(), dom.size(), CV_64FC1, Scalar::all(0));
    Mat sx(sy);   /// FIXME: check the copy constructor for Mat
    if (dom.size() == samples) {
        cout << "yes dom size = samples..good!! = " << samples << endl;
    }
    */
    Mat sx = repeat(Mat(dom), 1, samples);
    // cout << "size of sx = " << sx.size() << endl;
    Mat sy = sx.t();
    // cout << "size of sy = " << sy.size() << endl;
    Mat su = sx * sin(theta) + sy * cos(theta);
    Mat sv = sx * cos(theta) - sy * sin(theta);
    Mat mx(sx.size(), CV_32SC1);
    Mat my(sy.size(), CV_32SC1);
    int* dataMx = (int*)mx.data;
    int* dataMy = (int*)my.data;
    double* dataSu = (double*)su.data;
    double* dataSv = (double*)sy.data;
    for (int i = 0; i < (mx.rows * mx.cols); i++) {
        dataMx[i] = (int)round(dataSu[i]);
        dataMy[i] = (int)round(dataSv[i]);
    }
    Mat membership = (mx + hsz + 1) + (my + hsz) * sz;

    double R = r * sqrt(2) * 1.01;
    int fsamples = ceil(R * rate * frate);
    fsamples += (fsamples + 1) % 2;
    vector<double> fdom;
    fdom.reserve(fsamples);
    double fcurrent = -R;
    double finc = 2.0 * R / static_cast<double>(fsamples);
    /// FIXME: check
    generate_n(back_inserter(fdom), fsamples, [fcurrent, finc]() mutable { return fcurrent += finc; });
    double gap = 2.0 * R / ((double)fsamples - 1.0);

    vector<double> fx, fy;
    fx.resize(fdom.size());
    fy.resize(fdom.size());

    transform(fdom.begin(), fdom.end(), fx.begin(), bind(applyFx, _1, sigma));

    switch (deriv) {
        case 1:
            transform(fdom.begin(), fdom.end(), fy.begin(), bind(applyFy_case1, _1, sigma));
            break;
        case 2:
            transform(fdom.begin(), fdom.end(), fy.begin(), bind(applyFy_case2, _1, sigma));
            break;
        default:
            log_error("Derivative case wrong, can either be 1 or 2, found none");
            break;  /// FIXME: check cerr and endl
    }
    /*
    cout << "size of fx vector = " << fx.size() << endl;
    cout << "size of fy vector = " << fy.size() << endl;
    cout << "size of fdom = " << fdom.size() << endl;
    cout << "fsamples = " << fsamples << endl;
    */
    Mat xi(su.size(), CV_32S);
    Mat yi(sv.size(), CV_32S);
    int* dataXi = (int*)xi.data;
    int* dataYi = (int*)yi.data;
    for (int i = 0; i < xi.rows * yi.cols; i++) {
        dataXi[i] = (int)round(dataSu[i] / gap);
        dataYi[i] = (int)round(dataSv[i] / gap);
    }
    xi = xi + floor(fsamples / 2) + 1;
    yi = yi + floor(fsamples / 2) + 1;
    Mat f(su.size(), CV_64FC1, Scalar::all(0));
    /*
    double minXi, maxXi, minYi, maxYi;
    minMaxLoc(xi, &minXi, &maxXi);
    minMaxLoc(yi, &minYi, &maxYi);
    cout << "minXi = "<<minXi<< ", MaxXi = "<< maxXi << ", minYi = "<<minYi<< ", maxYi = "<<maxYi<<", " <<
    endl;

    minMaxLoc(sx, &minXi, &maxXi);
    minMaxLoc(sy, &minYi, &maxYi);
    cout << "minSx = "<<minXi<< ", MaxSx = "<< maxXi << ", minSy = "<<minYi<< ", maxSy = "<<maxYi<<", " <<
    endl;

    minMaxLoc(su, &minXi, &maxXi);
    minMaxLoc(sv, &minYi, &maxYi);
    cout << "minSu = "<<minXi<< ", MaxSu = "<< maxXi << ", minSv = "<<minYi<< ", maxSv = "<<maxYi<<", " <<
    endl;

    cout << "size of f = " << f.size() << endl;
    cout << "val of sz  = " << sz << endl;
    */
    for (int i = 0; i < su.rows; i++)
        for (int j = 0; j < su.cols; j++) f.at<double>(i, j) = fx.at(xi.at<int>(i, j)) * fy.at(yi.at<int>(i, j));
    /*
    cout << f.at<double>(0,0) << ", "<< f.at<double>(1,1) << ", " << f.at<double>(su.rows-1,su.cols-1) <<
    endl;

    cout << fx.at(xi.at<int>(0,0)) << ", "<< fx.at(xi.at<int>(1,1)) << ", " <<
    fx.at(xi.at<int>(su.rows-1,su.cols-1)) << endl;
    */

    /* accumulate samples into each bin */
    Mat acc(sz, sz, CV_64FC1);
    int numel = sz * sz;
    int* memData = (int*)membership.data;
    double* accData = (double*)acc.data;
    double* fData = (double*)f.data;
    for (int i = 0; i < numel; i++) {
        if (memData[i] < 0) continue;
        if (memData[i] > sz * sz) continue;
        accData[memData[i]] += fData[i];
    }

    /*
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            cout << acc.at<double>(i,j) << "\t";
        }
        cout << endl;
    }
    cout << endl;
    */

    /// FIXME: .val[0] or just [0]
    double meanF = mean(acc).val[0];
    acc = acc - meanF;

    /*
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            cout << acc.at<double>(i,j) << "\t";
        }
        cout << endl;
    }
    cout << endl;
    */

    double sumF = sum(abs(acc)).val[0];
    acc = acc / sumF;

    /* print filter values */
    /*
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            cout << acc.at<double>(i,j) << "\t";
        }
        cout << endl;
    }
    cout << endl;
    cout << "summed Filter = " << sum(abs(acc)).val[0] << endl;
    */

    return acc;
}
