#include <cv.h>
#include <highgui.h>
#include <algorithm>
#include <cmath>
#include <ctime>
// #include "detmpb.h"
#include "otherheaders.h"

using namespace cv;
using namespace std;

void make_filters(std::vector<std::vector<Mat> > &filters, const std::vector<int> &radii,
                  const std::vector<double> &gtheta);
void cumulativeProduct(std::vector<double> &cumProd, const std::vector<double> &a);

void multiscalePb(cv::Mat *&mPb_nmax, cv::Mat *&mPb_nmax_rsz, detmpb &detmpb_obj, Mat &im, Size orig_size, int nChan,
                  std::string outFile, double rsz) {
    std::vector<double> weights(13);
    if (nChan == 3) {
        weights = {0.0146, 0.0145, 0.0163, 0.0210, 0.0243, 0.0287, 0.0166, 0.0185, 0.0204, 0.0101, 0.0111, 0.0141};
    }
    /* check im */
    /* put all three channels equal or make sure to read imread, 1 */
    else {
        weights = {0.0245, 0.0220, 0.0233, 0, 0, 0, 0, 0, 0, 0.0208, 0.0210, 0.0229};
    }

    /* get gradients */
    /* insert tic toc and print times */
    time_t now1, now2;
    time(&now1);
    det_mPb(detmpb_obj, im);
    time(&now2);
    log_info("Time for Local cues = %ld", now2 - now1);

    std::vector<double> gtheta = {1.5708, 1.1781, 0.7854, 0.3927, 0, 2.7489, 2.3562, 1.9635};
    std::vector<int> radii = {3, 5, 10, 20};

    /*
    std::vector<std::vector<Mat>> filters;
    filters.resize(numRadii, std::vector<Mat> numGtheta);   //check here
    make_filters(filters, radii, gtheta);
    */

    std::vector<cv::Mat *> *bg1 = NULL, *bg2 = NULL, *bg3 = NULL, *cga1 = NULL, *cga2 = NULL, *cga3 = NULL,
                           *cgb1 = NULL, *cgb2 = NULL, *cgb3 = NULL, *tg1 = NULL, *tg2 = NULL, *tg3 = NULL;
    cv::Mat *textons = NULL;

    bg1 = detmpb_obj.GetBg1();
    bg2 = detmpb_obj.GetBg2();
    bg3 = detmpb_obj.GetBg3();

    cga1 = detmpb_obj.GetCga1();
    cga2 = detmpb_obj.GetCga2();
    cga3 = detmpb_obj.GetCga3();

    cgb1 = detmpb_obj.GetCgb1();
    cgb2 = detmpb_obj.GetCgb2();
    cgb3 = detmpb_obj.GetCgb3();

    tg1 = detmpb_obj.GetTg1();
    tg2 = detmpb_obj.GetTg2();
    tg3 = detmpb_obj.GetTg3();

    textons = detmpb_obj.GetTextons();

    int nOrient = tg1->size();

    std::string imageName;
    double alpha, beta, minXi, maxXi, minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::Mat destIm(textons->size(), CV_64FC1);

    /*
    for (int i = 0; i < nOrient; i++) {
        imageName = "image_bg1_" + to_string(i) + ".png";
        minMaxLoc(*bg1->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        bg1->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_bg2_" + to_string(i) + ".png";
        minMaxLoc(*bg2->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        bg2->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_bg3_" + to_string(i) + ".png";
        minMaxLoc(*bg3->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        bg3->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_cga1_" + to_string(i) + ".png";
        minMaxLoc(*cga1->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        cga1->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_cga2_" + to_string(i) + ".png";
        minMaxLoc(*cga2->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        cga2->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_cga3_" + to_string(i) + ".png";
        minMaxLoc(*cga3->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        cga3->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_cgb1_" + to_string(i) + ".png";
        minMaxLoc(*cgb1->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        cgb1->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_cgb2_" + to_string(i) + ".png";
        minMaxLoc(*cgb2->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        cgb2->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);
        //
        imageName = "image_cgb3_" + to_string(i) + ".png";
        minMaxLoc(*cgb3->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        cgb3->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_tg1_" + to_string(i) + ".png";
        minMaxLoc(*tg1->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        tg1->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_tg2_" + to_string(i) + ".png";
        minMaxLoc(*tg2->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        tg2->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);

        imageName = "image_tg3_" + to_string(i) + ".png";
        minMaxLoc(*tg3->at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        alpha = 255.0 / (maxVal - minVal);
        beta = -minVal * 255.0 / (maxVal - minVal);
        tg3->at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);
    }
    imageName = "image_textons.png";
    minMaxLoc(*textons, &minVal, &maxVal, &minLoc, &maxLoc);
    alpha = 255.0 / (maxVal - minVal);
    beta = -minVal * 255.0 / (maxVal - minVal);
    textons->convertTo(destIm, CV_8U, alpha, beta);
    imwrite(imageName, destIm);

    Mat *a = NULL;
    for (int o = 0; o < nOrient; o++) {
        a = fitparab(bg1->at(o), 3.0, 3.0 / 4.0, gtheta.at(o), filters.at(0).at(o));

        delete bg1->at(o);
        bg1->at(o) = a;  // FIXME: i think deleting early position of vector should be ok!!??

        a = fitparab(bg2->at(o), 5.0, 5.0 / 4.0, gtheta.at(o), filters.at(1).at(o));
        delete bg2->at(o);
        bg2->at(o) = a;
        a = fitparab(bg3->at(o), 10.0, 10.0 / 4.0, gtheta.at(o), filters.at(2).at(o));
        delete bg3->at(o);
        bg3->at(o) = a;

        a = fitparab(cga1->at(o), 5.0, 5.0 / 4.0, gtheta.at(o), filters.at(1).at(o));
        delete cga1->at(o);
        cga1->at(o) = a;
        a = fitparab(cga2->at(o), 10.0, 10.0 / 4.0, gtheta.at(o), filters.at(2).at(o));
        delete cga2->at(o);
        cga2->at(o) = a;
        a = fitparab(cga3->at(o), 20.0, 20.0 / 4.0, gtheta.at(o), filters.at(3).at(o));
        delete cga3->at(o);
        cga3->at(o) = a;

        a = fitparab(cgb1->at(o), 5.0, 5.0 / 4.0, gtheta.at(o), filters.at(1).at(o));
        delete cgb1->at(o);
        cgb1->at(o) = a;
        a = fitparab(cgb2->at(o), 10.0, 10.0 / 4.0, gtheta.at(o), filters.at(2).at(o));
        delete cgb2->at(o);
        cgb2->at(o) = a;
        a = fitparab(cgb3->at(o), 20.0, 20.0 / 4.0, gtheta.at(o), fildetmpbters.at(3).at(o));
        delete cgb3->at(o);
        cgb3->at(o) = a;

        a = fitparab(tg1->at(o), 5.0, 5.0 / 4.0, gtheta.at(o), filters.at(1).at(o));
        delete tg1->at(o);
        tg1->at(o) = a;
        a = fitparab(tg2->at(o), 10.0, 10.0 / 4.0, gtheta.at(o), filters.at(2).at(o));
        delete tg2->at(o);
        tg2->at(o) = a;
        a = fitparab(tg3->at(o), 20.0, 20.0 / 4.0, gtheta.at(o), filters.at(3).at(o));
        delete tg3->at(o);
        tg3->at(o) = a;

    }
    */

    /// TODO: put fprintf and toc here

    /* compute mPb at full scale */
    std::vector<cv::Mat *> mPb_all(nOrient);
    orig_size = tg1->at(0)->size();     /// TODO: size vs size() -- check??
    cv::Mat *dataMat;
    for (int o = 0; o < nOrient; o++) {
        dataMat = new cv::Mat(orig_size, CV_64FC1);         /// TODO: C1 or not??
        // mPb_all.at(o) = new Mat(orig_size, CV_64FC1);    /// TODO: C1 or not??
        for (int r = 0; r < orig_size.height; r++)
            for (int c = 0; c < orig_size.width; c++) {
                dataMat->at<double>(r, c) =
                    weights.at(0) * (bg1->at(o)->at<double>(r, c)) +
                    weights.at(1) * (bg2->at(o)->at<double>(r, c)) +    /// TODO: check precedence
                    weights.at(2) * (bg3->at(o)->at<double>(r, c)) + weights.at(3) * (cga1->at(o)->at<double>(r, c)) +
                    weights.at(4) * (cga2->at(o)->at<double>(r, c)) + weights.at(5) * (cga3->at(o)->at<double>(r, c)) +
                    weights.at(6) * (cgb1->at(o)->at<double>(r, c)) + weights.at(7) * (cgb2->at(o)->at<double>(r, c)) +
                    weights.at(8) * (cgb3->at(o)->at<double>(r, c)) + weights.at(9) * (tg1->at(o)->at<double>(r, c)) +
                    weights.at(10) * (tg2->at(o)->at<double>(r, c)) + weights.at(11) * (tg3->at(o)->at<double>(r, c));
            }

        mPb_all.at(o) = dataMat;
        /// @Yupeng comment
        /*
        imageName = outFile + "_mPb_i_" + to_string(o) + ".png";
        minMaxLoc(*dataMat, &minXi, &maxXi);
        alpha = 255.0 / (maxXi - minXi);
        beta = -minXi * 255.0 / (maxXi - minXi);
        dataMat->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);
        */
    }

    // Mat* mPb_nmax = NULL;

    mPb_nmax = nonmax_channels(mPb_all, PI / 8);

    double *data;
    for (int r = 0; r < mPb_nmax->rows; r++) {
        for (int c = 0; c < mPb_nmax->cols; c++) {
            if (1.2 * mPb_nmax->at<double>(r, c) > 1) {
                mPb_nmax->at<double>(r, c) = 1;
            }
            if (mPb_nmax->at<double>(r, c) < 0) {
                mPb_nmax->at<double>(r, c) = 0;
            }
        }
    }

    /// TODO: compute mPb_nmax resized if necessary
    // Mat* mPb_nmax_rsz = NULL;
    /// TODO: recalculate using imresize equivalent
    if (rsz < 1) {
        mPb_nmax_rsz = new Mat(mPb_nmax->size(), CV_64FC1);
    }
    else {
        mPb_nmax_rsz = mPb_nmax;
    }
    /// @Yupeng comment
    /*
    imageName = outFile + "_mPb_nMax.png";
    minMaxLoc(*mPb_nmax_rsz, &minXi, &maxXi);
    alpha = 255.0 / (maxXi - minXi);
    beta = -minXi*255.0 / (maxXi - minXi);
    mPb_nmax_rsz->convertTo(destIm, CV_8U, alpha, beta);
    imwrite(imageName, destIm);
    */

    // Mat destIm(mPb_nmax->size(), CV_64FC1);
}

void make_filters(std::vector<std::vector<Mat> > &filters, const std::vector<int> &radii,
                  const std::vector<double> &gtheta) {
    int numGtheta = gtheta.size();
    int numRadii = radii.size();
    filters.resize(numRadii, std::vector<Mat>(numGtheta));

    double ra = 0.0, rb = 0.0, theta = 0.0, ira2 = 0.0, irb2 = 0.0, sint = 0.0, cost = 0.0, ai = 0.0, bi = 0.0;
    int wr = 0, d = 2, wd = 0;
    std::vector<double> xx;
    std::vector<double> yy;
    std::vector<double> aa;
    std::vector<double> bb;
    std::vector<double> cumProd;

    /// TODO: check type and val
    double *data = NULL;
    /*
    Mat* A = NULL;
    Mat* b = NULL;
    Mat* sol = NULL;
    Mat* filt = NULL;
    */
    int *sizes = NULL;

    for (int r = 0; r < numRadii; r++)
        for (int t = 0; t < numGtheta; t++) {
            ra = double(radii.at(r));
            rb = ra / 4;
            theta = gtheta.at(t);
            ra = std::max(1.5, ra);
            rb = std::max(1.5, rb);
            ira2 = std::pow(1.0 / ra, 2);   /// TODO: does 2 need to be double??
            irb2 = std::pow(1.0 / rb, 2);
            wr = std::floor(std::max(ra, rb));
            wd = 2 * wr + 1;
            sizes = new int[3]{wd, wd, d + 1};
            sint = std::sin(theta);
            cost = std::cos(theta);

            /// TODO: check type and val
            Mat filt(3, sizes, CV_64F, Scalar::all(1));

            xx.assign(2 * d + 1, 0);
            // cumProd.clear() ;
            for (int u = -wr; u < wr; u++)
                for (int v = -wr; v < wr; v++) {
                    ai = -u * sint + v * cost;
                    bi = u * cost + v * sint;
                    if (ai * ai * ira2 + bi * bi * irb2 > 1) {
                        continue;
                    }
                    aa.assign(2 * d + 1, ai);
                    aa.at(0) = 1;

                    cumulativeProduct(cumProd, aa);
                    xx += cumProd;      // TODO: check vector addition
                }

            Mat A(d + 1, d + 1, CV_64F, Scalar::all(0));
            for (int i = 0; i < d + 1; i++) {
                data = (double*) A.ptr(i);
                for (int j = i; j < i + d + 1; j++) {
                    data[j] = xx.at(j);
                }
            }

            for (int u = -wr; u < wr; u++)
                for (int v = -wr; v < wr; v++) {
                    ai = -u * sint + v * cost;
                    bi = u * cost + v * sint;
                    if (ai * ai * ira2 + bi * bi * irb2 > 1) {
                        continue;
                    }
                    bb.assign(d + 1, ai);
                    bb.at(0) = 1;
                    cumulativeProduct(yy, bb);

                    Mat b(d + 1, 1, CV_64F, yy.data());     /// TODO: check C++11
                    Mat sol(d + 1, 1, CV_64F);
                    solve(A, b, sol, DECOMP_NORMAL);        /// TODO: check this pointer thing and bool return
                    /*
                    for (int k = 0 ; k < d+1 ; k++) {
                        filt->ptr(u+wr+1,v+wr+1,k) = sol->ptr(k);
                        // filt->ptr(u+wr+1,v+wr+1) = sol->ptr();   // check this or above
                    }
                    */
                    for (int k = 0; k < d + 1; k++) {
                        filt.at<double>(u + wr + 1, v + wr + 1, k) = sol.at<double>(k);
                    }
                }
            filters.at(r).at(t) = filt;
        }
}

void cumulativeProduct(std::vector<double> &cumProd, const std::vector<double> &a) {
    cumProd.assign(a.size(), 1);
    for (int p = 0; p < a.size(); p++) {
        for (int q = 0; q < p; q++) {
            cumProd.at(p) *= a.at(q);
        }
    }
}
