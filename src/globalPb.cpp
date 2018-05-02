#include <cv.h>
#include <highgui.h>
#include <fstream>

#include "otherheaders.h"

/* Damascene */
#include <globalPb.h>
#include <stdint.h>
#include <cstdio>

using namespace cv;
using namespace std;

void GlobalPb(String imgFile, String outFile, vector<cv::Mat>& gPb, double rsz) {
    time_t nowd1, nowd2, nowd3, nowd4, nowd5;
    time(&nowd1);

    Mat im = imread(imgFile);
    Mat imd;
    im.convertTo(imd, CV_64FC3, 1.0 / 255.0);   /// TODO: check

    int nChan = imd.channels();     /// TODO: check
    Size origSize = imd.size();     /// TODO: check

    /// TODO: @Yupeng - read the variable from file if already generated

    detmpb detmpbObj;
    Mat* mPbRsz = NULL;
    vector<double> weights(13);
    if (nChan == 3) {
        weights = {0, 0, 0.0039, 0.0050, 0.0058, 0.0069, 0.0040, 0.0044, 0.0049, 0.0024, 0.0027, 0.0170, 0.0074};
    }
    else {
        weights = {0, 0, 0.0054, 0, 0, 0, 0, 0, 0, 0.0048, 0.0049, 0.0264, 0.0090};
    }

    /// TODO: @Yupeng - fix below lines
    string mPbFilename = outFile + "_mPb.yml";
    ifstream infile(mPbFilename);
    time(&nowd2);
    if (infile.good()) {
        // cv::FileStorage fs(yupeng_filename, cv::FileStorage::READ);
        cv::FileStorage fs(mPbFilename, cv::FileStorage::READ);
        cv::Mat mPbRszMat;
        fs["mPb_rsz_mat"] >> mPbRszMat;
        fs.release();
        mPbRsz = &(mPbRszMat);
        // detmpbObj.loadObject(outFile);
    } else {
        Mat* mPbNmax = NULL;
        /* rsz not included, should take default 1.0 */
        multiscalePb(mPbNmax, mPbRsz, detmpbObj, imd, origSize, nChan, outFile);
        // detmpbObj.SaveObject(outFile);

        cv::FileStorage fs(mPbFilename, cv::FileStorage::WRITE);
        fs << "mPb_rsz_mat" << *mPbRsz;
        // fs << "detmpb_obj_pb" << detmpbObj;
        fs.release();
    }
    time(&nowd3);

    // String outFile2 = outFile + "_pbs.mat";
    vector<cv::Mat*> sPb;
    int nvec = 17;
    spectralPb(sPb, mPbRsz, origSize, outFile, nvec);
    time(&nowd4);

    /// TODO: @Yupeng - repeating in multiscalePb.cpp
    vector<cv::Mat *> *bg1 = NULL, *bg2 = NULL, *bg3 = NULL, *cga1 = NULL, *cga2 = NULL, *cga3 = NULL,
                           *cgb1 = NULL, *cgb2 = NULL, *cgb3 = NULL, *tg1 = NULL, *tg2 = NULL, *tg3 = NULL;
    cv::Mat* textons = NULL;

    bg1 = detmpbObj.GetBg1();
    bg2 = detmpbObj.GetBg2();
    bg3 = detmpbObj.GetBg3();

    cga1 = detmpbObj.GetCga1();
    cga2 = detmpbObj.GetCga2();
    cga3 = detmpbObj.GetCga3();

    cgb1 = detmpbObj.GetCgb1();
    cgb2 = detmpbObj.GetCgb2();
    cgb3 = detmpbObj.GetCgb3();

    tg1 = detmpbObj.GetTg1();
    tg2 = detmpbObj.GetTg2();
    tg3 = detmpbObj.GetTg3();

    textons = detmpbObj.GetTextons();

    int nOrient = tg1->size();

    // vector<cv::Mat> gPb(nOrient);
    gPb.resize(nOrient);

    for (int o = 0; o < nOrient; o++) {
        gPb.at(o) = weights.at(0) * *bg1->at(o) + weights.at(1) * *bg2->at(o) + weights.at(2) * *bg3->at(o) +
                    weights.at(3) * *cga1->at(o) + weights.at(4) * *cga2->at(o) + weights.at(5) * *cga3->at(o) +
                    weights.at(6) * *cgb1->at(o) + weights.at(7) * *cgb2->at(o) + weights.at(8) * *cgb3->at(o) +
                    weights.at(9) * *tg1->at(o) + weights.at(10) * *tg2->at(o) + weights.at(11) * *tg3->at(o) +
                    weights.at(12) * *sPb.at(o);
    }

    double minXi, maxXi;
    Point minLoc, maxLoc;
    string imageName;
    string filename;
    string gPbMatrixName;
    double alpha, beta;
    Mat destIm(origSize, CV_64FC1);

    filename = outFile + "_gPb.yml";
    cv::FileStorage file(filename, cv::FileStorage::WRITE);

    for (int i = 0; i < nOrient; i++) {
        imageName = outFile + "_gPb_i_" + to_string(i) + ".png";
        gPbMatrixName = "gPb_" + to_string(i);
        file << gPbMatrixName << gPb.at(i);
    }
    file.release();

    time(&nowd5);
    log_info("Running time: %ld (BmPb), %ld (mPb), %ld (sPb), %ld (AsPb)",
            nowd2 - nowd1, nowd3 - nowd2, nowd4 - nowd3, nowd5 - nowd4);
}

void GlobalPbWithNmax(String imgFile, String outFile, vector<cv::Mat>& gPb, double rsz) {
    time_t nowd1, nowd2, nowd3, nowd4, nowd5;
    time(&nowd1);

    Mat im = imread(imgFile);
    Mat imd;
    im.convertTo(imd, CV_64FC3, 1.0 / 255.0);   /// TODO: check
    int nChan = imd.channels();                 /// TODO: check
    Size origSize = imd.size();                 /// TODO: check
    detmpb detmpbObj;
    Mat* mPbRsz = NULL;

    vector<double> weights(13);
    if (nChan == 3) {
        weights = {0, 0, 0.0039, 0.0050, 0.0058, 0.0069, 0.0040, 0.0044, 0.0049, 0.0024, 0.0027, 0.0170, 0.0074};
    }
    else {
        weights = {0, 0, 0.0054, 0, 0, 0, 0, 0, 0, 0.0048, 0.0049, 0.0264, 0.0090};
    }

    string mPbFilename = outFile + "_mPb.yml";
    ifstream infile(mPbFilename);
    time(&nowd2);
    if (infile.good()) {
        cv::FileStorage fs(mPbFilename, cv::FileStorage::READ);
        cv::Mat mPbRszMat;
        fs["mPb_rsz_mat"] >> mPbRszMat;
        fs.release();
        mPbRsz = &(mPbRszMat);
    } else {
        Mat* mPbNmax = NULL;
        /* rsz not included, should take default 1.0 */
        multiscalePb(mPbNmax, mPbRsz, detmpbObj, imd, origSize, nChan, outFile);

        cv::FileStorage fs(mPbFilename, cv::FileStorage::WRITE);
        fs << "mPb_rsz_mat" << *mPbRsz;
        fs.release();
    }
    time(&nowd3);

    vector<cv::Mat*> sPb;
    int nvec = 17;
    spectralPb(sPb, mPbRsz, origSize, outFile, nvec);
    time(&nowd4);

    /// TODO: @Yupeng - repeating in multiscalePb.cpp
    vector<cv::Mat *> *bg1 = NULL, *bg2 = NULL, *bg3 = NULL, *cga1 = NULL, *cga2 = NULL, *cga3 = NULL,
                           *cgb1 = NULL, *cgb2 = NULL, *cgb3 = NULL, *tg1 = NULL, *tg2 = NULL, *tg3 = NULL;
    cv::Mat* textons = NULL;

    bg1 = detmpbObj.GetBg1();
    bg2 = detmpbObj.GetBg2();
    bg3 = detmpbObj.GetBg3();

    cga1 = detmpbObj.GetCga1();
    cga2 = detmpbObj.GetCga2();
    cga3 = detmpbObj.GetCga3();

    cgb1 = detmpbObj.GetCgb1();
    cgb2 = detmpbObj.GetCgb2();
    cgb3 = detmpbObj.GetCgb3();

    tg1 = detmpbObj.GetTg1();
    tg2 = detmpbObj.GetTg2();
    tg3 = detmpbObj.GetTg3();

    textons = detmpbObj.GetTextons();

    int nOrient = tg1->size();

    // vector<cv::Mat> gPb(nOrient);
    gPb.resize(nOrient);

    for (int o = 0; o < nOrient; o++) {
        gPb.at(o) = weights.at(0) * *bg1->at(o) + weights.at(1) * *bg2->at(o) + weights.at(2) * *bg3->at(o) +
                    weights.at(3) * *cga1->at(o) + weights.at(4) * *cga2->at(o) + weights.at(5) * *cga3->at(o) +
                    weights.at(6) * *cgb1->at(o) + weights.at(7) * *cgb2->at(o) + weights.at(8) * *cgb3->at(o) +
                    weights.at(9) * *tg1->at(o) + weights.at(10) * *tg2->at(o) + weights.at(11) * *tg3->at(o) +
                    weights.at(12) * *sPb.at(o);
    }

    double minXi, maxXi;
    Point minLoc, maxLoc;
    string imageName;
    string filename;
    string nmaxFileName;
    string gPbMatrixName;
    double alpha, beta;
    Mat destIm(origSize, CV_64FC1);
    Mat gPbNmax(gPb[0].rows, gPb[0].cols, CV_64FC1);

    filename = outFile + "_gPb.yml";
    nmaxFileName = outFile + "_gPb_nmax.yml";
    cv::FileStorage file(filename, cv::FileStorage::WRITE);
    cv::FileStorage nmaxFile(nmaxFileName, cv::FileStorage::WRITE);

    for (int i = 0; i < nOrient; i++) {
        imageName = outFile + "_gPb_i_" + to_string(i) + ".png";

        gPbMatrixName = "gPb_" + to_string(i);

        file << gPbMatrixName << gPb.at(i);

        for (int j = 0; j < gPb[0].rows; j++) {
            for (int k = 0; k < gPb[0].cols; k++) {
                gPbNmax.at<double>(j, k) = max(gPbNmax.at<double>(j, k), fabs(gPb[i].at<double>(j, k)));
            }
        }
    }
    file.release();

    nmaxFile << "gPb_nmax" << gPbNmax;
    nmaxFile.release();

    time(&nowd5);
    log_info("Running time: %ld (BmPb), %ld (mPb), %ld (sPb), %ld (AsPb)",
            nowd2 - nowd1, nowd3 - nowd2, nowd4 - nowd3, nowd5 - nowd4);
}

void GlobalPbCuda(cv::Mat& imagePatch, string outFile, vector<cv::Mat>& gPbOrient, cv::Mat& gPbNonmax,
                  unsigned int nOrient, unsigned int rank, double rsz) {
    // Mat imagePatch = imread(imgFile, CV_LOAD_IMAGE_COLOR);

    /* Damascene requires that the image be given an array of uint */
    /* in BGR order (B is Most significant byte and R is the Least Significant Byte) */
    const unsigned int nChannels = imagePatch.channels();
    const unsigned int rows = imagePatch.rows;
    const unsigned int cols = imagePatch.cols;

    /// FIXME: Possibly replace this with uint_least32_t to ensure minimum availablity
    ///        for this construct?
    unsigned int* patchData = new unsigned int[imagePatch.total()];

    float* hostGPb = NULL;
    float* hostGPbAllConcat = NULL;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            /* In this construct we have px = {uchar b, uchar g, uchar r} */
            Vec3b px = imagePatch.at<Vec3b>(i, j);

            /* B tp MSB */
            patchData[i * cols + j] = ((unsigned int)px[0]) << 16;
            /* G */
            patchData[i * cols + j] |= ((unsigned int)px[1]) << 8;
            /* R to LB */
            patchData[i * cols + j] |= ((unsigned int)px[2]);
        }
    }

    log_info("Processing image: (%d rows x %d cols)", rows, cols);

    /* calling computeGpb() from Damascene */
    try {
        computeGPb(rank, cols, rows, patchData, &hostGPb, &hostGPbAllConcat);
    } catch (int64_t error) {
        log_fatal("Damascene crashed, gPb computation failed");
        exit(2);
    }

    // gPbOrient.reserve(nOrient);

    Mat temp;
    double pb_min, pb_max, alpha, beta;
    /* Copy data back from hostGPbAllConcat to the gPb parameter */
    for (int orient = 0; orient < nOrient; ++orient) {
        cv::Mat gPbPiece = cv::Mat::zeros(rows, cols, CV_64FC1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                gPbPiece.at<double>(i, j) = hostGPbAllConcat[(orient * rows * cols) + (j * rows + i)];
            }
        }
        gPbOrient.push_back(gPbPiece);
    }

    /* Write host gpb to outfile */
    gPbNonmax = cv::Mat(rows, cols, CV_64FC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            gPbNonmax.at<double>(i, j) = hostGPb[i * cols + j];
        }
    }
    free(hostGPb);
    free(hostGPbAllConcat);
    delete[] patchData;
}
