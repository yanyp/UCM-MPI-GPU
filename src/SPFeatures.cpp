#include <ctime>
#include <sstream>
#include <limits>

#include "SPFeatures.h"
#include "log.h"

using namespace std;

/* constructor definition */
SPFeatures::SPFeatures(int binStep, int _nBaseLabels, int _nMidLabels, int _nCoarseLabels, int _nClasses,
                       bool _colorImageFlag, cv::Mat& _imageRef, cv::Mat& _labelMat, cv::Mat& _midLabelMat,
                       cv::Mat& _coarseLabelMat, string _pieceFilePath)
    : imageRef(_imageRef), labelMat(_labelMat), midLabelMat(_midLabelMat), coarseLabelMat(_coarseLabelMat) {
    /// TODO: Auto-generated constructor stub
    binStepSize = binStep;
    nBins = 255 / binStepSize + 1;

    nBaseLabels = _nBaseLabels;
    nMidLabels = _nMidLabels;
    nCoarseLabels = _nCoarseLabels;

    colorImageFlag = _colorImageFlag;
    nClasses = _nClasses;
    // nFeatures = nBins + 1 + nClasses;
    nFeatures = nBins + 1;

    if (colorImageFlag) {
        nFeatures += 3;
        cv::cvtColor(imageRef, grayScaleImage, CV_BGR2GRAY);
    }
    else {
        grayScaleImage = imageRef;
    }

    allDescrs = cv::Mat::zeros(nBaseLabels, nFeatures, CV_64F);

    baseLabelBuckets.resize(nBaseLabels);
    midLabelBuckets.resize(nMidLabels);
    coarseLabelBuckets.resize(nCoarseLabels);

    nAuxFeatures = 5;
    auxFeatures = cv::Mat::zeros(nBaseLabels, nAuxFeatures, CV_32F);

    DT = new CvDTree[nClasses];

    /// FIXME: added by @Yupeng
    pieceFilePath = _pieceFilePath;
    clusterCount = 32;
}

SPFeatures::~SPFeatures() {
    /// TODO: Auto-generated destructor stub
    delete[] DT;
}

const cv::Mat& SPFeatures::genAuxFeatures() {
    genLabelBucketsAtScaleK(midLabelBuckets, midLabelMat);
    genLabelBucketsAtScaleK(coarseLabelBuckets, coarseLabelMat);

    vector<int> nSmallBucketsInMidBucket(nMidLabels, 0);
    vector<int> nSmallBucketsInCoarseBucket(nCoarseLabels, 0);

    vector<int> midBucketEnclosingSmallBucket(nBaseLabels, -1);
    vector<int> coarseBucketEnclosingSmallBucket(nBaseLabels, -1);

    for (int i = 0; i < nBaseLabels; i++) {
        auxFeatures.at<float>(i, 0) = baseLabelBuckets.at(i).size();
        int any_element_I = baseLabelBuckets.at(i).at(0).first;
        int any_element_J = baseLabelBuckets.at(i).at(0).second;
        int midLabelBucket = midLabelMat.at<int>(any_element_I, any_element_J);
        int coarseLabelBucket = coarseLabelMat.at<int>(any_element_I, any_element_J);
        auxFeatures.at<float>(i, 1) = midLabelBuckets.at(midLabelBucket).size();
        auxFeatures.at<float>(i, 2) = coarseLabelBuckets.at(coarseLabelBucket).size();
        nSmallBucketsInMidBucket.at(midLabelBucket) += 1;
        nSmallBucketsInCoarseBucket.at(coarseLabelBucket) += 1;

        midBucketEnclosingSmallBucket.at(i) = midLabelBucket;
        coarseBucketEnclosingSmallBucket.at(i) = coarseLabelBucket;
    }

    for (int i = 0; i < nBaseLabels; i++) {
        assert((midBucketEnclosingSmallBucket.at(i) > -1) && (coarseBucketEnclosingSmallBucket.at(i) > -1));

        auxFeatures.at<float>(i, 3) = nSmallBucketsInMidBucket.at(midBucketEnclosingSmallBucket.at(i));
        auxFeatures.at<float>(i, 4) = nSmallBucketsInCoarseBucket.at(coarseBucketEnclosingSmallBucket.at(i));
    }

    return auxFeatures;
}

void SPFeatures::genLabelBucketsAtScaleK(vector<vector<pair<int, int>>>& scaleKBuckets,
                                         cv::Mat& labelMatAtScaleK) {
    int rows = labelMatAtScaleK.rows;
    int cols = labelMatAtScaleK.cols;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scaleKBuckets.at(labelMatAtScaleK.at<int>(i, j)).push_back(make_pair(i, j));
        }
    }
}

void SPFeatures::genIntensityHistograms() {
    genLabelBucketsAtScaleK(baseLabelBuckets, labelMat);

    for (int i = 0; i < nBaseLabels; i++) {
        for (auto it = baseLabelBuckets.at(i).begin(); it != baseLabelBuckets.at(i).end(); it++) {
            int bin = grayScaleImage.at<uchar>(it->first, it->second) / binStepSize;
            allDescrs.at<double>(i, bin)++;
        }
        for (int j = 0; j < nBins; j++) {
            allDescrs.at<double>(i, j) /= baseLabelBuckets.at(i).size();
            allDescrs.at<double>(i, j) *= 10;
        }
        /*
        if (i == nBaseLabels - 1)
       	    cout << baseLabelBuckets.at(i).size() << endl;
        }
        */
    }
}

/// TODO: @Yupeng Add texton features here
cv::Mat SPFeatures::makefilter(int sup, int sigma, int tau) {
    int hsup = (sup - 1) / 2;
    cv::Mat x, y, r;
    vector<int> t_x, t_y;
    for (int i = -hsup; i <= hsup; i++) {
        t_x.push_back(i);
        t_y.push_back(i);
    }

    /// TODO: @Yupeng Test this part
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, x);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), y);
    cv::Mat f(sup, sup, CV_64F);
    x.convertTo(x, CV_64F);
    y.convertTo(y, CV_64F);
    // OpenCVFileWriter(x, "test_x.yml", "x");
    // OpenCVFileWriter(y, "test_y.yml", "y");
    cv::pow(x.mul(x) + y.mul(y), 0.5, r);
    // OpenCVFileWriter(r, "test_r.yml", "r");
    // cout << "cos(PI) = " << cos(PI) << endl;
    double fsum = 0, fsum_abs = 0;
    stringstream sStream;
    for (int i = 0; i < sup; i++)
        for (int j = 0; j < sup; j++) {
            double temp = r.at<double>(i, j);
            f.at<double>(i, j) = cos(temp * (PI * tau / sigma));
            if (i == 0 && j == 0) {
                sStream << "First term: " << cos(temp * (PI * tau / sigma));
            }
            f.at<double>(i, j) *= exp(-temp * temp / (2 * sigma * sigma));
            if (i == 0 && j == 0) {
                sStream << ", Second term: " << exp(-temp * temp / (2 * sigma * sigma));
            }
            fsum += f.at<double>(i, j);
            fsum_abs += fabs(f.at<double>(i, j));
            if (i == 0 && j == 0) {
                sStream << ", f: " << f.at<double>(i, j);
                // log_debug("%s", sStream.str().c_str());
                sStream.str("");
            }
        }
    sStream << "f_1: " << f.at<double>(0, 0);
    /// FIXME: @Yupeng Strange element (very small) when save to file?
    // OpenCVFileWriter(f, "test_f1.yml", "f1");
    /* Pre-processing: zero mean and L1 normalization */
    f = f - fsum / (sup * sup);
    /* Pre-processing: L1 normalize */
    f = f / fsum_abs;
    sStream << ", f_2: " << f.at<double>(0, 0);
    // log_debug("%s", sStream.str().c_str());
    return f;
}

void SPFeatures::makeSfilters(vector<cv::Mat>* F, int NF, int SUP) {
    /* Returns the S filter bank of size 49*49*13 in F */
    F->push_back(makefilter(SUP, 2, 1));
    // OpenCVFileWriter((*F)[0], "test_f.yml", "f");
    F->push_back(makefilter(SUP, 4, 1));
    F->push_back(makefilter(SUP, 4, 2));
    F->push_back(makefilter(SUP, 6, 1));
    F->push_back(makefilter(SUP, 6, 2));
    F->push_back(makefilter(SUP, 6, 3));
    F->push_back(makefilter(SUP, 8, 1));
    F->push_back(makefilter(SUP, 8, 2));
    F->push_back(makefilter(SUP, 8, 3));
    F->push_back(makefilter(SUP, 10, 1));
    F->push_back(makefilter(SUP, 10, 2));
    F->push_back(makefilter(SUP, 10, 3));
    F->push_back(makefilter(SUP, 10, 4));
}

void SPFeatures::genTextonFeatures(int clusterCount) {
    vector<cv::Mat> F;
    int NF = 13, SUP = 49;
    makeSfilters(&F, NF, SUP);
    /* Expand the origin image for convolution */
    int margin = SUP / 2;
    cv::Mat imagePadded;
    /// TODO: @Yupeng Test below line -- BORDER_REPLICATE
    cv::copyMakeBorder(grayScaleImage, imagePadded, margin, margin, margin, margin, cv::BORDER_REFLECT);
    /*
    OpenCVFileWriter(grayScaleImage, "test_paddedbefore.yml", "grayScaleImage");
    OpenCVFileWriter(imagePadded, "test_padded.yml", "imagePadded");
    */

    /// FIXME: @Yupeng Compute gPb only if the file does not exist
    /* load the existing codebook from file */
    /* modified by Yupeng - next two lines */
    string textonMap_filename = pieceFilePath + "_textonMap.yml";
    ifstream ifile(textonMap_filename);
    cv::Mat textonMap(grayScaleImage.rows, grayScaleImage.cols, CV_32S);
    if (!ifile) {
        // string texton_filename = pieceFilePath + "_texton.yml";
        // ifstream ifile2(texton_filename);
    vector<cv::Mat> texton_responses;
        // if (!ifile2) {
            // log_info("Writing texton responses to file: %s", texton_filename.c_str());
            // cv::FileStorage fs(texton_filename, cv::FileStorage::WRITE);
    cv::Mat response, kernel;
    cv::Rect r(margin, margin, grayScaleImage.cols, grayScaleImage.rows);
    imagePadded.convertTo(imagePadded, CV_32F);     /* save the space */
    for (int i = 0; i < NF; i++) {
        F[i].convertTo(kernel, CV_32F);
        cv::filter2D(imagePadded, response, imagePadded.depth(), kernel);
        texton_responses.push_back(response(r).clone());
                // fs << "response_" + to_string(i) << texton_responses.back();
    }
            // fs.release();
        // }
        /*
        else {
            log_info("Reading texton responses from file: %s", texton_filename.c_str());
            cv::FileStorage fs(texton_filename, cv::FileStorage::READ);
            for (int i = 0; i < NF; i++) {
                cv::Mat response;       /// FIXME: @Yupeng noted - must be inside!!!!!
                fs["response_" + to_string(i)] >> response;
                texton_responses.push_back(response);
                // OpenCVFileWriter(response, string("test_texton_" + to_string(i) + ".yml"), "response");
            }
            fs.release();
        }
        */

        string bigPieceFilePath =
            pieceFilePath.substr(0, pieceFilePath.substr(0, pieceFilePath.find_last_of("_")).find_last_of("_"));
        string textonCodebookFilename = bigPieceFilePath + "_texton_codebook.yml";
        log_info("Reading the texton codebook from file: %s", textonCodebookFilename.c_str());
        cv::FileStorage fs(textonCodebookFilename, cv::FileStorage::READ);
        cv::Mat texton_codebook;
        fs["texton_codebook"] >> texton_codebook;       /* NF * clusterCount */
        fs.release();
        texton_codebook.convertTo(texton_codebook, CV_32F);
        log_info("Assigning cluster centers to pixels...");
        /* cluster center assignment */
        cv::Mat pixel_texton(NF, 1, CV_32F);
        // cout << "grayScaleImageSize = " << grayScaleImage.rows << ", " << grayScaleImage.cols << endl;
        clock_t mapTime = clock();
        float minDistance;
        int clusterNumber = -1;
        for (int i = 0; i < grayScaleImage.rows; i++) {
            for (int j = 0; j < grayScaleImage.cols; j++) {
                minDistance = numeric_limits<float>::infinity();
                for (int k = 0; k < NF; k++) {
                    pixel_texton.at<float>(k, 0) = texton_responses[k].at<float>(i, j);
                    if (i == 0 && j == 0) {
                        // log_debug("pixel_texton %d: %f", k, pixel_texton.at<float>(k, 0));
                    }
                }
                for (int k = 0; k < clusterCount; k++) {
                    /*
                    if(i == 0 && j < 5) {
                        cout << cv::norm(pixel_texton, texton_codebook.col(k), cv::NORM_L2) << "\t" << k << endl;
                    }
                    if(i == 0 && j == 0) {
                        cout << "texton_codebook 1st = " << texton_codebook.at<float>(0, k) << endl;
                    }
                    */
                    if (cv::norm(pixel_texton, texton_codebook.col(k), cv::NORM_L2) < minDistance) {
                        clusterNumber = k;
                    }
                }
                if (clusterNumber == -1) {
                    log_error("Unable to assign cluster to pixel (%d, %d)", i, j);
                }
                textonMap.at<int>(i, j) = clusterNumber;
            }
        }
        mapTime = clock() - mapTime;
        log_info("Time taken for cluster assignment: %f sec", (float) mapTime / CLOCKS_PER_SEC);
        /* modified by Yupeng - untio textonMap.release */
        OpenCVFileWriter(textonMap, pieceFilePath + "_textonMap.yml", "textonMap");
    }
    else {
        log_info("Reading the texton map from file: %s", textonMap_filename.c_str());
        cv::FileStorage fs(textonMap_filename, cv::FileStorage::READ);
        fs["textonMap"] >> textonMap;
        fs.release();
    }

    /* create histogram by average pooling */
    /// FIXME: @Yupeng - must be placed after genIntensityHistograms
    cv::Mat textonDescrs = cv::Mat::zeros(nBaseLabels, clusterCount, CV_64F);
    for (int i = 0; i < nBaseLabels; i++) {
        for (auto it = baseLabelBuckets.at(i).begin(); it != baseLabelBuckets.at(i).end(); it++) {
            textonDescrs.at<double>(i, textonMap.at<int>(it->first, it->second))++;
        }
        for (int j = 0; j < clusterCount; j++) {
            textonDescrs.at<double>(i, j) /= baseLabelBuckets.at(i).size();
            textonDescrs.at<double>(i, j) *= 10;    /* texton feature weight */
        }
    }
    // OpenCVFileWriter(textonDescrs, pieceFilePath + "_textonDescrs.yml", "textonDescrs");
    cv::hconcat(textonDescrs, allDescrs, allDescrs);
    nFeatures += clusterCount;
}

cv::Mat SPFeatures::genTextonMap(cv::Mat& textonImage, int clusterCount, bool isBroadcasted = false) {
    vector<cv::Mat> F;
    int NF = 13, SUP = 49;
    makeSfilters(&F, NF, SUP);
    /* Expand the origin image for convolution */
    int margin = SUP / 2;
    cv::Mat imagePadded;
    /// TODO: @Yupeng Test below line -- BORDER_REPLICATE
    cv::copyMakeBorder(textonImage, imagePadded, margin, margin, margin, margin, cv::BORDER_REFLECT);
    /*
    OpenCVFileWriter(textonImage, "test_paddedbefore.yml", "textonImage");
    OpenCVFileWriter(imagePadded, "test_padded.yml", "imagePadded");
    */

    /// FIXME: @Yupeng Compute gPb only if the file does not exist
    /* load the existing codebook from file */
    /* modified by Yupeng - next two lines*/

    string textonMap_filename = pieceFilePath + "_textonMap.yml";
    ifstream ifile(textonMap_filename);
    cv::Mat textonMap(textonImage.rows, textonImage.cols, CV_32S);
    if (isBroadcasted || !ifile) {
        // string texton_filename = pieceFilePath + "_texton.yml";
        // ifstream ifile2(texton_filename);
        vector<cv::Mat> texton_responses;
        // if (!ifile2) {
            // log_info("Writing texton responses to file: %s", texton_filename.c_str());
            // cv::FileStorage fs(texton_filename, cv::FileStorage::WRITE);
        cv::Mat response, kernel;
        cv::Rect r(margin, margin, textonImage.cols, textonImage.rows);
        imagePadded.convertTo(imagePadded, CV_32F);     /* save the space */
        for (int i = 0; i < NF; i++) {
            F[i].convertTo(kernel, CV_32F);
            cv::filter2D(imagePadded, response, imagePadded.depth(), kernel);
            texton_responses.push_back(response(r).clone());
                // fs << "response_" + to_string(i) << texton_responses.back();
        }
            // fs.release();
        // }
        /*
        else {
            log_info("Reading texton responses from file: %s", texton_filename.c_str());
            cv::FileStorage fs(texton_filename, cv::FileStorage::READ);
            for (int i = 0; i < NF; i++) {
                cv::Mat response;       /// FIXME: @Yupeng noted - must be inside!!!!!
                fs["response_" + to_string(i)] >> response;
                texton_responses.push_back(response);
                // OpenCVFileWriter(response, string("test_texton_" + to_string(i) + ".yml"), "response");
            }
            fs.release();
        }
        */

        string bigPieceFilePath =
            pieceFilePath.substr(0, pieceFilePath.substr(0, pieceFilePath.find_last_of("_")).find_last_of("_"));
        string textonCodebookFilename = bigPieceFilePath + "_texton_codebook.yml";
        if (!isBroadcasted) {
            log_info("Reading the texton codebook from file: %s", textonCodebookFilename.c_str());
        }
        cv::FileStorage fs(textonCodebookFilename, cv::FileStorage::READ);
        cv::Mat texton_codebook;
        fs["texton_codebook"] >> texton_codebook;       /* NF * clusterCount */
        fs.release();
        texton_codebook.convertTo(texton_codebook, CV_32F);
        if (!isBroadcasted) {
            log_info("Assigning cluster centers to pixels...");
        }
        /* cluster center assignment */
        cv::Mat pixel_texton(NF, 1, CV_32F);
        // cout << "textonImageSize = " << textonImage.rows << ", " << textonImage.cols << endl;
        clock_t mapTime = clock();
        float minDistance;
        int clusterNumber = -1;
        for (int i = 0; i < textonImage.rows; i++) {
            for (int j = 0; j < textonImage.cols; j++) {
                minDistance = numeric_limits<float>::infinity();
                for (int k = 0; k < NF; k++) {
                    pixel_texton.at<float>(k, 0) = texton_responses[k].at<float>(i, j);
                    if (i == 0 && j == 0) {
                        // log_debug("pixel_texton %d: %f", k, pixel_texton.at<float>(k, 0));
                    }
                }
                for (int k = 0; k < clusterCount; k++) {
                    /*
                    if(i == 0 && j < 5) {
                        cout << cv::norm(pixel_texton, texton_codebook.col(k), cv::NORM_L2) << "\t" << k << endl;
                    }
                    if(i == 0 && j == 0) {
                        cout << "texton_codebook 1st = " << texton_codebook.at<float>(0, k) << endl;
                    }
                    */
                    if (cv::norm(pixel_texton, texton_codebook.col(k), cv::NORM_L2) < minDistance) {
                        clusterNumber = k;
                    }
                }
                if (clusterNumber == -1) {
                    log_error("Unable to assign cluster to pixel (%d, %d)", i, j);
                }
                textonMap.at<int>(i, j) = clusterNumber;
            }
        }
        mapTime = clock() - mapTime;
        if (!isBroadcasted) {
            log_info("Time taken for cluster assignment: %f sec", (float) mapTime / CLOCKS_PER_SEC);
            OpenCVFileWriter(textonMap, pieceFilePath + "_textonMap.yml", "textonMap");
        }
    }
    
    else {
        log_info("Reading the texton map from file: %s", textonMap_filename.c_str());
        cv::FileStorage fs(textonMap_filename, cv::FileStorage::READ);
        fs["textonMap"] >> textonMap;
        fs.release();
    }
    return textonMap;
}


void SPFeatures::genCornerDescrs(int blockSize, int apertureSize, double k, int threshold) {
    cv::Mat dst, dst_norm;
    cv::cornerHarris(grayScaleImage, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int)dst_norm.at<float>(i, j) > threshold) {
                allDescrs.at<double>(labelMat.at<int>(i, j), nBins) += 1;
            }
        }
    }

    for (int i = 0; i < nBaseLabels; i++) {
        allDescrs.at<double>(i, nBins) /= baseLabelBuckets.at(i).size();
        allDescrs.at<double>(i, nBins) *= 1;    /// FIXME: @Yupeng - originally to be 10
    }
}

void SPFeatures::genAverageFeatures() {
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < nBaseLabels; i++) {
            for (auto it = baseLabelBuckets.at(i).begin(); it != baseLabelBuckets.at(i).end(); it++) {
                allDescrs.at<double>(i, nBins + j + 1) += imageRef.at<cv::Vec3b>(it->first, it->second)[j];
            }
            allDescrs.at<double>(i, nBins + j + 1) /= baseLabelBuckets.at(i).size();
            allDescrs.at<double>(i, nBins + j + 1) *= 10.0 / 255.0;
        }
    }
}

void SPFeatures::DTFeatures(const vector<int>& gtSuperpixels, const vector<int>& gtClassLabels, int labelOffset) {
    /* training the DT */
    int nGTPixels = gtClassLabels.size();
    log_debug("labelOffset: %d", labelOffset);

    cv::Mat auxFeatures_train(0, nAuxFeatures, CV_32F);
    for (auto it = gtSuperpixels.begin(); it != gtSuperpixels.end(); it++) {
        int p = *it - labelOffset;
        log_debug("p: %d", p);
        assert((p < nBaseLabels) && (p >= 0));
        auxFeatures_train.push_back(auxFeatures.row(*it - labelOffset));
    }

    assert(auxFeatures_train.rows == nGTPixels);

    CvDTreeParams params;
    params.max_depth = 8;
    params.min_sample_count = 2;
    params.regression_accuracy = 0;
    params.use_surrogates = 0;
    params.max_categories = nClasses;
    params.cv_folds = 2;
    params.use_1se_rule = false;
    params.truncate_pruned_tree = false;
    params.priors = NULL;

    cv::Mat var_type = cv::Mat(auxFeatures_train.cols + 1, 1, CV_8U);
    var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL));
    var_type.at<uchar>(auxFeatures_train.cols, 0) = CV_VAR_CATEGORICAL;

    for (int i = 0; i < nClasses; i++) {
        cv::Mat trainingLabels_DT = cv::Mat::ones(nGTPixels, 1, CV_32F);
        for (int j = 0; j < nGTPixels; j++) {
            if (gtClassLabels.at(j) != i + 1) {
                trainingLabels_DT.at<float>(j, 0) = -1;
            }
        }
        DT[i].train(auxFeatures_train, CV_ROW_SAMPLE, trainingLabels_DT, cv::Mat(), cv::Mat(), var_type, cv::Mat(),
                    params);
    }

    /* testing the DT */
    cv::Mat binFeatures = cv::Mat::zeros(nBaseLabels, nClasses, CV_64F);
    CvDTreeNode* resultNode = NULL;
    for (int j = 0; j < nBaseLabels; j++) {
        cv::Mat testSample = auxFeatures.row(j);
        for (int i = 0; i < nClasses; i++) {
            resultNode = DT[i].predict(testSample, cv::Mat(), false);
            /// TODO: check this line because of double or float and so comparison is invalid
            if (static_cast<int>(resultNode->value) == 1) {
                binFeatures.at<double>(j, i) = 1.0;
            }
        }
    }

    /// FIXME: @Yupeng comment
    // OpenCVFileWriter(binFeatures, "./labeledData/binFeatures.yml", "binF");

    cv::hconcat(binFeatures, allDescrs, allDescrs);
    nFeatures += nClasses;
}

void SPFeatures::addBiasOne() {
    cv::Mat verticalOnes = cv::Mat::ones(nBaseLabels, 1, CV_64F);
    cv::hconcat(verticalOnes, allDescrs, allDescrs);
    nFeatures += 1;
}

/// FIXME: overload by @Yupeng
void SPFeatures::addBiasOne(cv::Mat& anyDescrs) {
    if (anyDescrs.rows > 0) {
        cv::Mat verticalOnes = cv::Mat::ones(anyDescrs.rows, 1, CV_64F);
        cv::hconcat(verticalOnes, anyDescrs, anyDescrs);
    }
    else {
        anyDescrs = cv::Mat(anyDescrs.rows, 1 + anyDescrs.cols, CV_64F);
    }
}

/* design for patch-based method */
void SPFeatures::genSuperpixelFeatures() {
    genIntensityHistograms();
    genCornerDescrs();
    genAverageFeatures();
    // genTextonFeatures(32);
}

void SPFeatures::genRecPatchFeatures(vector<pair<int, int>>* centerCoords, cv::Mat& recDescrs) {
    if (!centerCoords->empty()) {
        /* create patch and extract features */
        cv::Mat dst, dst_norm;
        int blockSize = 2, apertureSize = 3, threshold = 60;    /// FIXME: @Yupeng - replace this and below lines
        double k = 0.04;
        cv::cornerHarris(grayScaleImage, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

        /// TODO @Yupeng - assume textonMap already generated, read directly
        cv::Mat textonMap, textonDescrs = cv::Mat::zeros(centerCoords->size(), clusterCount, CV_64F);
        textonMap = genTextonMap(grayScaleImage, clusterCount);
        ////cout << "clusterCount = " << clusterCount << endl;
        //// OpenCVFileWriter(textonMap, pieceFilePath + "_textonMapTest.yml", "textonMapTest");
        // string textonMap_filename = pieceFilePath + "_textonMap.yml";
        // log_info("Reading the texton map from file: %s", textonMap_filename.c_str());
        // cout << "Coord size = " << centerCoords->size() << "\trecDescrs size = " << recDescrs.rows <<", " <<
        // recDescrs.cols << endl;
        // cv::FileStorage fs(textonMap_filename, cv::FileStorage::READ);
        // fs["textonMap"] >> textonMap;
        // fs.release();
        int patchSize = 64;
        for (int i = 0; i < centerCoords->size(); i++) {
            /* create patch */
            cv::Mat rgbPatch, grayPatch;
            int north = min(patchSize / 2, (*centerCoords)[i].first);
            int south = min(patchSize / 2, grayScaleImage.rows - 1 - (*centerCoords)[i].first);
            int west = min(patchSize / 2, (*centerCoords)[i].second);
            int east = min(patchSize / 2, grayScaleImage.cols - 1 - (*centerCoords)[i].second);
            /*
            if (pieceFilePath.find("_0_0") != string::npos) {
                prototype rect(x, y, width, height)
                if (i == 539) {
                    cout << "i = " << i << "\tN = " << north << "\tS = " << south << "\tW" << west << "\tE" << east;
                }
                cout << pieceFilePath << "\t";
                cout << "i = " << i << "/" << centerCoords->size() << "\t";
                cout << "start.x = " << (*centerCoords)[i].second - west << "\t";
                cout << "start.y = " << (*centerCoords)[i].first - north << "\t";
                cout << "width = " << west + east + 1 << "\t";
                cout << "height = " << north + south + 1 << "\t";
                cout << "center.x = " << (*centerCoords)[i].second << "\t";
                cout << "center.y = " << (*centerCoords)[i].first << "\t";
                cout << "north = " << north << "\t";
                cout << "south = " << south << "\t";
                cout << "west = " << west << "\t";
                cout << "east = " << east << "\t";
                cout << "gray.cols = " << grayScaleImage.cols << "\t";
                cout << "gray.rows = " << grayScaleImage.rows << "\t";
                cout << endl;
            }
            */

            // cout << "Before Rect r??" << "\t";
            cv::Rect r((*centerCoords)[i].second - west, (*centerCoords)[i].first - north, west + east + 1,
                       north + south + 1);
            imageRef(r).copyTo(rgbPatch);
            // cout << "After imageRef copyTo??" << "\t";
            grayScaleImage(r).copyTo(grayPatch);
            /*
            cout << "After gray copyTo??" << "\t";
            if(i == 539) {
                cout << "\tExtracting" << endl;
            }
            */

            /* (1) Intensity Histogram + (2) Corner Density + (3) Average RGB + (4) Texton Codes */
            for (int j = 0; j < grayPatch.rows; j++) {
                for (int k = 0; k < grayPatch.cols; k++) {
                    /* row and col in original image space */
                    int row = (*centerCoords)[i].first - north + j;
                    int col = (*centerCoords)[i].second - west + k;
                    /*
                    if(i == 539) {
                        cout << "j = " << j << "\tk = " << k << "\trow = " << row << "\tcol = " << col << endl;
                    }
                    */
                    /* 1st Feature */
                    int bin = grayPatch.at<uchar>(j, k) / binStepSize;
                    recDescrs.at<double>(i, bin) += 1;
                    /* 2nd Feature */
                    if ((int)dst_norm.at<float>(row, col) > threshold) recDescrs.at<double>(i, nBins) += 1;
                    /* 3rd Feature */
                    for (int m = 1; m <= 3; m++)
                        recDescrs.at<double>(i, nBins + m) += rgbPatch.at<cv::Vec3b>(j, k)[m - 1];
                    /* 4th Feature */
                    textonDescrs.at<double>(i, textonMap.at<int>(row, col)) += 1;
                }
            }
            int area = grayPatch.rows * grayPatch.cols;
            ///cout << "i = " << i << ", area = " << area << endl;
            /*
            if(i == 538) {
                cout << "Stop at area??\tarea = " << area << "\tallDescrs Size = " << allDescrs.rows << "\t" <<
                    allDescrs.cols << endl;
            }
            */
            for (int j = 0; j < nBins; j++) {
                recDescrs.at<double>(i, j) /= area;
                recDescrs.at<double>(i, j) *= 10;
            }
            recDescrs.at<double>(i, nBins) /= area;
            recDescrs.at<double>(i, nBins) /= 1;    /// FIXME: @Yupeng - weight should be constant at one place
            for (int j = 1; j <= 3; j++) {
                recDescrs.at<double>(i, nBins + j) /= area;
                recDescrs.at<double>(i, nBins + j) *= 10.0 / 255.0;
            }
            for (int j = 0; j < clusterCount; j++) {
                textonDescrs.at<double>(i, j) /= area;
                textonDescrs.at<double>(i, j) *= 10;    /* texton feature weight */
            }
            // DT Feature is added after concatenation in main.cpp
            /*
            if(i == 538) {
                cout << "Pass the end?" << endl;
            }
            */
        }
        OpenCVFileWriter(recDescrs, pieceFilePath + "_recDescrs.yml", "recDescrs");
        OpenCVFileWriter(textonDescrs, pieceFilePath + "_textonDescrs.yml", "textonDescrs");

        log_info("Before hconcat");
        cv::hconcat(textonDescrs, recDescrs, recDescrs);
    }
    else {
        log_warn("No points are selected");
        recDescrs = cv::Mat(0, recDescrs.cols + clusterCount, CV_64F);
    }
}

void SPFeatures::genSpCenterFeatures() {
    log_debug("labelMat | rows = %d, cols = %d", labelMat.rows, labelMat.cols);
    genLabelBucketsAtScaleK(baseLabelBuckets, labelMat);
    /* find the mean coordinates for each superpixel */
    vector<pair<int, int>> baseMeanCoords;
    log_debug("nBaseLabels = %d, bucketSize = %lu", nBaseLabels, baseLabelBuckets.size());
    for (int i = 0; i < nBaseLabels; i++) {
        pair<float, float> meanCoord(0, 0);     /// FIXME @Yupeng - check vector push_back
        for (int j = 0; j < baseLabelBuckets[i].size(); j++) {
            /* avoid potential overflow */
            meanCoord.first += baseLabelBuckets[i][j].first * 1.0 / baseLabelBuckets[i].size();
            meanCoord.second += baseLabelBuckets[i][j].second * 1.0 / baseLabelBuckets[i].size();
        }
        meanCoord.first = floor(meanCoord.first);
        meanCoord.second = floor(meanCoord.second);
        baseMeanCoords.push_back(meanCoord);
        /*
        if(i == 538 || i == 539) {
            cout << "i = " << i << "\tfirst = " << baseMeanCoords.back().first << "\tsecond = " <<
        }
        baseMeanCoords.back().second << endl;
        */
    }

    genRecPatchFeatures(&baseMeanCoords, allDescrs);
    nFeatures += clusterCount;
}

void SPFeatures::genSeedCenterFeatures(vector<pair<int, int>>* gtCoords, cv::Mat& trainDescrs) {
    /* get the coordinates of seed points */
    genRecPatchFeatures(gtCoords, trainDescrs);
    /// FIXME @Yupeng - we don't need below line any more
    // nFeatures += clusterCount;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Globally get the training features
void SPFeatures::genRecPatchFeatures(cv::Mat& rgbPatch, cv::Mat& recDescrs, cv::Mat& textonDescrs, int n) {
    cv::Mat grayScalePatch;
    cv::cvtColor(rgbPatch, grayScalePatch, CV_BGR2GRAY);
        
    cv::Mat dst, dst_norm;
    int blockSize = 2, apertureSize = 3, threshold = 60;    /// FIXME: @Yupeng - replace this and below lines
    double k_para = 0.04;
    cv::cornerHarris(grayScalePatch, dst, blockSize, apertureSize, k_para, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    cv::Mat textonMap = genTextonMap(grayScalePatch, 32, true);

    /* (1) Intensity Histogram + (2) Corner Density + (3) Average RGB + (4) Texton Codes */
    for (int j = 0; j < grayScalePatch.rows; j++) {
        for (int k = 0; k < grayScalePatch.cols; k++) {
            /* 1st Feature */
            int bin = grayScalePatch.at<uchar>(j, k) / binStepSize;
            recDescrs.at<double>(n, bin) += 1;
            /* 2nd Feature */
            if ((int)dst_norm.at<float>(j, k) > threshold) {
                recDescrs.at<double>(n, nBins) += 1;
            }
            /* 3rd Feature */
            for (int m = 1; m <= 3; m++) {
                recDescrs.at<double>(n, nBins + m) += rgbPatch.at<cv::Vec3b>(j, k)[m - 1];
            }
                
            /* 4th Feature */
            textonDescrs.at<double>(n, textonMap.at<int>(j, k)) += 1;
        }
    }

    int area = grayScalePatch.rows * grayScalePatch.cols;
    assert(area == 64 * 64);

    for (int j = 0; j < nBins; j++) {
        recDescrs.at<double>(n, j) /= area;
        recDescrs.at<double>(n, j) *= 10;
    }
    recDescrs.at<double>(n, nBins) /= area;
    recDescrs.at<double>(n, nBins) /= 1;    /// FIXME: @Yupeng - weight should be constant at one place
    for (int j = 1; j <= 3; j++) {
        recDescrs.at<double>(n, nBins + j) /= area;
        recDescrs.at<double>(n, nBins + j) *= 10.0 / 255.0;
    }
    for (int j = 0; j < clusterCount; j++) {
        textonDescrs.at<double>(n, j) /= area;
        textonDescrs.at<double>(n, j) *= 10;    /* texton feature weight */
    }
    return;
}

void SPFeatures::genSeedCenterFeatures(std::vector<cv::String>* filenames, vector<int>& gtClassLabels, cv::Mat& trainDescrs) {
    int cnt = static_cast<int>(filenames->size());
    gtClassLabels.clear();  // MUST DO
    cv::Mat textonDescrs = cv::Mat::zeros(cnt, clusterCount, CV_64F);

    for (int i = 0; i < cnt; i++) {
        std::string filename = filenames->at(i);
        int start = filename.find("Class_") + (int)std::strlen("Class_");
        std::string c = filename.substr(start, filename.find("_Seed") - start);
        /*
        cout << "GT Image Filename = " << filename << endl;
        cout << "A = " << filename.find("Class_") << ", B = " << (int)std::strlen("Class_") << ", C = " << filename.find("_Seed") << endl;
        cout << "c = " << c << ", " << std::stoi(c) << endl;
        */
        gtClassLabels.push_back(std::stoi(c));
        cv::Mat gtImage = cv::imread(filename);
        genRecPatchFeatures(gtImage, trainDescrs, textonDescrs, i);
    }
    // OpenCVFileWriter(trainDescrs, pieceFilePath + "_broad_recDescrs.yml", "recDescrs");
    // OpenCVFileWriter(textonDescrs, pieceFilePath + "_broad_textonDescrs.yml", "textonDescrs");

    cv::hconcat(textonDescrs, trainDescrs, trainDescrs);
    OpenCVFileWriter(trainDescrs, pieceFilePath + "_broad_trainDescrs.yml", "trainDescrs");
}
