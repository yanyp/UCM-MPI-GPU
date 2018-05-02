#include <cv.h>
#include <highgui.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <limits>
#include <utility>
#include "otherheaders.h"

/// TODO: @Yupeng added
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <MatOp/SparseCholesky.h>
#include <MatOp/SparseGenMatProd.h>
#include <SymGEigsSolver.h>

// using namespace Eigen;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double, double> T;

using namespace cv;
using namespace std;

/* sort pair<int,int> based on first value */
typedef std::pair<int, int> mypair;
bool comparatorPair(const mypair& l, const mypair& r) {
    return l.first < r.first;
}

void spectralPb(std::vector<Mat*>& sPb, Mat* mPb, Size orig_size, std::string outFile, int nvec) {
    int ty = mPb->rows;
    int tx = mPb->cols;
    /*
    Mat A1  = mPb->t();
    int ty = A1.rows;
    int tx = A1.cols;
    */

    /*
    cout << "========Printing mPb ============" <<endl;
    double* ptrData;
    for (int i = 0; i < mPb->rows; i++) {
        ptrData = (double*) mPb->ptr(i);
        for (int j = 0; j < mPb->cols; j++) {
            cout << ptrData[j] << ",";
        }
        cout << endl;
    }
    cout << "========Ended Printing mPb ============" <<endl;
    */

    Mat l1 = Mat::zeros(ty + 1, tx, CV_64F);
    Mat l2 = Mat::zeros(ty, tx + 1, CV_64F);

    mPb->copyTo(l1(Rect(0, 1, tx, ty)));
    mPb->copyTo(l2(Rect(1, 0, tx, ty)));

    log_info("Setting the sparse symmetric affinity matrix");
    vector<double> val;
    vector<int> I;      /// FIXME: long int or long long int??
    vector<int> J;      /// FIXME: long int or long long int??
    vector<double> valD;

    time_t now1, now2, now3, now4, now5, now5a, now5b, now6, now7, now8, now9, now10, now11;
    time(&now1);
    buildW(valD, val, I, J, l1, l2);    /// FIXME: check the output order of val I and J
    time(&now2);

    log_info("At the end of buildW: %ld", now2 - now1);

    /// TODO: @Yupeng uncomment @Manu's display codes
    /*
    cout << "=====Display matrix =======" << endl;
    for (int i = 0 ;i < val.size(); i++) {
        cout << "("<<I.at(i)+1<<"," << J.at(i)+1<<") =  " << val.at(i) << endl;
    }
    cout << "=====Display ended =======" << endl;
    */

    int nnz = val.size();
    int nnzD1 = valD.size();

    /// FIXME: @Yupeng added
    log_debug("nnz = %d, nnzD1 = %d", nnz, nnzD1);
    log_debug("ty = %d, tx = %d", ty, tx);

    assert(val.size() == I.size() && I.size() == J.size());

    int wy = (int) *max_element(I.begin(), I.end());
    time(&now3);
    log_info("Time for one max: %ld", now3 - now2);
    int wx = (int) *max_element(J.begin(), J.end());

    // cout << "wy = " << wy << "wx = " << wx << endl;
    int wymin = (int)*min_element(I.begin(), I.end());
    int wxmin = (int)*min_element(J.begin(), J.end());
    // cout << "wymin = " << wymin << "wxmin = " << wxmin << endl;
    int n = (wy == wx) ? wy + 1 : max(wy, wx) + 1;      /*  added one because indices start from zero */

    double valmin = (double)*min_element(val.begin(), val.end());
    double valmax = (double)*max_element(val.begin(), val.end());

    time(&now4);
    log_info("Time for all comparisons: %ld", now4 - now2);
    // cout << "valmax = " << valmax << " valmin = " << valmin << endl;

    vector<int> pcol(n + 1, -1);
    int colInd = 0;
    pcol.at(colInd) = 0;
    for (int i = 1; i < nnz; i++) {
        if (I.at(i) != I.at(i - 1)) pcol.at(++colInd) = i;
    }
    pcol.at(++colInd) = nnz;

    time(&now5a);

    log_debug("Time pcol: %ld", now5a - now4);

    /// FIXME: part below commented because of more efficient version above
    /*
    vector<int> pcol_old(n + 1, -1);
    std::vector<int>::iterator it;
    for (int i = 0; i < n; i++) {
        fs << "orig_size" << orig_size;
        it = find(I.begin(), I.end(), i);
        if (it != I.end()) {
            pcol_old.at(i) = it - I.begin();
        }
    }
    pcol_old.at(n) = nnz;
    */

    time(&now5);
    log_debug("Time pcol_old: %ld", now5 - now5a);

    time(&now5b);
    log_debug("Comparison time: %ld", now5b - now5);

    int nnzD = n;
    std::vector<double> valDmW(nnz);
    for (int i = 0; i < nnz; i++) {
        if (I.at(i) != J.at(i)) {
            valDmW.at(i) = -val.at(i);
        }
        else {
            /// FIXME: if any element changes to zero
            valDmW.at(i) = valD.at(I.at(i)) - val.at(i);
        }
    }

    time(&now6);
    log_info("Time pcol: %ld", now6 - now5);
    std::vector<int> IB;
    IB.reserve(n);
    int cnt = 0;
    std::generate_n(std::back_inserter(IB), n, [cnt]() mutable { return cnt++; });

    time(&now7);
    log_info("Time generate_n: %ld", now7 - now6);
    // cout << IB.at(0) << "," << IB.at(n - 1) << "," << nnzD << endl;

    std::vector<int> JB(IB);
    JB.push_back(nnzD);

    std::vector<double> valB(nnzD);
    for (int i = 0; i < nnzD; i++) {
        valB.at(i) = 0;
        if (pcol.at(i) != -1)
            for (int j = pcol[i]; j < pcol[i + 1]; j++) {
                valB.at(i) += val.at(j);
            }
    }

    /*
    cout << "=====Display matrix =======" << endl;
    for (int i = 0; i < valD.size(); i++) {
        cout << "("<<IB.at(i)+1<<"," << JB.at(i)+1<<") =  " << valD.at(i) << endl;
    }
    cout << "=====Display ended =======" << endl;
    */

    time(&now8);

    /// FIXME: @Yupeng fixed
    log_info("Before Spectra: %ld", now8 - now7);
    log_info("Overall orig reported: %ld", now8 - now2);
    log_info("Setting up matrices D and D-W for Spectra");

    time(&now9);

    /// FIXME: @Yupeng output for debugging
    log_info("OutFile: %s | ty (rows) = %d, tx (cols) = %d", outFile.c_str(), ty, tx);

    /* eigenvector computation using Spectra library begins */
    double* p_eigVals;
    double* p_eigVec;

    int nD = ty * tx;       /* nrows * ncols */
    Eigen::SparseMatrix<double> D2mW2(nD, nD);
    std::vector<T> D2mW2_tripletList;
    D2mW2_tripletList.reserve(nnz);
    for (int i = 0; i < nnz; i++) {
        /* I is rowIdx, J is colIdx, condition "tempCol >= row" for push_back of I, J, val */
        D2mW2_tripletList.push_back(T(J.at(i), I.at(i), valDmW.at(i)));
    }
    D2mW2.setFromTriplets(D2mW2_tripletList.begin(), D2mW2_tripletList.end());
    D2mW2.makeCompressed();

    Eigen::SparseMatrix<double> D2(nD, nD);
    std::vector<T> D2_tripletList;
    D2_tripletList.reserve(nD);
    for (int i = 0; i < nD; i++) {
        D2_tripletList.push_back(T(i, i, valD.at(i)));
    }
    D2.setFromTriplets(D2_tripletList.begin(), D2_tripletList.end());
    D2.makeCompressed();

    Spectra::SparseGenMatProd<double> opA(D2mW2);
    Spectra::SparseCholesky<double> opB(D2);

    /// FIXME: @Yupeng maxLabel is something got after watershed
    // int ncv = (maxLabel < 2*nvec) ? maxLabel : 2*nvec;
    log_debug("Spectra nvec: %d", nvec);
    int ncv = 2 * nvec;     /*  nev < ncv <= n (size of matrix) */

    Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseGenMatProd<double>,
                            Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY>
    geigs(&opA, &opB, nvec, ncv);

    geigs.init();
    int nconv = geigs.compute();

    Eigen::VectorXd evalues;
    Eigen::MatrixXd evecs;
    if (geigs.info() == Spectra::SUCCESSFUL) {
        evalues = geigs.eigenvalues();      /* nvec */
        evecs = geigs.eigenvectors();       /* (rows, cols) = (nD, nvec) */
    }
    else {
        log_error("Geigs is failing");
    }
    log_debug("Rows: %ld, Cols: %ld", evecs.rows(), evecs.cols());
    stringstream sStream;
    sStream << evalues;
    log_debug("Generalized eigenvalues found: %s", sStream.str().c_str());
    sStream.str("");
    sStream << evecs.topRows(10);
    log_debug("Generalized eigenvectors found: %s", sStream.str().c_str());

    log_debug("n: %d", n);
    log_debug("Rows: %ld, Cols: %ld", evecs.rows(), evecs.cols());
    log_debug("VectorXd size: %ld", evalues.size());

    /* data interface conversion */
    p_eigVals = new double[nvec];
    for (int i = 0; i < nvec; i++) p_eigVals[i] = evalues(i);
    p_eigVec = new double[n * nvec];    /// TODO: not sure about the conflict between n and nD
    for (int i = 0; i < nvec; i++) {
        for (int j = 0; j < nD; j++) {
            p_eigVec[i * nD + j] = evecs(j, i);
        }
    }

    /* eigenvector computation using Spectra library ends */

    std::vector<Mat*> vect(nvec);
    vect.at(0) = new Mat(ty, tx, CV_64FC1, Scalar::all(0));

    double* data;
    double minVal, maxVal, minXi, maxXi;
    Point minLoc, maxLoc;
    std::string imageName;
    double alpha, beta;
    Mat destIm(ty, tx, CV_64FC1);
    for (int i = 1; i < nvec; i++) {    /* excluding i = 0 */
        vect.at(i) = new Mat(ty, tx, CV_64FC1);
        data = (double*)vect.at(i)->data;
        for (int j = 0; j < n; j++) {
            data[j] = p_eigVec[i * n + j];
        }
        minMaxLoc(*vect.at(i), &minVal, &maxVal, &minLoc, &maxLoc);
        *(vect.at(i)) -= minVal;
        *(vect.at(i)) /= maxVal - minVal;   /// FIXME: check order of precedence
        minMaxLoc(*vect.at(i), &minVal, &maxVal, &minLoc, &maxLoc);
    }

    /* OE parameters */
    int hil = 0;
    int deriv = 1;
    int support = 3;
    double sigma = 1.0;
    int nOrient = 8;
    double dtheta = PI / nOrient;
    vector<int> ch_per = {3, 2, 1, 0, 7, 6, 5, 4};
    double theta = 0.0;
    sPb.resize(nOrient);
    /* initilaize sPb */
    for (int i = 0; i < nOrient; i++) {
        sPb.at(i) = new Mat(ty, tx, CV_64FC1, Scalar::all(0));
    }

    log_info("Filtering the sPb values");

    for (int i = 1; i < nvec; i++) {
        if (p_eigVals[i] >
            /// TODO: check what happens for zero of smallest eigenvalue
            std::numeric_limits<double>::epsilon()) {
            log_debug("Eigenvalue %d: %f", i, p_eigVals[i]);
            *(vect.at(i)) /= sqrt(p_eigVals[i]);
            for (int o = 0; o < nOrient; o++) {
                double theta = dtheta * static_cast<double>(o);
                /// FIXME: hil not handled and also not required
                Mat f = oeFilter(sigma, support, theta, deriv, hil);
                Mat destIm;
                filter2D(*vect.at(i), destIm, -1, f, Point(-1, -1), 0, BORDER_REFLECT_101);
                *(sPb.at(ch_per.at(o))) += abs(destIm);
            }
        }
    }

    /// FIXME: @Yupeng comment
    /*
    double alpha, beta;
    for (int i = 0; i < nOrient; i++) {
        // cout << "here_seg" << endl;
        imageName = outFile + "_sPb_i_" + to_string(i) + ".png";
        minMaxLoc(*sPb.at(i), &minXi, &maxXi);
        alpha = 255.0 / (maxXi - minXi);
        beta = -minXi * 255.0 / (maxXi - minXi);
        Mat destIm;
        sPb.at(i)->convertTo(destIm, CV_8U, alpha, beta);
        imwrite(imageName, destIm);
    }
    cout << "here seg 1" << endl;
    */

    /// FIXME: @Yupeng potential memory leak problems
    delete[] p_eigVals;
    delete[] p_eigVec;
}
