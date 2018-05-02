#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "affinity.hh"
#include "ic.hh"
#include "smatrix.hh"
#include "util/array.hh"
#include "util/configure.hh"
#include "util/exception.hh"
#include "util/string.hh"
#include "util/types.hh"
#include "util/util.hh"

#include "otherheaders.h"

using namespace std;

void buildW(vector<double> &valD, vector<double> &val, vector<int> &I, vector<int> &J, cv::Mat &l1, cv::Mat &l2) {
    int dThresh = 5;
    float sigma = 0.1;

    /*
    Group::DualLattice boundaries;
    cout << "========Printing l1 ============" << endl;
    double *ptrData = NULL;
    boundaries.H.resize(l1.rows, l1.cols);
    for (int i = 0; i < l1.rows; i++) {
        ptrData = reinterpret_cast<double *>(l1.ptr(i));
        for (int j = 0; j < l1.cols; j++) {
            boundaries.H(i, j) = ptrData[j];
            cout << boundaries.H(i, j) << ",";
        }
        cout << endl;
    }
    cout << "========Ended Printing l1 ============" << endl;

    cout << "========Printing l2 ============" << endl;
    boundaries.V.resize(l2.rows, l2.cols);
    for (int i = 0; i < l2.rows; i++) {
        ptrData = reinterpret_cast<double *>(l2.ptr(i));
        for (int j = 0; j < l2.cols; j++) {
            boundaries.V(i, j) = ptrData[j];
            cout << boundaries.V(i, j) << ",";
        }
        cout << endl;
    }
    cout << "========Ended Printing l2 ============" << endl;
    */

    /* copy edge info into lattice struct */
    Group::DualLattice boundaries;

    double *H = reinterpret_cast<double *>(l1.data);
    int H_h = l1.cols;
    int H_w = l1.rows;
    log_debug("H_h = %d H_w = %d", H_h, H_w);
    boundaries.H.resize(H_h, H_w);
    /*
    for (int i = 0; i < H_h; i++) {
        for (int j = 0; j < H_w; j++) {
            boundaries.H(i, j) = H[i * H_w + j];
            cout << boundaries.H(i, j) << ",";
        }
        cout << endl;
    }

    for (int j = 0; j < H_w; j++) {
        for (int i = 0; i < H_h; i++) {
            boundaries.H(i, j) = H[j * H_h + i];
            cout << boundaries.H(i, j) << ",";
        }
        cout << endl;
    }
    */

    for (int i = 0; i < H_h; i++) {
        for (int j = 0; j < H_w; j++) {
            boundaries.H(i, j) = H[j * H_h + i];
            // cout << boundaries.H(i,j) << ",";
        }
        // cout << endl;
    }

    double *V = reinterpret_cast<double *>(l2.data);
    int V_h = l2.cols;
    int V_w = l2.rows;
    // cout << "V_h = " << V_h << "V_w = " << V_w << endl;
    boundaries.V.resize(V_h, V_w);
    /*
    for (int i = 0; i < V_h; i++) {
        for (int j = 0; j < V_w; j++) {
            boundaries.V(i, j) = V[i * V_w + j];
            cout << boundaries.V(i, j) << ",";
        }
        cout << endl;
    }
    for (int j = 0; j < V_w; j++) {
        for (int i = 0; i < V_h; i++) {
            boundaries.V(i, j) = V[j * V_h + i];
            cout << boundaries.V(i, j) << ",";
        }
        cout << endl;
    }
    */

    for (int i = 0; i < V_h; i++) {
        for (int j = 0; j < V_w; j++) {
            boundaries.V(i, j) = V[j * V_h + i];
            // cout << boundaries.V(i,j) << ",";
        }
        // cout << endl;
    }

    boundaries.width = boundaries.H.size(0);
    boundaries.height = boundaries.V.size(1);

    Group::SupportMap ic;
    Group::computeSupport(boundaries, dThresh, 1.0f, ic);

    SMatrix *W = NULL;
    Group::computeAffinities2(ic, sigma, dThresh, &W);
    if (W == NULL) {
        log_error("Compute affinities failed");
    }

    int nnz = 0;
    for (int i = 0; i < W->n; i++) {
        nnz += W->nz[i];
    }
    /*
    cout << "Here is nnz " << nnz << endl;

    for (int i = 0; i < W->n; i++) {
        for (int j = 0; j < W->nz[i]; j++) {
            cout << W->values[i][j] << ",";
        }
        cout << endl;
    }

    cout << "val size " << val.size() << endl;
    cout << "I size " << I.size() << endl;
    cout << "J size " << J.size() << endl;
    */

    val.reserve(nnz);   /* note we use reserve and not resize so that we use push_back() */
    I.reserve(nnz);
    J.reserve(nnz);
    valD.resize(W->n);  /* note resize here because we already know nnz = n */

    /*
    cout << "val size " << val.size() << endl;
    cout << "I size " << I.size() << endl;
    cout << "J size " << J.size() << endl;
    */

    int ct = 0;
    int tempCol = 0;
    double tempVal = 0.0;
    int flag = 0;
    for (int row = 0; row < W->n; row++) {
        valD.at(row) = 0;
        for (int i = 0; i < W->nz[row]; i++) {
            /* add one for matlab indexing */
            /*
            I.at(ct+i) = static_cast<double>(row + 1);
            J.at(ct+i) = static_cast<double>(W->col[row][i] + 1);
            val.at(ct+i) = static_cast<double>(W->values[row][i]);
            */
            tempCol = W->col[row][i];
            tempVal = static_cast<double>(W->values[row][i]);
            valD.at(row) += tempVal;
            /*
            if (tempCol == row) flag = 1;
            if (tempCol > row && flag == 0) {
                I.push_back(row);
                J.push_back(row);
                val.push_back(static_cast<double>(1.0));
                flag = 1;
            }

            if (tempCol >= row && flag == 1) {
                I.at(ct+i) = row;
                J.at(ct+i) = W->col[row][i];
                val.at(ct+i) = static_cast<double>(W->values[row][i]);
                I.push_back(row);
                J.push_back(tempCol);
                val.push_back(tempVal);
            }
            */

            /// FIXME: if i remove the if for Upper Triangle matrix then arpack throws error
            ///     that matrix has inconsistencies. So should I use nonsymmetric arpack? would it save time in
            ///     buildW. it does save one second. what does matlab use?
            if (tempCol >= row) {
                /*
                I.at(ct+i) = row;
                J.at(ct+i) = W->col[row][i];
                val.at(ct+i) = static_cast<double>(W->values[row][i]);
                */
                I.push_back(row);
                J.push_back(tempCol);
                val.push_back(tempVal);
            }
        }
        ct = ct + W->nz[row];
        flag = 0;
    }

    /*
    // ========================== test ==================================
    val.resize(nnz);  // note we use reserve and not resize so that we use push_back()
    I.resize(nnz);
    J.resize(nnz);

    int ct = 0;
    for (int row = 0; row < W->n; row++) {
        for (int i = 0; i < W->nz[row]; i++) {
            I.at(ct + i) = (row);  // add one for matlab indexing
            J.at(ct + i) = (W->col[row][i]);
            val.at(ct + i) = static_cast<double>(W->values[row][i]);
        }
        ct = ct + W->nz[row];
    }
    // ========================= test ended =============================

    cout << "count  = " << ct << endl;
    cout << "nnz  = " << nnz << endl;
    cout << numeric_limits<int>::max() << endl;
    cout << sizeof(int) << endl;
    cout << W->n << endl;
    cout << valD.size() << endl;
    cout << val.size() << endl;

    cout << "=====Display matrix =======" << endl;
    for (int i = 0; i < val.size(); i++)
        cout << "(" << I.at(i) + 1 << "," << J.at(i) + 1 << ") =  " << val.at(i) << endl;
    cout << "=====Display ended =======" << endl;
    */

    delete W;
}
