/*
Source code for computing ultrametric contour maps based on average boundary strength, as described in :

P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik. From contours to regions: An empirical evaluation. In CVPR, 2009.

Pablo Arbelaez <arbelaez@eecs.berkeley.edu>
March 2009.
*/

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <vector>
#include "head.h"
#include "log.h"

using namespace std;
using namespace cv;

/*#include "mex.h"*/

/*************************************************************/

/******************************************************************************/

#ifndef ORDER_NODE_H
#define ORDER_NODE_H

class OrderNode {
   public:
    double energy;
    int region1;
    int region2;

    OrderNode() {
        energy = 0.0;
        region1 = 0;
        region2 = 0;
    }

    OrderNode(const double& e, const int& _region1, const int& _region2) {
        energy = e;
        region1 = _region1;
        region2 = _region2;
    }

    ~OrderNode() {}
    // LEXICOGRAPHIC ORDER on priority queue: (energy,label)
    bool operator<(const OrderNode& x) const {
        return ((energy > x.energy) || ((energy == x.energy) && (region1 > x.region1)) ||
                ((energy == x.energy) && (region1 == x.region1) && (region2 > x.region2)));
    }
};

#endif

/******************************************************************************/

#ifndef NEIGHBOR_REGION_H
#define NEIGHBOR_REGION_H

class NeighborRegion {
   public:
    double energy;
    double totalPb;
    double bdryLength;

    NeighborRegion() {
        energy = 0.0;
        totalPb = 0.0;
        bdryLength = 0.0;
    }

    NeighborRegion(const NeighborRegion& v) {
        energy = v.energy;
        totalPb = v.totalPb;
        bdryLength = v.bdryLength;
    }

    NeighborRegion(const double& en, const double& tt, const double& bor) {
        energy = en;
        totalPb = tt;
        bdryLength = bor;
    }

    ~NeighborRegion() {}
};

#endif

/******************************************************************************/

#ifndef BDRY_ELEMENT_H
#define BDRY_ELEMENT_H

class BdryElement {
   public:
    int coord;
    int ccNeigh;

    // BdryElement(){}

    BdryElement(const int& c, const int& v) {
        coord = c;
        ccNeigh = v;
    }

    BdryElement(const BdryElement& n) {
        coord = n.coord;
        ccNeigh = n.ccNeigh;
    }

    ~BdryElement() {}

    bool operator==(const BdryElement& n) const {
        return ((coord == n.coord) && (ccNeigh == n.ccNeigh));
    }
    // LEXICOGRAPHIC ORDER: (ccNeigh, coord)
    bool operator<(const BdryElement& n) const {
        return ((ccNeigh < n.ccNeigh) || ((ccNeigh == n.ccNeigh) && (coord < n.coord)));
    }
};

#endif

/******************************************************************************/

#ifndef REGION_H
#define REGION_H

class Region {
   public:
    list<int> elements;
    map<int, NeighborRegion, less<int> > neighbors;
    list<BdryElement> boundary;

    Region() {}

    Region(const int& l) {
        elements.push_back(l);
    }

    ~Region() {}

void Merge(Region& r, int* labels, const int& label, double* ucm, const double& saliency, const int& son,
               const int& tx);
};

void Region::Merge(Region& r, int* labels, const int& label, double* ucm, const double& saliency, const int& son,
                   const int& tx) {
    /* 			I. BOUNDARY        */

    // 	Ia. update father's boundary
    list<BdryElement>::iterator itrb, itrb2;
    itrb = boundary.begin();
    while (itrb != boundary.end()) {
        if (labels[(*itrb).ccNeigh] == son) {
            itrb2 = itrb;
            ++itrb;
            boundary.erase(itrb2);
        }
        else {
            ++itrb;
        }
    }

    int coordContour;

    //	Ib. move son's boundary to father
    for (itrb = r.boundary.begin(); itrb != r.boundary.end(); ++itrb) {
        if (ucm[(*itrb).coord] < saliency) {
            ucm[(*itrb).coord] = saliency;
        }

        if (labels[(*itrb).ccNeigh] != label) {
            boundary.push_back(BdryElement(*itrb));
        }
    }
    r.boundary.erase(r.boundary.begin(), r.boundary.end());

    /* 			II. ELEMENTS      */

    for (list<int>::iterator p = r.elements.begin(); p != r.elements.end(); ++p) {
        labels[*p] = label;
    }
    elements.insert(elements.begin(), r.elements.begin(), r.elements.end());
    r.elements.erase(r.elements.begin(), r.elements.end());

    /* 			III. NEIGHBORS        */

    map<int, NeighborRegion, less<int>>::iterator itr, itr2;

    // 	IIIa. remove inactive neighbors from father
    itr = neighbors.begin();
    while (itr != neighbors.end()) {
        if (labels[(*itr).first] != (*itr).first) {
            itr2 = itr;
            ++itr;
            neighbors.erase(itr2);
        }
        else {
            ++itr;
        }
    }

    // 	IIIb. remove inactive neighbors from son y and neighbors belonging to father
    itr = r.neighbors.begin();
    while (itr != r.neighbors.end()) {
        if ((labels[(*itr).first] != (*itr).first) || (labels[(*itr).first] == label)) {
            itr2 = itr;
            ++itr;
            r.neighbors.erase(itr2);
        }
        else {
            ++itr;
        }
    }
}

#endif

/*************************************************************/

void CompleteContourMap(double* ucm, const int& txc, const int& tyc)
/* complete contour map by max strategy on Khalimsky space  */
{
    int vx[4] = {1, 0, -1, 0};
    int vy[4] = {0, 1, 0, -1};
    int nxp, nyp, cv;
    double maximo;

    for (int x = 0; x < txc; x = x + 2)
        for (int y = 0; y < tyc; y = y + 2) {
            maximo = 0.0;
            for (int v = 0; v < 4; v++) {
                nxp = x + vx[v];
                nyp = y + vy[v];
                cv = nxp + nyp * txc;
                if ((nyp >= 0) && (nyp < tyc) && (nxp < txc) && (nxp >= 0) && (maximo < ucm[cv])) {
                    maximo = ucm[cv];
                }
            }
            ucm[x + y * txc] = maximo;
        }
}

/***************************************************************************************************************************/
void ComputeUcm(double* localBoundaries, int* initialPartition, const int& totcc, double* ucm, const int& tx,
                 const int& ty) {
    // I. INITIATE
    int p, c;
    int* labels = new int[totcc];

    for (c = 0; c < totcc; c++) {
        labels[c] = c;
    }

    // II. ULTRAMETRIC
    Region* R = new Region[totcc];
    priority_queue<OrderNode, vector<OrderNode>, less<OrderNode> > mergingQueue;
    double totalPb, totalBdry, dissimilarity;
    int v, px;

    for (p = 0; p < (2 * tx + 1) * (2 * ty + 1); p++) {
        ucm[p] = 0.0;
    }

    // INITIATE REGI0NS
    for (c = 0; c < totcc; c++) {
        R[c] = Region(c);
    }

    // INITIATE UCM
    int vx[4] = {1, 0, -1, 0};
    int vy[4] = {0, 1, 0, -1};
    int nxp, nyp, cnp, xp, yp, label;

    for (p = 0; p < tx * ty; p++) {
        // cout << "p = " << p << ",\t";
        xp = p % tx;
        yp = p / tx;
        for (v = 0; v < 4; v++) {
            nxp = xp + vx[v];
            nyp = yp + vy[v];
            cnp = nxp + nyp * tx;
            if ((nyp >= 0) && (nyp < ty) && (nxp < tx) && (nxp >= 0) &&
                (initialPartition[cnp] != initialPartition[p])) {
                // cout << v << " ";
                // cout << "(" << cnp << "-" << initialPartition[cnp] << "," << p << "-" << initialPartition[p] << "),
                // ";
                R[initialPartition[p]].boundary.push_back(
                    BdryElement((xp + nxp + 1) + (yp + nyp + 1) * (2 * tx + 1), initialPartition[cnp]));
            }
        }
        // cout << "\tp = " << p << ", tx*ty = " << tx*ty << endl;
    }

    // INITIATE mergingQueue
    list<BdryElement>::iterator itrb;
    for (c = 0; c < totcc; c++) {
        R[c].boundary.sort();

        label = (*R[c].boundary.begin()).ccNeigh;
        totalBdry = 0.0;
        totalPb = 0.0;

        for (itrb = R[c].boundary.begin(); itrb != R[c].boundary.end(); ++itrb) {
            if ((*itrb).ccNeigh == label) {
                totalBdry++;
                totalPb += localBoundaries[(*itrb).coord];
            }
            else {
                R[c].neighbors[label] = NeighborRegion(totalPb / totalBdry, totalPb, totalBdry);
                if (label > c) {
                    mergingQueue.push(OrderNode(totalPb / totalBdry, c, label));
                }
                label = (*itrb).ccNeigh;
                totalBdry = 1.0;
                totalPb = localBoundaries[(*itrb).coord];
            }
        }
        R[c].neighbors[label] = NeighborRegion(totalPb / totalBdry, totalPb, totalBdry);
        if (label > c) {
            mergingQueue.push(OrderNode(totalPb / totalBdry, c, label));
        }
    }

    // MERGING
    OrderNode minor;
    int father, son;
    map<int, NeighborRegion, less<int> >::iterator itr;
    double currentEnergy = 0.0;

    while (!mergingQueue.empty()) {
        minor = mergingQueue.top();
        mergingQueue.pop();
        if ((labels[minor.region1] == minor.region1) && (labels[minor.region2] == minor.region2) &&
            (minor.energy == R[minor.region1].neighbors[minor.region2].energy)) {
            if (currentEnergy <= minor.energy) {
                currentEnergy = minor.energy;
            }
            else {
                log_error("ERROR: currentEnergy = %f minor.energy = %f", currentEnergy, minor.energy);
                delete[] R;
                delete[] labels;
                /*			mexErrMsgTxt(" BUG: THIS IS NOT AN ULTRAMETRIC !!! ");*/
            }

            dissimilarity = R[minor.region1].neighbors[minor.region2].totalPb /
                            R[minor.region1].neighbors[minor.region2].bdryLength;

            if (minor.region1 < minor.region2) {
                son = minor.region1;
                father = minor.region2;
            }
            else {
                son = minor.region2;
                father = minor.region1;
            }

            R[father].Merge(R[son], labels, father, ucm, dissimilarity, son, tx);

            // move and update neighbors
            while (R[son].neighbors.size() > 0) {
                itr = R[son].neighbors.begin();

                R[father].neighbors[(*itr).first].totalPb += (*itr).second.totalPb;
                R[(*itr).first].neighbors[father].totalPb += (*itr).second.totalPb;

                R[father].neighbors[(*itr).first].bdryLength += (*itr).second.bdryLength;
                R[(*itr).first].neighbors[father].bdryLength += (*itr).second.bdryLength;

                R[son].neighbors.erase(itr);
            }

            // update mergingQueue
            for (itr = R[father].neighbors.begin(); itr != R[father].neighbors.end(); ++itr) {
                dissimilarity =
                    R[father].neighbors[(*itr).first].totalPb / R[father].neighbors[(*itr).first].bdryLength;

                mergingQueue.push(OrderNode(dissimilarity, (*itr).first, father));
                R[father].neighbors[(*itr).first].energy = dissimilarity;
                R[(*itr).first].neighbors[father].energy = dissimilarity;
            }
        }
    }

    CompleteContourMap(ucm, 2 * tx + 1, 2 * ty + 1);

    delete[] R;
    delete[] labels;
}

/*************************************************************************************************/
// void mexFunction(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[])
cv::Mat UcmMeanPb(const cv::Mat& wsWt2, const cv::Mat& labels) {
    /* if (nrhs != 2) mexErrMsgTxt("INPUT: (localBoundaries, initialPartition) ");
     if (nlhs != 1) mexErrMsgTxt("OUTPUT: [ucm] ");*/

    /* double* localBoundaries = mxGetPr(prhs[0]);
     double* pi = mxGetPr(prhs[1]);*/

    // size of original image
    /* int fil = mxGetM(prhs[1]);
     int col = mxGetN(prhs[1]);*/
    int fil = labels.rows;
    int col = labels.cols;

    double* localBoundaries = new double[(2 * fil + 1) * (2 * col + 1)];
    for (int px = 0; px < (2 * fil + 1) * (2 * col + 1); px++) {
        // cout << "px = " << px << endl;
        localBoundaries[px] = wsWt2.at<double>(px % (2 * fil + 1), px / (2 * fil + 1));
    }

    int totcc = -1;
    /* int* initialPartition = new int[fil*col];*/
    int* initialPartition = new int[fil * col];
    for (int px = 0; px < fil * col; px++) {
        initialPartition[px] = labels.at<int>(px % fil, px / fil);
        if (totcc < initialPartition[px]) {
            totcc = initialPartition[px];
        }
    }
    /* if (totcc < 0) mexErrMsgTxt("\n ERROR : number of connected components < 0 : \n");*/
    totcc++;

    /* plhs[0] = mxCreateDoubleMatrix(2*fil+1, 2*col+1, mxREAL);
     double* ucm = mxGetPr(plhs[0]);*/
    double* ucm = new double[(2 * fil + 1) * (2 * col + 1)];

    ComputeUcm(localBoundaries, initialPartition, totcc, ucm, fil, col);

    delete[] initialPartition;

    // TODO: return a mat to show
    Mat ucmMat(2 * fil + 1, 2 * col + 1, CV_64F);
    for (int i = 0; i < ucmMat.rows; i++) {
        for (int j = 0; j < ucmMat.cols; j++) {
            ucmMat.at<double>(i, j) = ucm[i + j * ucmMat.rows];
        }
    }
    return ucmMat;
}
