#include <cmath>
#include <cstdlib>      /* for NULL */
#include <limits>
#include "otherheaders.h"
/// TODO: should I put highgui and other cv file

/// FIXME: is this ok wothout opencv files ^^ see above
using namespace cv;

void savgol_border(cv::Mat* b, cv::Mat* a, cv::Mat* z, double ra, double rb, double theta) {
    ra = std::max(1.5, ra);
    rb = std::max(1.5, rb);

    const double ira2 = std::pow(1.0 / ra, 2);
    const double irb2 = std::pow(1.0 / rb, 2);
    const int wr = (int)std::floor(std::max(ra, rb));
    const double sint = sin(theta);
    const double cost = cos(theta);

    double d0, d1, d2, d3, d4, v0, v1, v2;
    int xi, yi, x, y, cpi;
    double di, ei, zi, di2, detA, invA, param;
    const double eps = std::numeric_limits<double>::epsilon();

    const int w = a->cols;
    const int h = a->rows;

    double* data = NULL;
    for (int cp = 0; cp < w * h; cp++) {
        y = cp % h;
        x = cp / h;
        if ((x >= wr) && (x < (w - wr)) && (y >= wr) && (y < (h - wr))) {
            data = (double*) a->data;
            b->data[cp] = data[cp];
        }
        else {
            d0 = 0;
            d1 = 0;
            d2 = 0;
            d3 = 0;
            d4 = 0;
            v0 = 0;
            v1 = 0;
            v2 = 0;
            for (int u = -wr; u <= wr; u++) {
                xi = x + u;
                if ((xi < 0) || (xi >= w)) {
                    continue;
                }
                for (int v = -wr; v <= wr; v++) {
                    yi = y + v;
                    if ((yi < 0) || (yi >= h)) {
                        continue;
                    }
                    di = -u * sint + v * cost;
                    ei = u * cost + v * sint;
                    if ((di * di * ira2 + ei * ei * irb2) > 1) {
                        continue;
                    }
                    cpi = yi + xi * h;
                    data = (double*)(z->data);
                    zi = data[cpi];     /// TODO: check or split lines
                    di2 = di * di;
                    d0 = d0 + 1;
                    d1 = d1 + di;
                    d2 = d2 + di2;
                    d3 = d3 + di * di2;
                    d4 = d4 + di2 * di2;
                    v0 = v0 + zi;
                    v1 = v1 + zi * di;
                    v2 = v2 + zi * di2;
                }
            }
            detA = -d2 * d2 * d2 + 2 * d1 * d2 * d3 - d0 * d3 * d3 - d1 * d1 * d4 + d0 * d2 * d4;
            if (detA > eps) {
                b->data[cp] = ((-d3 * d3 + d2 * d4) * v0 + (d2 * d3 - d1 * d4) * v1 + (-d2 * d2 + d1 * d3) * v2) / detA;
            }
        }
    }
}
