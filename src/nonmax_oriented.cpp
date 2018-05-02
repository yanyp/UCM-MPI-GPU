#include <cv.h>
#include <highgui.h>
#include "log.h"

#include "collections/array_list.hh"
#include "collections/pointers/auto_collection.hh"
#include "io/streams/cout.hh"
#include "io/streams/iomanip.hh"
#include "io/streams/ios.hh"
#include "lang/array.hh"
#include "lang/exceptions/ex_index_out_of_bounds.hh"
#include "lang/exceptions/exception.hh"
#include "lang/pointers/auto_ptr.hh"
#include "math/libraries/lib_image.hh"
#include "math/math.hh"
#include "math/matrices/matrix.hh"

#include "concurrent/threads/thread.hh"

using namespace cv;

using collections::pointers::auto_collection;
using collections::array_list;
using io::streams::cout;
using io::streams::ios;
using io::streams::iomanip::setiosflags;
using io::streams::iomanip::setw;
using lang::array;
using lang::exceptions::exception;
using lang::exceptions::ex_index_out_of_bounds;
using lang::pointers::auto_ptr;
using math::libraries::lib_image;
using math::matrices::matrix;

using concurrent::threads::thread;

/*
 * Convert an mxArray to a matrix.
 */
matrix<> ToMatrix(const Mat* a) {
    unsigned long mrows = static_cast<unsigned long>(a->rows);
    unsigned long ncols = static_cast<unsigned long>(a->cols);
    double* data = NULL;
    matrix<> m(mrows, ncols);
    for (unsigned long r = 0; r < mrows; r++) {
        data = (double*) (a->ptr(r));       /// TODO: check template for double
        for (unsigned long c = 0; c < ncols; c++) {
            m(r, c) = data[c];
        }
    }
    return m;
}

/*
 * Convert a 2D matrix to an mxArray.
 */
Mat* to_openCVMat(const matrix<>& m);
/*
Mat* to_openCVMat(const matrix<>& m) {
    unsigned long mrows = m.size(0);
    unsigned long ncols = m.size(1);
    /// TODO: check if CV_64FC1 means double
    Mat* a = new Mat(static_cast<int>(mrows), static_cast<int>(ncols), CV_64FC1);
    double* data = NULL;
    for (unsigned long r = 0; r < mrows; r++) {
        data = (double*)(a->ptr(r));        /// TODO: check if I need to use ptr<double>
        for (unsigned long c = 0; c < ncols; c++) {
            data[c] = m(r, c);
        }
    }
    return a;
}
*/

void nonmax_oriented(Mat*& nmax, Mat* a, Mat* ind, double nonmax_ori_tol) {
    try {
        /* get arguments */
        matrix<> pb = ToMatrix(a);
        matrix<> pb_ori = ToMatrix(ind);
        /* nonmax suppress pb */
        matrix<> pb_nmax = lib_image::nonmax_oriented_2D(pb, pb_ori, nonmax_ori_tol);
        /* return */
        nmax = to_openCVMat(pb_nmax);

    }
    catch (ex_index_out_of_bounds& e) {
        log_error("Index: %lu | %s", e.index(), e.what());
    }
    catch (exception& e) {
        log_error("%s", e.what());
    }
}
