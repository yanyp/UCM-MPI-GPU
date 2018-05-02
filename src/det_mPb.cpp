#include <cv.h>
#include <highgui.h>

#include "detmpb.h"
#include "otherheaders.h"

using namespace cv;

void det_mPb(detmpb& returnVals, Mat& im) {
    vector<Mat> channels(3);
    split(im, channels);  /// FIXME: do I need nChan above?

    Mat* plhs0 = NULL;
    Mat*** plhs_bg = NULL;
    Mat*** plhs_cga = NULL;
    Mat*** plhs_cgb = NULL;
    Mat*** plhs_tg = NULL;

    mex_pb_parts_final_selected(plhs0, plhs_bg, plhs_cga, plhs_cgb, plhs_tg, channels[0], channels[1], channels[2]);

    returnVals.AssignPtrsUsingShallowCopy(plhs_bg, plhs_cga, plhs_cgb, plhs_tg, plhs0);
}
