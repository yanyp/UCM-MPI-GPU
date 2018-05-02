#ifndef SRC_WATERSHED_H_
#define SRC_WATERSHED_H_

using namespace std;
using namespace cv;

#ifndef coordinates_
#define coordinates_
struct coordinates {
    int row;
    int col;
};
#endif

Mat Matrix2Image(Mat& Pb_8U);
vector<coordinates> find_8U(Mat& matrix, int level);
Mat watershed_yupeng(Mat& A, int conn);

#endif
