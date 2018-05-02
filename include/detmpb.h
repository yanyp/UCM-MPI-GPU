#ifndef DETMPB_H_
#define DETMPB_H_

#include <cv.h>
#include <highgui.h>
#include <vector>

class detmpb {
   private:
    std::vector<cv::Mat *> *bg1, *bg2, *bg3, *cga1, *cga2, *cga3, *cgb1, *cgb2, *cgb3, *tg1, *tg2, *tg3;
    cv::Mat* textons;
    // int nRows, nCols, nChan;

    /*
    void allocDouble3DArray(double *** &x, int nRows, int nCols, int nChan);
    void deleteDouble3DArray(double *** &x, int nRows, int nCols);
    */
   public:
    detmpb();
    void AssignPtrsUsingShallowCopy(cv::Mat*** plhs_bg, cv::Mat*** plhs_cga, cv::Mat*** plhs_cgb, cv::Mat*** plhs_tg,
                                    cv::Mat* plhs_textons);

    std::vector<cv::Mat*>* GetBg1();
    std::vector<cv::Mat*>* GetBg2();
    std::vector<cv::Mat*>* GetBg3();

    std::vector<cv::Mat*>* GetCga1();
    std::vector<cv::Mat*>* GetCga2();
    std::vector<cv::Mat*>* GetCga3();

    std::vector<cv::Mat*>* GetCgb1();
    std::vector<cv::Mat*>* GetCgb2();
    std::vector<cv::Mat*>* GetCgb3();

    std::vector<cv::Mat*>* GetTg1();
    std::vector<cv::Mat*>* GetTg2();
    std::vector<cv::Mat*>* GetTg3();

    cv::Mat* GetTextons();

    void SaveObject(std::string outFile);
    void LoadObject(std::string outFile);
    void SaveVectorMats(std::string outFile, std::string suffix, std::vector<cv::Mat*>* object);
    void LoadVectorMats(std::string outFile, std::string suffix, std::vector<cv::Mat*>* object);

    virtual ~detmpb();
};

#endif
