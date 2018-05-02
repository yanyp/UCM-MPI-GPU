#include "detmpb.h"
#include "log.h"

detmpb::detmpb() {
    /*
    bg1.reserve(8);
    bg2.reserve(8);
    bg3.reserve(8);

    cga1.reserve(8);
    cga2.reserve(8);
    cga3.reserve(8);
    cgb1.reserve(8);
    cgb2.reserve(8);
    cgb3.reserve(8);

    tg1.reserve(8);
    tg2.reserve(8);
    tg3.reserve(8);

    bg1.resize(8);
    bg2.resize(8);
    bg3.resize(8);

    cga1.resize(8);
    cga2.resize(8);
    cga3.resize(8);
    cgb1.resize(8);
    cgb2.resize(8);
    cgb3.resize(8);

    tg1.resize(8);
    tg2.resize(8);
    tg3.resize(8);
    */
    bg1 = new std::vector<cv::Mat *>(8);
    bg2 = new std::vector<cv::Mat *>(8);
    bg3 = new std::vector<cv::Mat *>(8);

    cga1 = new std::vector<cv::Mat *>(8);
    cga2 = new std::vector<cv::Mat *>(8);
    cga3 = new std::vector<cv::Mat *>(8);

    cgb1 = new std::vector<cv::Mat *>(8);
    cgb2 = new std::vector<cv::Mat *>(8);
    cgb3 = new std::vector<cv::Mat *>(8);

    tg1 = new std::vector<cv::Mat *>(8);
    tg2 = new std::vector<cv::Mat *>(8);
    tg3 = new std::vector<cv::Mat *>(8);

    textons = NULL;
}

void detmpb::AssignPtrsUsingShallowCopy(cv::Mat ***plhs_bg, cv::Mat ***plhs_cga, cv::Mat ***plhs_cgb,
                                        cv::Mat ***plhs_tg, cv::Mat *plhs_textons) {
    for (int rnum = 0; rnum < 3; rnum++) {
        for (int n = 0; n < 8; n++) {
            switch (rnum) {
                case 0:
                    bg1->at(n) = plhs_bg[rnum][n];
                    cga1->at(n) = plhs_cga[rnum][n];
                    cgb1->at(n) = plhs_cgb[rnum][n];
                    tg1->at(n) = plhs_tg[rnum][n];
                    break;
                case 1:
                    bg2->at(n) = plhs_bg[rnum][n];
                    cga2->at(n) = plhs_cga[rnum][n];
                    cgb2->at(n) = plhs_cgb[rnum][n];
                    tg2->at(n) = plhs_tg[rnum][n];
                    break;
                case 2:
                    bg3->at(n) = plhs_bg[rnum][n];
                    cga3->at(n) = plhs_cga[rnum][n];
                    cgb3->at(n) = plhs_cgb[rnum][n];
                    tg3->at(n) = plhs_tg[rnum][n];
                    break;
            }
        }
    }
    /*
    for (int rnum = 0; rnum < 3; rnum++) {
        for (int n = 0; n < 8; n++) {
            switch (rnum) {
                case 0:
                    cga1.at(n) = plhs_cga[rnum][n];
                    break;
                case 1:
                    cga2.at(n) = plhdet_mPbs_cga[rnum][n];
                    break;
                case 2:
                    cga3.at(n) = plhs_cga[rnum][n];
                    break;
            }
        }
    }
    for (int rnum = 0; rnum < 3; rnum++) {
        for (int n = 0; n < 8; n++) {
            switch (rnum) {
                case 0:
                    cgb1.at(n) = plhs_cgb[rnum][n];
                    break;
                case 1:
                    cgb2.at(n) = plhs_cgb[rnum][n];
                    break;
                case 2:
                    cgb3.at(n) = plhs_cgb[rnum][n];
                    break;
            }
        }
    }
    for (int rnum = 0; rnum < 3; rnum++) {
        for (int n = 0; n < 8; n++) {
            switch (rnum) {
                case 0:
                    tg1.at(n) = plhs_tg[rnum][n];
                    break;
                case 1:
                    tg2.at(n) = plhs_tg[rnum][n];
                    break;
                case 2:
                    tg3.at(n) = plhs_tg[rnum][n];
                    break;
            }
        }
    }
    */
    textons = plhs_textons;
}

std::vector<cv::Mat *> *detmpb::GetBg1() {
    return bg1;
}

std::vector<cv::Mat *> *detmpb::GetBg2() {
    return bg2;
}

std::vector<cv::Mat *> *detmpb::GetBg3() {
    return bg3;
}

std::vector<cv::Mat *> *detmpb::GetCga1() {
    return cga1;
}

std::vector<cv::Mat *> *detmpb::GetCga2() {
    return cga2;
}

std::vector<cv::Mat *> *detmpb::GetCga3() {
    return cga3;
}

std::vector<cv::Mat *> *detmpb::GetCgb1() {
    return cgb1;
}

std::vector<cv::Mat *> *detmpb::GetCgb2() {
    return cgb2;
}

std::vector<cv::Mat *> *detmpb::GetCgb3() {
    return cgb3;
}

std::vector<cv::Mat *> *detmpb::GetTg1() {
    return tg1;
}

std::vector<cv::Mat *> *detmpb::GetTg2() {
    return tg2;
}

std::vector<cv::Mat *> *detmpb::GetTg3() {
    return tg3;
}

cv::Mat *detmpb::GetTextons() {
    return textons;
}

detmpb::~detmpb() {
    /// TODO Auto-generated destructor stub
    delete bg1;
    delete bg2;
    delete bg3;
    delete cga1;
    delete cga2;
    delete cga3;
    delete cgb1;
    delete cgb2;
    delete cgb3;
    delete tg1;
    delete tg2;
    delete tg3;
}

void detmpb::SaveObject(std::string outFile) {
    /* bg1-3, cga1-3, cgb1-3, tg1-3, NO texton */
    log_info("Saving detmpb_object to file series: %s", outFile.c_str());
    SaveVectorMats(outFile, "bg1", bg1);
    SaveVectorMats(outFile, "bg2", bg2);
    SaveVectorMats(outFile, "bg3", bg3);
    SaveVectorMats(outFile, "cga1", cga1);
    SaveVectorMats(outFile, "cga2", cga2);
    SaveVectorMats(outFile, "cga3", cga3);
    SaveVectorMats(outFile, "cgb1", cgb1);
    SaveVectorMats(outFile, "cgb2", cgb2);
    SaveVectorMats(outFile, "cgb3", cgb3);
    SaveVectorMats(outFile, "tg1", tg1);
    SaveVectorMats(outFile, "tg2", tg2);
    SaveVectorMats(outFile, "tg3", tg3);
}

void detmpb::LoadObject(std::string outFile) {
    log_info("Loading detmpb_object from file series: %s", outFile.c_str());
    LoadVectorMats(outFile, "bg1", bg1);
    LoadVectorMats(outFile, "bg2", bg2);
    LoadVectorMats(outFile, "bg3", bg3);
    LoadVectorMats(outFile, "cga1", cga1);
    LoadVectorMats(outFile, "cga2", cga2);
    LoadVectorMats(outFile, "cga3", cga3);
    LoadVectorMats(outFile, "cgb1", cgb1);
    LoadVectorMats(outFile, "cgb2", cgb2);
    LoadVectorMats(outFile, "cgb3", cgb3);
    LoadVectorMats(outFile, "tg1", tg1);
    LoadVectorMats(outFile, "tg2", tg2);
    LoadVectorMats(outFile, "tg3", tg3);
}

void detmpb::SaveVectorMats(std::string outFile, std::string suffix, std::vector<cv::Mat *> *object) {
    cv::FileStorage fs(outFile + "_" + suffix + ".yml", cv::FileStorage::WRITE);
    for (int o = 0; o < object->size(); o++) {
        fs << suffix + "_" + std::to_string(o) << *object->at(o);
    }
    fs.release();
}

void detmpb::LoadVectorMats(std::string outFile, std::string suffix, std::vector<cv::Mat *> *object) {
    cv::FileStorage fs(outFile + "_" + suffix + ".yml", cv::FileStorage::READ);
    for (int o = 0; o < object->size(); o++) {
        fs[suffix + "_" + std::to_string(o)] >> *object->at(o);
    }
    fs.release();
}
