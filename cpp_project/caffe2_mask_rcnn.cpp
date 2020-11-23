#include <iostream>
#include <string>
#include <cassert>
#include <chrono>
#include <dirent.h>
#include <c10/util/Flags.h>

#include <caffe2/core/blob.h>
#include <caffe2/core/init.h>
#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace caffe2;
using namespace cv;

vector<string> classes;
vector<cv::Scalar> colors;

void getFileNames(string& path,vector<string>& filenames);

void drawBox(Mat &frame, int classId, float conf, Rect box, Mat &objectMask);

int main() {
    // init colors of annotation
    RNG rng1;
    colors.emplace_back(rng1.uniform(0, 255), rng1.uniform(0, 255), rng1.uniform(0, 255), 255.0);
    rng1.next();
    colors.emplace_back(rng1.uniform(0, 255), rng1.uniform(0, 255), rng1.uniform(0, 255), 255.0);

    // init classes label of annotation
    classes.emplace_back("fissure");
    classes.emplace_back("water");

    // model file
    string predictNetPath = "../../python_project/output/model.pb";
    // weights file
    string initNetPath = "../../python_project/output/model_init.pb";

    // initialize net and workspace
    caffe2::NetDef initNet_, predictNet_;
    CAFFE_ENFORCE(ReadProtoFromFile(initNetPath, &initNet_));
    CAFFE_ENFORCE(ReadProtoFromFile(predictNetPath, &predictNet_));

    Workspace workSpace;
    for (auto &str : predictNet_.external_input()) {
        workSpace.CreateBlob(str);
    }
    CAFFE_ENFORCE(workSpace.CreateNet(predictNet_));
    CAFFE_ENFORCE(workSpace.RunNetOnce(initNet_));

    const int batch = 1;
    const int channels = 3;

    // target image list
    vector<string> filenames;
    // target images folder
    string folder_path = "../../python_project/images/train";
    // load and fill the path of files to vector
    getFileNames(folder_path,  filenames);

    auto i = 0;

    // loop and detect images
    for (const auto& image_path : filenames) {
        // resize our input images to fit with model
        cv::Size dSize = cv::Size(800, 800);

        // load image from file path, the size of image maybe w:2050 h:2411
        Mat mat_temp = imread(image_path, IMREAD_COLOR);

        // define a new Mat object with w:800 h:800
        Mat mat_input(dSize, CV_32F);

        // resize input image, from w:2050 h:2411 to w:800 h:800
        resize(mat_temp, mat_input, dSize, INTER_LINEAR);

        // release the image matrix from imread
        mat_temp.release();

        const int height = mat_input.rows;
        const int width = mat_input.cols;
        // FPN models require divisibility of 32
        assert(height % 32 == 0 && width % 32 == 0);

        // setup inputs
        auto data = BlobGetMutableTensor(workSpace.GetBlob("data"), caffe2::CPU);
        data->Resize(batch, channels, height, width);
        auto *ptr = data->mutable_data<float>();
        // HWC to CHW
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < height * width; ++i) {
                ptr[c * height * width + i] = static_cast<float>(mat_input.data[3 * i + c]);
            }
        }

        auto im_info = BlobGetMutableTensor(workSpace.GetBlob("im_info"), caffe2::CPU);
        im_info->Resize(batch, 3);

        auto *im_info_ptr = im_info->mutable_data<float>();
        im_info_ptr[0] = height;
        im_info_ptr[1] = width;
        im_info_ptr[2] = 1.0;

        // run the network
        CAFFE_ENFORCE(workSpace.RunNet(predictNet_.name()));

        // run 3 more times to benchmark
//        int N_benchmark = 3;
        int N_benchmark = 1;

        auto start_time = chrono::high_resolution_clock::now();
        for (int i = 0; i < N_benchmark; ++i) {
            CAFFE_ENFORCE(workSpace.RunNet(predictNet_.name()));
        }
        auto end_time = chrono::high_resolution_clock::now();
        auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                .count();
        cout << "Latency (should vary with different inputs): "
             << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

        // parse Mask R-CNN outputs
        caffe2::Tensor bbox(
                workSpace.GetBlob("bbox_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
        caffe2::Tensor scores(
                workSpace.GetBlob("score_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
        caffe2::Tensor labels(
                workSpace.GetBlob("class_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
        caffe2::Tensor mask_probs(
                workSpace.GetBlob("mask_fcn_probs")->Get<caffe2::Tensor>(), caffe2::CPU);
        cout << "bbox:" << bbox.DebugString() << endl;
        cout << "scores:" << scores.DebugString() << endl;
        cout << "labels:" << labels.DebugString() << endl;
        cout << "mask_probs: " << mask_probs.DebugString() << endl;

        int num_instances = bbox.sizes()[0];
        for (int i = 0; i < num_instances; ++i) {
            float score = scores.data<float>()[i];
            if (score < 0.6)
                continue; // skip them

            const float *box = bbox.data<float>() + i * 4;
            int label = labels.data<float>()[i];

            cout << "Prediction " << i << ", xyxy=(";
            cout << box[0] << ", " << box[1] << ", " << box[2] << ", " << box[3]
                 << "); score=" << score << "; label=" << label << endl;

            auto rect_box = Rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);

            const float* mask = mask_probs.data<float>() +
                                i * mask_probs.size_from_dim(1) + label * mask_probs.size_from_dim(2);

            // save the 28x28 mask
            Mat mat_mask(28, 28, CV_32F);
            memcpy(mat_mask.data, mask, 28 * 28 * sizeof(float));
            mat_mask = mat_mask * 255.;

            // draw box, mask and label text on the origin image
            drawBox(mat_input, label, score, rect_box, mat_mask);

            // release the 28x28 mask object
            mat_mask.release();
        }

        imwrite(to_string(i) + ".jpg", mat_input);

        // imshow("image", mat_input);
        // waitKey(100);

        // release the w:800 h:800 image matrix object
        mat_input.release();

        i++;
    }

    return 0;
}


/// load and fill the path of files to vector
/// \param path folder path
/// \param filenames vector of filenames
void getFileNames(string& path,vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=nullptr) {
        if (strstr(ptr->d_name, ".jpg") != nullptr)
        {
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
}

/// draw box, mask and label text on the origin image
/// \param frame origin image
/// \param classId annotation classes id
/// \param conf confidence
/// \param box bbox
/// \param objectMask mask object
void drawBox(Mat &frame, int classId, float conf, Rect box, Mat &objectMask)
{
    // get color by different classId
    auto color1 = colors[classId];

    //draw a rectangle to display the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color1, 1);

    //generate a label text for the class name and its confidence
    string label = to_string(conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        // get class name, fissure or water
        label = classes[classId] + ":" + label;
    }

    //display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
//    rectangle(frame, Point(box.x, box.y - round(1.5 * labelSize.height)), Point(box.x + round(1.5 * labelSize.width), box.y + baseLine), color1, FILLED);
    // draw a filled rectangle around the text
    rectangle(frame, Point(box.x, box.y - round(labelSize.height)), Point(box.x + round(labelSize.width), box.y + baseLine), color1, FILLED);
    // draw annotation text
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 255, 255), 1);

    cv::Size dSize = cv::Size(box.width, box.height);

    // resize mask matrix from w:28 h:28 to w:800 h:800
    Mat mat_temp(dSize, CV_32F);
    resize(objectMask, mat_temp, dSize, INTER_LINEAR);
    Mat mat_coloredRoi = (0.3 * color1 + 0.7 * frame(box));
    mat_coloredRoi.convertTo(mat_coloredRoi, CV_8U);

    // draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mat_temp.convertTo(mat_temp, CV_8U);
    findContours(mat_temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(mat_coloredRoi, contours, -1, color1, 2, LINE_8, hierarchy, 100);
    mat_coloredRoi.copyTo(frame(box), mat_temp);

    mat_temp.release();
    mat_coloredRoi.release();
}
