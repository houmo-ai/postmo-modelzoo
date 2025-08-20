#include "hpp/utils.hpp"
#include "infer/infer.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

typedef struct Detection {
    int32_t x1;
    int32_t y1;
    int32_t x2;
    int32_t y2;
    float score;
    int32_t cls_idx;
} detection_t;

float bbox_overlap(const Detection &vi, const Detection &vo) {
    int xx1 = std::max(vi.x1, vo.x1);
    int yy1 = std::max(vi.y1, vo.y1);
    int xx2 = std::min(vi.x2, vo.x2);
    int yy2 = std::min(vi.y2, vo.y2);
    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);
    int area = w * h;
    float dist = float(area) / float((vi.x2 - vi.x1) * (vi.y2 - vi.y1) +
                                     (vo.y2 - vo.y1) * (vo.x2 - vo.x1) - area);
    return dist;
}

int non_max_suppression(std::vector<Detection> &detections, const float iou_threshold) {
    // sort
    std::sort(detections.begin(), detections.end(),
              [](const Detection &d1, const Detection &d2) {
                  return d1.score > d2.score;
              });

    // nms
    std::vector<Detection> keep_detections;
    std::vector<bool> suppressed = std::vector<bool>(detections.size(), false);
    const int num_detections = detections.size();
    for (int i = 0; i < num_detections; ++i) {
        if (suppressed[i])
            continue;
        keep_detections.emplace_back(detections[i]);
        for (int j = i + 1; j < num_detections; ++j) {
            if (suppressed[j])
                continue;
            float iou = bbox_overlap(detections[i], detections[j]);
            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }
    keep_detections.swap(detections);
    return 0;
}

int postprocess(const std::vector<cv::Mat> &imgs, const std::vector<tcim::Tensor> &outputs,
                std::vector<std::vector<detection_t>> &detections) {
    const int32_t MODEL_INPUT_H = 640;
    const int32_t MODEL_INPUT_W = 640;
    const float SCORE_THRESH = 0.25f;
    const float NMS_THRESH = 0.45f;
    const float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62,
                               45, 59, 119, 116, 90, 156, 198, 373, 326};

    const int32_t batch = outputs[0].Info().Shape()[0];  // 获取batch
    assert(batch == imgs.size());

    detections.clear();
    std::vector<detection_t> single_detections;
    for (int32_t db = 0; db < batch; ++db) {
        const int32_t IMG_H = imgs[db].rows;
        const int32_t IMG_W = imgs[db].cols;
        float gain = std::min(MODEL_INPUT_H / float(IMG_H), MODEL_INPUT_W / float(IMG_W));
        float pad_h = (MODEL_INPUT_H - IMG_H * gain) * 0.5f;
        float pad_w = (MODEL_INPUT_W - IMG_W * gain) * 0.5f;
        single_detections.clear();
        for (int32_t i = 0; i < outputs.size(); ++i) {
            const tcim::Tensor &output = outputs[i];  // 1, 3, ny, nx, (nc + 5)
            auto &shape = output.Info().Shape();
            const int32_t nc = shape[4] - 5;
            const int32_t H = shape[2];
            const int32_t W = shape[3];
            const int32_t na = shape[1];
            const float stride_h = float(MODEL_INPUT_H) / H;
            const float stride_w = float(MODEL_INPUT_W) / W;
            float *data = (float *)output.Data();
            for (int32_t dh = 0; dh < H; ++dh) {
                for (int32_t dw = 0; dw < W; ++dw) {
                    for (int32_t dc = 0; dc < na; ++dc) {
                        int32_t step = db * na * H * W * (nc + 5) + dc * H * W * (nc + 5) + dh * W * (nc + 5) + dw * (nc + 5);
                        float obj_conf = data[step + 4];
                        if (obj_conf < SCORE_THRESH)
                            continue;
                        // 找最大得分
                        float *p = data + step + 5;
                        auto it = std::max_element(p, p + nc);
                        int32_t max_idx = std::distance(p, it);
                        float conf = *it;
                        float score = obj_conf * conf;
                        if (score < SCORE_THRESH)
                            continue;
                        // 计算框坐标
                        float cx = (data[step + 0] * 2.0f + dw - 0.5f) * stride_w;
                        float cy = (data[step + 1] * 2.0f + dh - 0.5f) * stride_h;
                        float w = powf(data[step + 2] * 2.0f, 2.0f) * anchors[i * na * 2 + dc * 2 + 0];
                        float h = powf(data[step + 3] * 2.0f, 2.0f) * anchors[i * na * 2 + dc * 2 + 1];
                        // 映射回原图
                        int x1 = int((cx - w * 0.5f - pad_w) / gain);
                        int y1 = int((cy - h * 0.5f - pad_h) / gain);
                        int x2 = int((cx + w * 0.5f - pad_w) / gain);
                        int y2 = int((cy + h * 0.5f - pad_h) / gain);
                        // 截断
                        detection_t detection;
                        detection.x1 = x1 < 0 ? 0 : x1;
                        detection.y1 = y1 < 0 ? 0 : y1;
                        detection.x2 = x2 >= IMG_W ? IMG_W - 1 : x2;
                        detection.y2 = y2 >= IMG_H ? IMG_H - 1 : y2;
                        detection.score = score;
                        detection.cls_idx = max_idx;
                        single_detections.emplace_back(detection);
                    }
                }
            }
        }
        // nms
        non_max_suppression(single_detections, NMS_THRESH);
        printf("detection size: %d\n", single_detections.size());
        detections.emplace_back(single_detections);
    }
    return 0;
}

int main() {
    const std::string model_path = "/data/repo/imodelzoo/models/detection/yolov5s/output/xh1/yolov5s_clip_xh1_b1_1roi_1core_O2_dynamic_v2.hmm";
    const std::string img_path = "/data/repo/imodelzoo/data/datasets/coco2017/val2017/000000000139.jpg";
    const std::string input_name = "images";
    // 读图
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        printf("Failed to read image: %s\n", img_path.c_str());
        return -1;
    }
    printf("Image Size: %d x %d\n", img.cols, img.rows);
    // 获取运行时版本信息
    printf("Runtime Version: %s\n", tcim::GetVersion().c_str());
    // 获取device列表
    int64_t deviceNum = tcim::GetDeviceNum();
    printf("Device Num: %ld\n", deviceNum);
    // 加载模型
    ModuleEx module = ModuleEx::LoadFromFile(model_path);
    if (!module) {
        printf("load model failed\n");
        return -1;
    }
    // 获取模型输入输出信息
    printf("Backend Name: %s\n", module.GetBackendName().c_str());
    printf("Model Version: %s\n", module.GetModelVersion().c_str());
    printf("Model CoreNum: %d\n", module.GetCoreNum());
    printf("Model InputNum: %d\n", module.GetInputNum());
    printf("Model OutputNum: %d\n", module.GetInputNum());
    for (int i = 0; i < module.GetInputNum(); ++i) {
        std::string name = module.GetInputName(i);
        auto info = module.GetInputInfo(name);
        printf("Input[%d] name: %s, shape: [%s], dtype: %s, fmt: %s, memSize: %d\n", i,
               name.c_str(), ShapeToString(info.Shape()).c_str(), DataTypeToString(info.DataType()).c_str(),
               FormatToString(info.Format()).c_str(), info.MemSize());
    }
    std::vector<tcim::Tensor> outputs;
    for (int i = 0; i < module.GetOutputNum(); ++i) {
        std::string name = module.GetOutputName(i);
        auto info = module.GetOutputInfo(name);
        printf("Output[%d] name: %s, shape: [%s], dtype: %s, fmt: %s, memSize: %d\n", i,
               name.c_str(), ShapeToString(info.Shape()).c_str(), DataTypeToString(info.DataType()).c_str(),
               FormatToString(info.Format()).c_str(), info.MemSize());
        outputs.emplace_back(tcim::Tensor::CreateHostTensor(info.AsType(tcim::DataType::FLOAT32)));
    }
    // 将cv::Mat描述为ImageTensor
    tcim::TensorInfo bgrInfo = tcim::TensorInfo::CreateNDInfo({1, img.rows, img.cols, 3}, tcim::DataType::UINT8);
    ImageTensor bgrTensor = ImageTensor::CreateHostTensor(bgrInfo, bgrInfo.MemSize(), img.data);
    bgrTensor.SetPixelFormat(pixelFormat_t::BGR_PACKED);

    // Run
    tcim::Status status = module.Run(bgrTensor);
    if (status != tcim::Status::OK) {
        return -1;
    }
    module.Sync();

    // 获取推理结果，并反量化存入预先分配的outputs中
    for (int i = 0; i < module.GetOutputNum(); ++i) {
        std::string name = module.GetOutputName(i);
        module.GetOutput(name).CastTo(outputs[i]);
    }

    // 后处理
    std::vector<std::vector<detection_t>> detections;
    postprocess({img}, outputs, detections);

    // 画图保存
    for (auto &det : detections[0]) {  // 获取单张图片的检测结果
        cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 0, 255));
    }
    cv::imwrite("result.png", img);
    return 0;
}