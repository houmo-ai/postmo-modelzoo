#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

struct Box {
    int x1{0};
    int y1{0};
    int x2{0};
    int y2{0};

    Box() = default;

    Box(int _x1, int _y1, int _x2, int _y2) {
        x1 = _x1;
        y1 = _y1;
        x2 = _x2;
        y2 = _y2;
    }

    int w() const { return x2 - x1 + 1; }
    int h() const { return y2 - y1 + 1; }
    int x() const { return x1; }
    int y() const { return y1; }
    int cx() const { return (x1 + x2) / 2; }
    int cy() const { return (y1 + y2) / 2; }
};

typedef struct {
    float conf{0.0f};
    int cls{-1};       // cls index
    std::string name;  // cls_name
    Box box;
    float mask[32]{};
} DetectResult;

typedef struct {
    float *data{nullptr};
    int num_anchors{0};
    int stride{0};
} DetectOutput;

inline float bbox_overlap(const Box &vi, const Box &vo) {
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

inline int non_max_suppression(std::vector<DetectResult> &detections, const float iou_threshold) {
    // sort
    std::sort(detections.begin(), detections.end(),
              [](const DetectResult &d1, const DetectResult &d2) { return d1.conf > d2.conf; });

    // nms
    std::vector<DetectResult> keep_detections;
    bool *suppressed = new bool[detections.size()];
    memset(suppressed, 0, sizeof(bool) * detections.size());
    const int num_detections = detections.size();
    for (int i = 0; i < num_detections; ++i) {
        if (suppressed[i])
            continue;
        keep_detections.emplace_back(detections[i]);
        for (int j = i + 1; j < num_detections; ++j) {
            if (suppressed[j])
                continue;
            float iou = bbox_overlap(detections[i].box, detections[j].box);
            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }
    keep_detections.swap(detections);
    delete[] suppressed;

    return 0;
}

inline cv::Mat letterbox(cv::Mat &img, int height, int width) {
    float scale;
    int resize_rows;
    int resize_cols;
    if ((height * 1.0 / img.rows) < (width * 1.0 / img.cols)) {
        scale = height * 1.0 / img.rows;
    } else {
        scale = width * 1.0 / img.cols;
    }
    resize_cols = int(scale * img.cols);
    resize_rows = int(scale * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(width, height, CV_8UC3, cv::Scalar(0, 0, 0));
    int top = (height - resize_rows) / 2;
    int bot = (height - resize_rows + 1) / 2;
    int left = (width - resize_cols) / 2;
    int right = (width - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return img_new;
}

class YoloV5 {
public:
    std::vector<DetectResult> postprocess(const cv::Mat &image, std::vector<DetectOutput> outputs) {
        std::vector<DetectResult> detections;
        static float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

        float scale = (float)input_sizes_[0] / std::max(image.rows, image.cols);
        float pad_h = (input_sizes_[0] - image.rows * scale) * 0.5f;
        float pad_w = (input_sizes_[1] - image.cols * scale) * 0.5f;

        int anchor_num = 3;
        int cls_num = 80;
        int anchor_group;
        for (auto &output : outputs) {
            int stride = output.stride;
            float *feat = output.data;
            int feat_w = input_sizes_[1] / stride;
            int feat_h = input_sizes_[0] / stride;
            if (stride == 8)
                anchor_group = 1;
            else if (stride == 16)
                anchor_group = 2;
            else if (stride == 32)
                anchor_group = 3;
            else {
                printf("[error] wrong stride: %d\n", stride);
                return detections;
            }
            for (int h = 0; h <= feat_h - 1; h++) {
                for (int w = 0; w <= feat_w - 1; w++) {
                    for (int a = 0; a <= anchor_num - 1; a++) {
                        //process cls score
                        int class_index = 0;
                        float class_score = -1.0;
                        for (int s = 0; s <= cls_num - 1; s++) {
                            float score = feat[a * feat_w * feat_h * (cls_num + 5) + h * feat_w * (cls_num + 5) + w * (cls_num + 5) + s + 5];
                            if (score < conf_threshold_)
                                continue;
                            if (score > class_score) {
                                class_index = s;
                                class_score = score;
                            }
                        }
                        //process box score
                        float box_score = feat[a * feat_w * feat_h * (cls_num + 5) + (h * feat_w) * (cls_num + 5) + w * (cls_num + 5) + 4];
                        float final_score = box_score * class_score;
                        if (final_score >= conf_threshold_) {
                            int loc_idx = a * feat_h * feat_w * (cls_num + 5) + h * feat_w * (cls_num + 5) + w * (cls_num + 5);
                            float dx = feat[loc_idx + 0];
                            float dy = feat[loc_idx + 1];
                            float dw = feat[loc_idx + 2];
                            float dh = feat[loc_idx + 3];
                            float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                            float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                            float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                            float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                            float pred_w = dw * dw * 4.0f * anchor_w;
                            float pred_h = dh * dh * 4.0f * anchor_h;
                            // scale_coords
                            int x1 = (int)((pred_cx - pred_w * 0.5f - pad_w) / scale);
                            int y1 = (int)((pred_cy - pred_h * 0.5f - pad_h) / scale);
                            int x2 = (int)((pred_cx + pred_w * 0.5f - pad_w) / scale);
                            int y2 = (int)((pred_cy + pred_h * 0.5f - pad_h) / scale);

                            // clip
                            x1 = x1 < 0 ? 0 : x1;
                            y1 = y1 < 0 ? 0 : y1;
                            x2 = x2 >= image.cols ? image.cols - 1 : x2;
                            y2 = y2 >= image.rows ? image.rows - 1 : y2;

                            DetectResult detection;
                            detection.box.x1 = x1;
                            detection.box.y1 = y1;
                            detection.box.x2 = x2;
                            detection.box.y2 = y2;
                            detection.cls = class_index;
                            detection.conf = final_score;
                            detections.emplace_back(detection);
                        }
                    }
                }
            }
        }

        if (!detections.empty()) {
            non_max_suppression(detections, iou_threshold_);
        }

        return detections;
    }

private:
    int min_wh_{0};
    int max_wh_{7680};
    float iou_threshold_{0.45f};
    float conf_threshold_{0.25f};
    const int input_sizes_[2] = {640, 640};  // wh
    const int num_anchors_{25200};
    const int num_classes_{80};
};