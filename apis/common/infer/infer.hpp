#pragma once
#include "nlohmann/json.hpp"
#include "preprocess.hpp"
#include "tcim/tcim_runtime.h"
#include <cassert>
#include <map>
#include <mutex>

using json = nlohmann::json;

static std::string FormatToString(tcim::DataFmt format) {
    switch (format) {
    case tcim::YUV420SP:
        return "YUV420SP";
    case tcim::YUV422SP:
        return "YUV422SP";
    case tcim::YUV444SP:
        return "YUV444SP";
    case tcim::ND:
        return "ND";
    default:
        return "Unknown";
    }
}

static std::string ShapeToString(const std::vector<int64_t> &shape) {
    std::string shape_str = std::to_string(shape[0]);
    for (int j = 1; j < shape.size(); ++j) {
        shape_str += ", " + std::to_string(shape[j]);
    }
    return shape_str;
}

static std::string DataTypeToString(tcim::DataType dtype) {
    switch (dtype) {
    case tcim::INT8:
        return "INT8";
    case tcim::UINT8:
        return "UINT8";
    case tcim::INT16:
        return "INT16";
    case tcim::UINT16:
        return "UINT16";
    case tcim::INT32:
        return "INT32";
    case tcim::UINT32:
        return "UINT32";
    case tcim::FLOAT16:
        return "FLOAT16";
    case tcim::FLOAT32:
        return "FLOAT32";
    default:
        return "UNKNOWN";
    }
}

// 颜色空间
typedef enum PixelFormat {
    UNKNOWN = -1,
    YUV420SP = 0,  // NV12
    YUV422SP,
    YUV444SP,
    RGB_PACKED,
    BGR_PACKED,
    // RGB_PLANAR,
    // BGR_PLANAR,
} pixelFormat_t;

// 外部预处理回调函数
// typedef void (*CallbackFunc)(tcim::Tensor &input, int32_t modelInputHeight, int32_t modelInputWidth, );

/**
 * @brief 封装Tensor类
 */
class ImageTensor : public tcim::Tensor {
public:
    // 继承默认的 special members
    ImageTensor() = default;
    ~ImageTensor() = default;

    ImageTensor(const ImageTensor &other) = default;
    ImageTensor(ImageTensor &&other) = default;
    ImageTensor &operator=(const ImageTensor &other) = default;
    ImageTensor &operator=(ImageTensor &&other) = default;

    // 可以用基类 Tensor 构造/包装（方便将已有 Tensor 包装为 ImageTensor）
    explicit ImageTensor(const tcim::Tensor &base) : tcim::Tensor(base) {}
    explicit ImageTensor(tcim::Tensor &&base) noexcept : tcim::Tensor(std::move(base)) {}

    // 提供与基类相同的构造签名（直接转发到基类）
    ImageTensor(const tcim::TensorInfo &info, const tcim::Buffer &buffer)
        : tcim::Tensor(info, buffer) {}

    // 与基类静态工厂签名一致，返回 ImageTensor（内部调用基类并包装返回值）
    static ImageTensor CreateHostTensor(const tcim::TensorInfo &info, size_t mem_size = 0, void *ptr = nullptr) {
        tcim::Tensor base = tcim::Tensor::CreateHostTensor(info, mem_size, ptr);
        return ImageTensor(std::move(base));
    }

    static ImageTensor CreateDeviceTensor(const tcim::TensorInfo &info, size_t mem_size = 0, int device_id = 0,
                                          const std::string &backend_name = "") {
        tcim::Tensor base = tcim::Tensor::CreateDeviceTensor(info, mem_size, device_id, backend_name);
        return ImageTensor(std::move(base));
    }

    void SetPixelFormat(pixelFormat_t format) {
        pixel_format_ = format;
    }

    pixelFormat_t GetPixelFormat() const {
        return pixel_format_;
    }
    // 其余所有接口均来自基类 Tensor。
private:
    pixelFormat_t pixel_format_ = UNKNOWN;
};

class ModuleEx : public tcim::Module {
public:
    // 保留原module的Run函数
    using tcim::Module::Run;
    ModuleEx() : tcim::Module() {}
    explicit ModuleEx(const tcim::Module &module) : tcim::Module(module) {}
    ModuleEx(ModuleEx &&module) : tcim::Module(std::move(module)) {}
    // 带文件名和配置的构造函数
    ModuleEx(const std::string &filename, const Option &option = Option())
        : tcim::Module(filename, option) {}  // 调用基类带参构造[7](@ref)
    static ModuleEx LoadFromFile(const std::string &filename, const Option &option = Option()) {
        tcim::Module module = tcim::Module::LoadFromFile(filename, option);
        return ModuleEx(module);
    }
    static ModuleEx LoadFromMem(const void *data, int len, const Option &option = Option()) {
        tcim::Module module = tcim::Module::LoadFromMem(data, len, option);
        return ModuleEx(module);
    }
    /**
     * @brief ModuleEx的推理接口，reszier相关处理由该函数处理，用户只需要传入原始图片
     *        目前仅适合单输入模型，线程安全，但是不建议多线程共用Module，效率比较差
     * @param image  YUV420/422/444图像
     * @param sync  是否使用同步推理
     * @return tcim::Status
     */
    tcim::Status Run(ImageTensor &image, bool sync = false, const RunOption &option = RunOption()) {
        std::lock_guard<std::mutex> lock(rw_mtx_);
        tcim::Status status;
        // 形状检查
        if (image.Info().Shape().size() != 4) {
            printf("[ModuleEx] Input shape not match, except shape size == 4\n");
            return tcim::Status::INVALID_ARGUMENT;
        }
        // 像素格式检查
        switch (image.GetPixelFormat()) {
        case YUV420SP:
        case YUV422SP:
        case YUV444SP:
        case RGB_PACKED:
        case BGR_PACKED:
            break;
        default:
            printf("[ModuleEx] Input image pixel format not support\n");
            return tcim::Status::INVALID_ARGUMENT;
        }
        // 获取CustomMsg
        std::string customMsgStr = GetCustomMsg();
        json customMsg = json::parse(customMsgStr);
        std::string name = GetInputName(0);
        // 0. no resizer
        // 1. dynamic verison2  [y, x, crop_h, crop_w, resize_h, resize_w, top, left, bottom, right]
        // 2. dynamic verison1  [y, x, crop_h, crop_w]
        // 3. static
        const int32_t resizerMode = customMsg[name]["resizer_mode"];
        const auto &modelInputShape = customMsg[name]["shape"];
        const auto &modelInputInfo = customMsg[name]["input_cfg"];
        // 没有resizer的情况下暂不支持，后续根据透传的预处理信息，由该Module处理
        if (resizerMode == 0) {
            printf("[ModuleEx] multi input not spoort yet\n");
            return tcim::Status::UNSUPPORTED;
        }
        tcim::TensorInfo imgInputInfo = GetInputInfo(name);
        auto &shape = imgInputInfo.Shape();
        const int32_t RESIZER_INPUT_H = shape[2];   // resizer输入高
        const int32_t RESIZER_INPUT_W = shape[3];   // resizer输入宽
        int32_t IMAGE_H = image.Info().Shape()[2];  // 输入图像高
        int32_t IMAGE_W = image.Info().Shape()[3];  // 输入图像宽
        if (image.GetPixelFormat() == BGR_PACKED || image.GetPixelFormat() == RGB_PACKED) {
            IMAGE_H = image.Info().Shape()[1];
            IMAGE_W = image.Info().Shape()[2];
        }
        const int32_t MODEL_INPUT_H = modelInputShape[2];            // 原模型高
        const int32_t MODEL_INPUT_W = modelInputShape[3];            // 原模型宽
        const int32_t paddingMode = modelInputInfo["padding_mode"];  // 填充模式
        printf("RESIZER_INPUT_H: %d, RESIZER_INPUT_W: %d\n", RESIZER_INPUT_H, RESIZER_INPUT_W);
        printf("MODEL_INPUT_H: %d, MODEL_INPUT_W: %d\n", MODEL_INPUT_H, MODEL_INPUT_W);
        printf("IMAGE_H: %d, IMAGE_W: %d\n", IMAGE_H, IMAGE_W);
        // 检查输入图像是否超过resizer的输入限制
        // TODO 如果超过限制，后续由此module缩放处理
        if (IMAGE_H > RESIZER_INPUT_H || IMAGE_W > RESIZER_INPUT_W) {
            printf("[ModuleEx] Input HW is too large, except shape: [%d, %d]\n", RESIZER_INPUT_H, RESIZER_INPUT_W);
            return tcim::Status::INVALID_ARGUMENT;
        }
        if (MODEL_INPUT_H % 2 != 0) {
            printf("[ModuleEx] Input H must be even number\n");
            return tcim::Status::INVALID_ARGUMENT;
        }
        if (MODEL_INPUT_W % 2 != 0) {
            printf("[ModuleEx] Input W must be even number\n");
            return tcim::Status::INVALID_ARGUMENT;
        }
        // 计算dynamic resizer参数
        if (resizerMode == 1 || resizerMode == 2) {
            std::string resizerInputName = "resizer_crop_" + name;
            auto dyninfo = GetInputInfo(resizerInputName);
            // 输入图像必须保证偶数
            IMAGE_H &= ~1;
            IMAGE_W &= ~1;
            dynamicInfo_[0] = 0;
            dynamicInfo_[1] = 0;
            dynamicInfo_[2] = IMAGE_H;  // 必须是偶数
            dynamicInfo_[3] = IMAGE_W;
            int32_t resize_H = MODEL_INPUT_H;
            int32_t resize_W = MODEL_INPUT_W;
            if (resizerMode == 1) {
                float scale = std::min(MODEL_INPUT_H * 1.0f / IMAGE_H, MODEL_INPUT_W * 1.0f / IMAGE_W);
                resize_H = std::round(IMAGE_H * scale);
                resize_W = std::round(IMAGE_W * scale);
                resize_H &= ~1;
                resize_W &= ~1;
                int32_t top = paddingMode == 0 ? 0 : ((MODEL_INPUT_H - resize_H) / 2) & ~1;
                int32_t left = paddingMode == 0 ? 0 : ((MODEL_INPUT_W - resize_W) / 2) & ~1;
                int32_t bottom = MODEL_INPUT_H - resize_H - top;
                int32_t right = MODEL_INPUT_W - resize_W - left;
                // 计算等比例缩放后宽高
                dynamicInfo_[4] = resize_H;
                dynamicInfo_[5] = resize_W;
                dynamicInfo_[6] = top;
                dynamicInfo_[7] = left;
                dynamicInfo_[8] = bottom;
                dynamicInfo_[9] = right;
            }
            float sw = static_cast<float>(resize_W) / IMAGE_W;
            float sh = static_cast<float>(resize_H) / IMAGE_H;
            if (sw < 1.0f / 32 || sw > 16) {
                printf("[ModuleEx] W scale ratio is too large, except range: [1/32, 16]\n");
                return tcim::Status::INVALID_ARGUMENT;
            }
            if (sh < 1.0f / 32 || sh > 16) {
                printf("[ModuleEx] H scale ratio is too large, except range: [1/32, 16]\n");
                return tcim::Status::INVALID_ARGUMENT;
            }
            dynamicInfoTensor_ = tcim::Tensor::CreateHostTensor(dyninfo, dyninfo.MemSize(), dynamicInfo_);
            status = SetInput(resizerInputName, dynamicInfoTensor_);
            if (status != tcim::Status::OK) {
                printf("[ModuleEx] DynamicInfo SetInput failed.");
                return status;
            }
        }
        // 根据内存类型按需分配内存
        if (imgInputTensor_.GetInitStatus() != tcim::Status::OK) {
            imgInputTensor_ = tcim::Tensor::CreateDeviceTensor(imgInputInfo, imgInputInfo.MemSize());
        }

        auto fmt = imgInputInfo.Format();  // 获取ressizer期望的数据格式
        // 如果外部传入的图像格式和resizer期望的不一致，需要内部转换则申请yuvTensor，目前只支持RGB_PACKED和BGR_PACKED
        if (image.Info().Format() != fmt) {
            if (yuvHostBuffer_.GetInitStatus() != tcim::Status::OK) {
                yuvHostBuffer_ = tcim::Buffer::CreateHostBuffer(imgInputInfo.MemSize());
            }
            // 外部传入的tensor可能是device内存，需要拷贝到host
            if (image.Device() == tcim::Device::HDPL) {
                if (bgrHostBuffer_.GetInitStatus() != tcim::Status::OK) {
                    bgrHostBuffer_ = tcim::Buffer::CreateHostBuffer(imgInputInfo.MemSize());
                }
                bgrHostTensor_ = ImageTensor::CreateHostTensor(image.Info(), image.Info().MemSize(), bgrHostBuffer_.Data());
                bgrHostTensor_.SetPixelFormat(pixelFormat_t::BGR_PACKED);
                image.CopyTo(bgrHostTensor_);
            } else {
                bgrHostTensor_ = image;  // 如果是host直接贴过去
            }
            // 如果是RGB/BGR格式的图片就转为YUV420SP/YUV422SP/YUV444P
            tcim::TensorInfo yuvInfo;
            if (image.GetPixelFormat() == pixelFormat_t::BGR_PACKED) {
                yuvInfo = tcim::TensorInfo::CreateYUVInfo(1, IMAGE_W, IMAGE_H, fmt);
                BGR2YUVSP((uint8_t *)(bgrHostTensor_.Data()), (uint8_t *)(yuvHostBuffer_.Data()), IMAGE_H, IMAGE_W, fmt);
            } else {
                printf("[ModuleEx] Input format not match, except fmt: %s\n", FormatToString(fmt).c_str());
                return tcim::Status::INVALID_ARGUMENT;
            }
            yuvHostTensor_ = tcim::Tensor::CreateHostTensor(yuvInfo, yuvInfo.MemSize(), yuvHostBuffer_.Data());
        } else {
            yuvHostTensor_ = image;  // 格式相同直接贴过去
        }

        // 图像数据传入模型输入Tensor
        tcim::Tensor y_plane, uv_plane;
        status = yuvHostTensor_.SplitYUV(y_plane, uv_plane);
        if (status != tcim::Status::OK) {
            printf("[ModuleEx] SplitYUV failed.\n");
            return status;
        }
        tcim::Tensor imgInputTensor_y, imgInputTensor_uv;
        status = imgInputTensor_.SplitYUV(imgInputTensor_y, imgInputTensor_uv);
        if (status != tcim::Status::OK) {
            printf("[ModuleEx] SplitYUV failed.\n");
            return status;
        }
        auto imageTensorYRoi = imgInputTensor_y.SelectROI({0, 0, 0}, y_plane.Info().Shape());
        auto imageTensorUVRoi = imgInputTensor_uv.SelectROI({0, 0, 0, 0}, uv_plane.Info().Shape());
        status = y_plane.CopyTo(imageTensorYRoi);
        if (status != tcim::Status::OK) {
            printf("[ModuleEx] Y Plane CopyTo failed.");
            return status;
        }
        status = uv_plane.CopyTo(imageTensorUVRoi);
        if (status != tcim::Status::OK) {
            printf("[ModuleEx] UV Plane CopyTo failed.");
            return status;
        }
        status = SetInput(name, imgInputTensor_);
        if (status != tcim::Status::OK) {
            printf("[ModuleEx] SetInput failed.");
            return status;
        }
        return Run(sync, option);
    }

private:
    std::mutex rw_mtx_;
    tcim::Buffer bgrHostBuffer_;  // 缓存外部传入BGR图像是device同时需要转YUV的情况缓存数据
    ImageTensor bgrHostTensor_;
    tcim::Buffer yuvHostBuffer_;  // 缓存外部传入BGR图像转YUV后的数据
    tcim::Tensor yuvHostTensor_;
    tcim::Tensor imgInputTensor_;     // 图像输入
    tcim::Tensor dynamicInfoTensor_;  // dynamic输入
    int32_t dynamicInfo_[10];
};
