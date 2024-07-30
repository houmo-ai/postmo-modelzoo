// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
#include <unistd.h>
#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <cmath>
#include <queue>

#include "threads.hpp"

#include "tcim/tcim_runtime.h"
#include "idnnl/idnnl_utils.h"

#define GET_TIME() std::chrono::system_clock::now()
#define GET_COST(start, end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()

static constexpr int kNumAnchor = 116 * 116 * 6;
static constexpr int kNumOutputBoxFeature = 7;
static constexpr int kBatchSize = 1;
static constexpr int kNumIndsForScan = 1024;
static constexpr int kNumThreads = 64;
static constexpr int kNumBoxCorners = 4;
static constexpr float kPillarXSize = 0.22f;
static constexpr float kPillarYSize = 0.22f;
static constexpr float kPillarZSize = 6.0f;
static constexpr float kMinXRange = -51.04f;
static constexpr float kMinYRange = -51.04f;
static constexpr float kMinZRange = -2.0f;
static constexpr float kMaxXRange = 51.04f;
static constexpr float kMaxYRange = 51.04f;
static constexpr float kMaxZRange = 4.0f;
static constexpr int kNumClass = 3;
static constexpr int kMaxNumPillars = 12000;
static constexpr int kMaxNumPointsPerPillar = 20;
static constexpr int kNumPointFeature = 5;  // x, y, z, i, delta of time
static constexpr int h = 464;
static constexpr int w = 464;
static constexpr float score_threshold = 0.3;
static constexpr float nms_overlap_threshold = 0.5;
static constexpr int gridXSize = (kMaxXRange - kMinXRange) / kPillarXSize;
static constexpr int gridYSize = (kMaxYRange - kMinYRange) / kPillarYSize;
static constexpr int gridZSize = (kMaxZRange - kMinZRange) / kPillarZSize;
static constexpr int rpnOutputWidth = 116;
static constexpr int rpnOutputHeight = 116;
#define ROUND_UP(A, B) ((((A) + (B) - (1)) / (B)) * B)


template <typename point_data_type>
bool readPointFile(const std::string& filename, int pointDim,
                   void* point_buffer) {
  // open the file:
  std::streampos fileSize;
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "[Error] Open file " << filename << " failed" << std::endl;
    return false;
  }
  // get its size:
  file.seekg(0, std::ios::end);
  fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  if (fileSize / sizeof(point_data_type) % pointDim != 0) {
    std::cerr << "[Error] File Size Error! " << fileSize << std::endl;
    return false;
  }
  // read the data:
  char* buffer = reinterpret_cast<char*>(point_buffer);
  file.read(buffer, fileSize);
  file.close();
  return true;
}

template <typename T>
void SaveInt8MemoryToFile(const T* memory, size_t size,
                          const std::string& filename, int line_num = 64) {
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  size_t numLines = size / line_num;
  for (size_t i = 0; i < numLines; ++i) {
    for (size_t j = 0; j < line_num; ++j) {
      file << static_cast<int>(memory[i * line_num + j]) << " ";
    }
    file << std::endl;
  }

  size_t remainingBytes = size % line_num;
  if (remainingBytes > 0) {
    for (size_t j = 0; j < remainingBytes; ++j) {
      file << static_cast<int>(memory[numLines * line_num + j]) << " ";
    }
    file << std::endl;
  }

  file.close();
}

class IdnnlDesc {
 public:
  IdnnlDesc() {
    idnnlCreateTensorDescriptor(&tdesc_);
    idnnlCreateMemoryDescriptor(&mdesc_);
  }
  ~IdnnlDesc() {
    idnnlDestroyTensorDescriptor(tdesc_);
    idnnlDestroyMemoryDescriptor(mdesc_);
    if (data_) {
      hdplFree(data_);
      data_ = nullptr;
    }
  }

  idnnlTensorDescriptor_t tdesc_;
  idnnlMemoryDescriptor_t mdesc_;
  void* data_ = nullptr;
};


class PointPillars {
 public:
  PointPillars(int point_num) {
    idnnlCreate(&handle_);
    hdplStreamCreate(&stream_);
    idnnlSetStream(handle_, stream_);

    InitVoxelization(point_num);
    InitPfe1();
    InitScatter();
    InitRpn();
    InitPillarsDetection();
  }

  ~PointPillars() {
    hdplReservedMemFree(voxelization_wsdata_);
    hdplFree(detection_wsdata_);
    hdplStreamDestroy(stream_);
    idnnlDestroy(handle_);
  }

  void InitPoints(const std::string& file, int point_num) {
    int point_dim_num = kNumPointFeature;
    // build points
    void* host_point = malloc(point_num * point_dim_num * sizeof(point_dim_num));
    readPointFile<float>(file, point_dim_num, host_point);
    printf("point[0] %f point[54908] %f \n",
          (reinterpret_cast<float*>(host_point))[0],
          reinterpret_cast<float*>(host_point)[54907 * 5]);

    const int point_shape[2] = {point_num, point_dim_num};
    idnnlSetTensorNdDescriptor(points_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              point_shape, 2);
    hdplMalloc(&points_.data_, point_num * point_dim_num * sizeof(float));
    idnnlSetMemoryDescriptor(points_.mdesc_, &points_.data_, 1, IDNNL_MEM_GM);
    hdplMemcpy(points_.data_, host_point,
               point_num * point_dim_num * sizeof(float), hdplMemcpyHostToDevice);
    free(host_point);
  }

  void InitVoxelization(int point_num) {
    std::string point_file = "../data/point_464.txt";
    InitPoints(point_file, point_num);

    // voxel
    const int voxel_shape[3] = {kMaxNumPillars, kMaxNumPointsPerPillar,
                                kNumPointFeature};
    hdplMalloc(&voxel_.data_, kMaxNumPillars * kMaxNumPointsPerPillar *
               ROUND_UP(sizeof(int8_t) * kNumPointFeature, 64));
    idnnlSetTensorNdDescriptor(voxel_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_INT8,
                               voxel_shape, 3);
    idnnlSetMemoryDescriptor(voxel_.mdesc_, &voxel_.data_, 1, IDNNL_MEM_GM);

    // coors
    const int coors_shape[] = {kMaxNumPillars, 4};
    hdplMalloc(&coors_.data_, kMaxNumPillars * ROUND_UP(4 * sizeof(int16_t), 64));
    idnnlSetTensorNdDescriptor(coors_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_INT16,
                               coors_shape, 2);
    idnnlSetMemoryDescriptor(coors_.mdesc_, &coors_.data_, 1, IDNNL_MEM_GM);

    // num_per_pillar
    const int num_per_pillar_shape[] = {kMaxNumPillars};
    hdplMalloc(&num_per_pillar_.data_, kMaxNumPillars * sizeof(int16_t));
    idnnlSetTensorNdDescriptor(num_per_pillar_.tdesc_, IDNNL_TENSOR_ND,
                               IDNNL_DATA_INT16, num_per_pillar_shape, 1);
    idnnlSetMemoryDescriptor(num_per_pillar_.mdesc_, &num_per_pillar_.data_, 1,
                             IDNNL_MEM_GM);

    // get gm workspace size
    idnnlCreateWorkspaceDescriptor(&voxelization_workspace_);
    idnnlGetVoxelizationWorkspace(handle_, h, w, kMaxNumPointsPerPillar,
                                  kMaxNumPillars, voxelization_workspace_);
    int size = 0;
    idnnlGetGMWorkspaceSize(voxelization_workspace_, &size);
    // size = ROUND_UP(size, 64);
    size = 105 * 1024 * 1024;  // todo: need fix
    hdplReservedMemAlloc(&voxelization_wsdata_, size);
    idnnlSetGMWorkspace(voxelization_workspace_, voxelization_wsdata_);
    printf("voxelization workspace h %d, w %d , kMaxNumPointsPerPillar %d, "
           "kMaxNumPillars %d, addr %p, size %d\n", h, w, kMaxNumPointsPerPillar,
           kMaxNumPillars, voxelization_wsdata_, size);
  }

  void InitPfe1() {
    pfe1_module_ = tcim::Module::LoadFromFile("../pfe_1.hmm");
    if (!pfe1_module_) {
      std::cout << "pfe1 load model fail, exit..." << std::endl;
      exit(-1);
    }
    tcim::TensorInfo input_info;
    std::string input_name = pfe1_module_.GetInputName(0);
    pfe1_module_.GetInputInfo(input_name, input_info, tcim::HDPL);
    // {1, 12032, 20, 64},
    std::cout << "Pfe1 input[" << input_name << "] " << input_info << std::endl;
    tcim::Tensor input_data(input_info, voxel_.data_, input_info.MemSize());
    pfe1_module_.SetInput("voxels", input_data);

    tcim::TensorInfo output_info;
    std::string output_name = pfe1_module_.GetOutputName(0);
    pfe1_module_.GetOutputInfo(output_name, output_info, tcim::HDPL, true);
    // {1, 12032, 1, 64},
    std::cout << "Pfe1 output[" << output_name << "] " << output_info << std::endl;
    int ret = hdplMalloc(&pfe1_out_.data_, output_info.MemSize());
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc pfe1_out return %d\n", ret);
      return;
    }
    tcim::Tensor output_data(output_info, pfe1_out_.data_, output_info.MemSize());
    pfe1_module_.SetOutput(pfe1_module_.GetOutputName(0), output_data);
    pfe1_module_.SetStream(stream_);
  }

  void InitScatter() {
    int scatter_output_size = h * w * 64;
    int ret = hdplMalloc(&scatter_out_.data_, scatter_output_size);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc scatter_out return %d\n", ret);
      return;
    }

    void* indices_mem_ptr[] = {coors_.data_};
    void* update_mem_ptr[] = {pfe1_out_.data_};
    void* scatter_output_ptr[] = {scatter_out_.data_};
    idnnlSetMemoryDescriptor(coors_.mdesc_, indices_mem_ptr, 1, IDNNL_MEM_GM);
    idnnlSetMemoryDescriptor(pfe1_out_.mdesc_, update_mem_ptr, 1, IDNNL_MEM_GM);
    idnnlSetMemoryDescriptor(scatter_out_.mdesc_, scatter_output_ptr, 1, IDNNL_MEM_GM);

    // loop_time need check
    idnnlSetTensor4dDescriptor(coors_.tdesc_, IDNNL_TENSOR_NCHW, IDNNL_DATA_INT16,
                               1, 1, 12032, 4);  // need to check
    idnnlSetTensor4dDescriptor(pfe1_out_.tdesc_, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8,
                               1, 1, 12032, 64);  // need to check
    idnnlSetTensor4dDescriptor(scatter_out_.tdesc_, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8,
                               1, h, w, 64);  // need to check

    int thread_num = 4;
    uint32_t* pillar_point_index_ddr =
        reinterpret_cast<uint32_t*>(voxelization_wsdata_) + thread_num * h * w;
    uint32_t* active_pillar_point_count_ddr =
        pillar_point_index_ddr + thread_num * h * w * kMaxNumPointsPerPillar;
    uint32_t* active_pillar_point_index_ddr =
        active_pillar_point_count_ddr + h * w * kMaxNumPointsPerPillar;
    uint32_t* x_ddr = active_pillar_point_index_ddr + h * w;
    uint32_t* active_pillar_addr = x_ddr + kMaxNumPillars;

    left_pillar_count_ = 9226;
    void* active_pillar_ptr[] = {active_pillar_addr};
    idnnlSetMemoryDescriptor(active_pillar_.mdesc_, active_pillar_ptr, 1,
                             IDNNL_MEM_GM);
  }

  void InitRpn() {
    rpn_module_ = tcim::Module::LoadFromFile("../rpn.hmm");
    if (!rpn_module_) {
      std::cout << "rpn load model fail, exit..." << std::endl;
      exit(-1);
    }

    int rpn_output_size = 116 * 116 * 64;
    int ret1 = hdplMalloc(&dir_cls_preds_.data_, rpn_output_size);
    int ret2 = hdplMalloc(&bbox_preds_.data_, rpn_output_size);
    int ret3 = hdplMalloc(&cls_scores_.data_, rpn_output_size);
    if ((ret1 || ret2 || ret3) != hdplError_t::hdplSuccess) {
      printf("hdplMalloc rpn outputs return %d, %d, %d\n", ret1, ret2, ret3);
      return;
    }

    tcim::TensorInfo input_info;
    std::string input_name = rpn_module_.GetInputName(0);
    rpn_module_.GetInputInfo(input_name, input_info, tcim::HDPL);
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
    // {1, h, w, 64}
    tcim::Tensor input_data(input_info, scatter_out_.data_, input_info.MemSize());
    rpn_module_.SetInput("features", input_data);

    int output_num = rpn_module_.GetOutputNum();
    tcim::TensorInfo dir_cls_preds_info;
    tcim::TensorInfo bbox_preds_info;
    tcim::TensorInfo cls_scores_info;
    // {1, 116, 116, 12},
    rpn_module_.GetOutputInfo("dir_cls_preds", dir_cls_preds_info, tcim::HDPL, true);
    std::cout << "Rpn output[dir_cls_preds] " << dir_cls_preds_info << std::endl;
    // {1, 116, 116, 42}, ?
    rpn_module_.GetOutputInfo("bbox_preds", bbox_preds_info, tcim::HDPL, true);
    std::cout << "Rpn output[bbox_preds] " << dir_cls_preds_info << std::endl;
    // {1, 116, 116, 18},
    rpn_module_.GetOutputInfo("cls_scores", cls_scores_info, tcim::HDPL, true);
    std::cout << "Rpn output[cls_scores] " << cls_scores_info << std::endl;
    tcim::Tensor dir_cls_preds(dir_cls_preds_info, dir_cls_preds_.data_, dir_cls_preds_info.MemSize());
    tcim::Tensor bbox_preds(bbox_preds_info, bbox_preds_.data_, bbox_preds_info.MemSize());
    tcim::Tensor cls_scores(cls_scores_info, cls_scores_.data_, cls_scores_info.MemSize());
    rpn_module_.SetOutput("dir_cls_preds", dir_cls_preds);
    rpn_module_.SetOutput("bbox_preds", bbox_preds);
    rpn_module_.SetOutput("cls_scores", cls_scores);
    rpn_module_.SetStream(stream_);
  }

  void InitPillarsDetection() {
    // 初始化anchor
    float* anchors_px = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    float* anchors_py = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    float* anchors_pz = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    float* anchors_dx = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    float* anchors_dy = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    float* anchors_dz = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    float* anchors_ro = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    float* anchors_diagonal = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
    // w*h*6 总的anchor数量
    for (int i = 0; i < kNumAnchor; ++i) {
      anchors_px[i] = 0;
      anchors_py[i] = 0;
      anchors_pz[i] = 0;
      anchors_dx[i] = 0;
      anchors_dy[i] = 0;
      anchors_dz[i] = 0;
      anchors_ro[i] = 0;
      anchors_diagonal[i] = 0;
    }

    std::vector<int> kNumAnchorSets{6};
    std::vector<int> kAnchorStrides{1};
    int kGridXSize;
    int kGridYSize;
    std::vector<int> kAnchorRanges;
    std::vector<std::vector<int>> kNumAnchorRo;
    std::vector<std::vector<float>> kAnchorRo;
    std::vector<std::vector<float>> kAnchorZCoors;
    std::vector<std::vector<float>> kAnchorDxSizes;
    std::vector<std::vector<float>> kAnchorDySizes;
    std::vector<std::vector<float>> kAnchorDzSizes;
    kGridXSize = static_cast<int>((kMaxXRange - kMinXRange) / kPillarXSize);
    kGridYSize = static_cast<int>((kMaxYRange - kMinYRange) / kPillarYSize);
    kNumAnchorSets = std::vector<int>{6};
    kAnchorRanges = std::vector<int>{0,
                                    kGridXSize,
                                    0,
                                    kGridYSize,
                                    static_cast<int>(kGridXSize * 0.25),
                                    static_cast<int>(kGridXSize * 0.75),
                                    static_cast<int>(kGridYSize * 0.25),
                                    static_cast<int>(kGridYSize * 0.75)};
    kNumAnchorRo = std::vector<std::vector<int>>{std::vector<int>{2, 2, 2}};
    kAnchorStrides = std::vector<int>{4};
    kAnchorRo = std::vector<std::vector<float>>{
      std::vector<float>{0, M_PI / 2, 0, M_PI / 2, 0, M_PI / 2}};
    kAnchorZCoors =
      std::vector<std::vector<float>>{std::vector<float>{-0.0345, -0.1188, 0}};
    kAnchorDxSizes =
      std::vector<std::vector<float>>{std::vector<float>{2.08, 0.84, 0.84}};
    kAnchorDySizes =
      std::vector<std::vector<float>>{std::vector<float>{4.73, 1.81, 0.91}};
    kAnchorDzSizes =
      std::vector<std::vector<float>>{std::vector<float>{1.77, 1.77, 1.74}};
    float dir_cls_preds_scale[12] = {
      0.056628406047821045, 0.056697290390729904, 0.05978292599320412,
      0.05864154174923897,  0.04730793088674545,  0.0475882813334465,
      0.02695307321846485,  0.024648763239383698, 0.04782559722661972,
      0.04482092708349228,  0.02724352478981018,  0.02580839768052101};
    float bbox_preds_scale[42] = {
      0.001961667789146304,  0.0020628366619348526, 0.07357551902532578,
      0.0032179260160773993, 0.0030672349967062473, 0.0050014290027320385,
      0.02508682757616043,   0.002050700131803751,  0.002026189351454377,
      0.054453786462545395,  0.0034170718863606453, 0.002612282754853368,
      0.005372609477490187,  0.06300502270460129,   0.01125330664217472,
      0.007873781956732273,  0.09599286317825317,   0.06064479425549507,
      0.03604217991232872,   0.01280643604695797,   0.0644073337316513,
      0.002326297340914607,  0.008363761007785797,  0.11336273699998856,
      0.05861886218190193,   0.019393285736441612,  0.010374098084867,
      0.11800024658441544,   0.007738177198916674,  0.0057998038828372955,
      0.0808228999376297,    0.049410074949264526,  0.013926314190030098,
      0.0058728428557515144, 0.09419936686754227,   0.002718044677749276,
      0.019094906747341156,  0.09254813194274902,   0.05099649727344513,
      0.009553642012178898,  0.008247417397797108,  0.10400533676147461};
    float cls_scores_scale[18] = {
      0.07753868401050568, 0.3241787552833557,  0.3362697660923004,
      0.07925483584403992, 0.29115620255470276, 0.35586923360824585,
      0.24571871757507324, 0.21192100644111633, 0.2042505443096161,
      0.20361800491809845, 0.20350350439548492, 0.21768109500408173,
      0.23759114742279053, 0.19255879521369934, 0.20059050619602203,
      0.2807539403438568,  0.19895122945308685, 0.2204863280057907};

    int ind = 0;
    for (size_t head = 0; head < kNumAnchorSets.size(); ++head) {
      float x_stride = kPillarXSize * kAnchorStrides[head];
      float y_stride = kPillarYSize * kAnchorStrides[head];
      int x_ind_start = kAnchorRanges[head * 4 + 0] / kAnchorStrides[head];
      int x_ind_end = kAnchorRanges[head * 4 + 1] / kAnchorStrides[head];
      int y_ind_start = kAnchorRanges[head * 4 + 2] / kAnchorStrides[head];
      int y_ind_end = kAnchorRanges[head * 4 + 3] / kAnchorStrides[head];
      // coors of first anchor's center
      float x_offset = kMinXRange + x_stride / 2.0;
      float y_offset = kMinYRange + y_stride / 2.0;
      std::vector<float> anchor_x_count, anchor_y_count;
      for (int i = x_ind_start; i < x_ind_end; ++i) {
        float anchor_coor_x = static_cast<float>(i) * x_stride + x_offset;
        anchor_x_count.push_back(anchor_coor_x);
      }
      for (int i = y_ind_start; i < y_ind_end; ++i) {
        float anchor_coor_y = static_cast<float>(i) * y_stride + y_offset;
        anchor_y_count.push_back(anchor_coor_y);
      }
      for (int y = 0; y < y_ind_end - y_ind_start; ++y) {
        for (int x = 0; x < x_ind_end - x_ind_start; ++x) {
          // AINFO<<"x="<<x<<"| y="<<y;
          int ro_count = 0;
          for (size_t c = 0; c < kNumAnchorRo[head].size(); ++c) {
            for (int i = 0; i < kNumAnchorRo[head][c]; ++i) {
              anchors_px[ind] = anchor_x_count[x];
              anchors_py[ind] = anchor_y_count[y];
              anchors_ro[ind] = kAnchorRo[head][ro_count];
              anchors_pz[ind] = kAnchorZCoors[head][c];
              anchors_dx[ind] = kAnchorDxSizes[head][c];
              anchors_dy[ind] = kAnchorDySizes[head][c];
              anchors_dz[ind] = kAnchorDzSizes[head][c];
              anchors_diagonal[ind] = sqrtf(anchors_dx[ind] * anchors_dx[ind] +
                                            anchors_dy[ind] * anchors_dy[ind]);
              ro_count++;
              ind++;
            }
          }
        }
      }
    }

    // anchors_px
    const int anchors_px_shape[1] = {kNumAnchor};
    idnnlSetTensorNdDescriptor(anchors_px_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              anchors_px_shape, 1);
    int ret = hdplMalloc(&anchors_px_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_px_ return %d\n", ret);
      return;
    }
    idnnlSetMemoryDescriptor(anchors_px_.mdesc_, &anchors_px_.data_, 1, IDNNL_MEM_GM);
    hdplMemcpy(anchors_px_.data_, anchors_px, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // anchors_py
    idnnlSetTensorNdDescriptor(anchors_py_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              anchors_px_shape, 1);
    ret = hdplMalloc(&anchors_py_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_py_ return %d\n", ret);
      return;
    }
    void* anchors_py_ptr[] = {anchors_py_.data_};
    idnnlSetMemoryDescriptor(anchors_py_.mdesc_, anchors_py_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(anchors_py_.data_, anchors_py, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // anchors_pz
    const int anchors_pz_shape[1] = {kNumAnchor};
    idnnlSetTensorNdDescriptor(anchors_pz_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              anchors_pz_shape, 1);
    ret = hdplMalloc(&anchors_pz_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_pz_ return %d\n", ret);
      return;
    }
    void* anchors_pz_ptr[] = {anchors_pz_.data_};
    idnnlSetMemoryDescriptor(anchors_pz_.mdesc_, anchors_pz_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(anchors_pz_.data_, anchors_pz, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // anchors_dx
    const int anchors_dx_shape[1] = {kNumAnchor};
    idnnlSetTensorNdDescriptor(anchors_dx_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              anchors_dx_shape, 1);
    ret = hdplMalloc(&anchors_dx_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_dx_ return %d\n", ret);
      return;
    }
    void* anchors_dx_ptr[] = {anchors_dx_.data_};
    idnnlSetMemoryDescriptor(anchors_dx_.mdesc_, anchors_dx_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(anchors_dx_.data_, anchors_dx, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // anchors_dy
    const int anchors_dy_shape[1] = {kNumAnchor};
    idnnlSetTensorNdDescriptor(anchors_dy_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              anchors_dy_shape, 1);
    ret = hdplMalloc(&anchors_dy_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_dy_ return %d\n", ret);
      return;
    }
    void* anchors_dy_ptr[] = {anchors_dy_.data_};
    idnnlSetMemoryDescriptor(anchors_dy_.mdesc_, anchors_dy_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(anchors_dy_.data_, anchors_dy, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // anchors_dz
    const int anchors_dz_shape[1] = {kNumAnchor};
    idnnlSetTensorNdDescriptor(anchors_dz_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              anchors_dz_shape, 1);
    ret = hdplMalloc(&anchors_dz_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_dz_ return %d\n", ret);
      return;
    }
    void* anchors_dz_ptr[] = {anchors_dz_.data_};
    idnnlSetMemoryDescriptor(anchors_dz_.mdesc_, anchors_dz_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(anchors_dz_.data_, anchors_dz, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // anchors_ro
    const int anchors_ro_shape[1] = {kNumAnchor};
    idnnlSetTensorNdDescriptor(anchors_ro_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              anchors_ro_shape, 1);
    ret = hdplMalloc(&anchors_ro_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_ro_ return %d\n", ret);
      return;
    }
    void* anchors_ro_ptr[] = {anchors_ro_.data_};
    idnnlSetMemoryDescriptor(anchors_ro_.mdesc_, anchors_ro_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(anchors_ro_.data_, anchors_ro, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // anchors_diagonal
    const int anchors_diagonal_shape[1] = {kNumAnchor};
    idnnlSetTensorNdDescriptor(anchors_diagonal_.tdesc_, IDNNL_TENSOR_ND,
                              IDNNL_DATA_FLOAT, anchors_diagonal_shape, 1);
    ret = hdplMalloc(&anchors_diagonal_.data_, kNumAnchor * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc anchors_diagonal_ return %d\n", ret);
      return;
    }
    void* anchors_diagonal_ptr[] = {anchors_diagonal_.data_};
    idnnlSetMemoryDescriptor(anchors_diagonal_.mdesc_, anchors_diagonal_ptr, 1,
                            IDNNL_MEM_GM);
    hdplMemcpy(anchors_diagonal_.data_, anchors_diagonal, kNumAnchor * 4,
              hdplMemcpyHostToDevice);

    // preds_scalar
    const int preds_scalar_shape[1] = {64};
    idnnlSetTensorNdDescriptor(preds_scalar_.tdesc_, IDNNL_TENSOR_ND,
                              IDNNL_DATA_FLOAT, preds_scalar_shape, 1);
    ret = hdplMalloc(&preds_scalar_.data_, 64 * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc preds_scalar_ return %d\n", ret);
      return;
    }
    void* preds_scalar_ptr[] = {preds_scalar_.data_};
    idnnlSetMemoryDescriptor(preds_scalar_.mdesc_, preds_scalar_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(preds_scalar_.data_, dir_cls_preds_scale, 64 * 4,
              hdplMemcpyHostToDevice);

    // scores_scalar
    const int scores_scalar_shape[1] = {64};
    idnnlSetTensorNdDescriptor(scores_scalar_.tdesc_, IDNNL_TENSOR_ND,
                              IDNNL_DATA_FLOAT, scores_scalar_shape, 1);
    ret = hdplMalloc(&scores_scalar_.data_, 64 * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc scores_scalar_ return %d\n", ret);
      return;
    }
    void* scores_scalar_ptr[] = {scores_scalar_.data_};
    idnnlSetMemoryDescriptor(scores_scalar_.mdesc_, scores_scalar_ptr, 1,
                            IDNNL_MEM_GM);
    hdplMemcpy(scores_scalar_.data_, cls_scores_scale, 64 * 4,
              hdplMemcpyHostToDevice);

    // bbox_scalar
    const int bbox_scalar_shape[1] = {64};
    idnnlSetTensorNdDescriptor(bbox_scalar_.tdesc_, IDNNL_TENSOR_ND,
                              IDNNL_DATA_FLOAT, bbox_scalar_shape, 1);
    ret = hdplMalloc(&bbox_scalar_.data_, 64 * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc bbox_scalar_ return %d\n", ret);
      return;
    }
    void* bbox_scalar_ptr[] = {bbox_scalar_.data_};
    idnnlSetMemoryDescriptor(bbox_scalar_.mdesc_, bbox_scalar_ptr, 1, IDNNL_MEM_GM);
    hdplMemcpy(bbox_scalar_.data_, bbox_preds_scale,
              sizeof(bbox_preds_scale) / sizeof(bbox_preds_scale[0]) * 4,
              hdplMemcpyHostToDevice);

    free(anchors_px);
    free(anchors_py);
    free(anchors_pz);
    free(anchors_dx);
    free(anchors_dy);
    free(anchors_dz);
    free(anchors_ro);
    free(anchors_diagonal);

    // bbox_preds
    idnnlSetTensor4dDescriptor(bbox_preds_.tdesc_, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                              116, 116, 42);
    idnnlSetTensor4dDescriptor(cls_scores_.tdesc_, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8,
                                1, 116, 116, 18);
    idnnlSetTensor4dDescriptor(dir_cls_preds_.tdesc_, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8,
                                1, 116, 116, 12);
    idnnlSetMemoryDescriptor(bbox_preds_.mdesc_, &bbox_preds_.data_, 1, IDNNL_MEM_GM);
    idnnlSetMemoryDescriptor(cls_scores_.mdesc_, &cls_scores_.data_, 1, IDNNL_MEM_GM);
    idnnlSetMemoryDescriptor(dir_cls_preds_.mdesc_, &dir_cls_preds_.data_, 1, IDNNL_MEM_GM);

    // out_box
    const int out_box_shape[2] = {1024, 9};
    idnnlSetTensorNdDescriptor(out_box_.tdesc_, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                              out_box_shape, 2);
    ret = hdplMalloc(&out_box_.data_, 1024 * 9 * 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc out_box_ return %d\n", ret);
      return;
    }
    void* out_box_ptr[] = {out_box_.data_};
    idnnlSetMemoryDescriptor(out_box_.mdesc_, out_box_ptr, 1, IDNNL_MEM_GM);
    // out_box_num
    const int out_box_num_shape[1] = {1};
    idnnlSetTensorNdDescriptor(out_box_num_.tdesc_, IDNNL_TENSOR_ND,
                              IDNNL_DATA_INT32, out_box_num_shape, 1);
    ret = hdplMalloc(&out_box_num_.data_, 4);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc out_box_num return %d\n", ret);
      return;
    }
    void* out_box_num_ptr[] = {out_box_num_.data_};
    idnnlSetMemoryDescriptor(out_box_num_.mdesc_, out_box_num_ptr, 1, IDNNL_MEM_GM);

    idnnlCreateWorkspaceDescriptor(&detection_workspace_);
    idnnlGetPillarsDetectionWorkspaceSize(nullptr, kNumAnchor, kNumClass,
                                          kNumOutputBoxFeature, rpnOutputHeight,
                                          rpnOutputWidth, detection_workspace_);
    int post_workspace_size = 0;
    idnnlGetGMWorkspaceSize(detection_workspace_, &post_workspace_size);
    std::cout << "detection workspace size " << post_workspace_size << std::endl;
    ret = hdplMalloc(&detection_wsdata_, post_workspace_size);
    if (ret != hdplError_t::hdplSuccess) {
      printf("hdplMalloc detection_workspace return %d\n", ret);
      return;
    }
    idnnlSetGMWorkspace(detection_workspace_, detection_wsdata_);
  }

  void Run() {
    // Voxelization
    idnnlRvvVoxelization(
      handle_,
      points_.tdesc_, points_.mdesc_,
      voxel_.tdesc_, voxel_.mdesc_,
      coors_.tdesc_, coors_.mdesc_,
      num_per_pillar_.tdesc_, num_per_pillar_.mdesc_,
      voxelization_workspace_,
      gridXSize, gridYSize, gridZSize, kMinXRange, kMinYRange, kMinZRange,
      kPillarXSize, kPillarYSize, kPillarZSize, quant_scale);

    // pfe1
    pfe1_module_.Run();

    // scatter
    idnnlScatterTranspose(handle_,
                          coors_.tdesc_, coors_.mdesc_,
                          active_pillar_.mdesc_, left_pillar_count_,
                          pfe1_out_.tdesc_, pfe1_out_.mdesc_,
                          scatter_out_.tdesc_, scatter_out_.mdesc_, scatter_out_.mdesc_,
                          voxelization_workspace_);

    // rpn
    rpn_module_.Run();

    // pillarsdetection
    idnnlRvvPillarsDetection(
      handle_,
      bbox_preds_.tdesc_, bbox_preds_.mdesc_,
      cls_scores_.tdesc_, cls_scores_.mdesc_,
      dir_cls_preds_.tdesc_, dir_cls_preds_.mdesc_,
      anchors_px_.tdesc_, anchors_px_.mdesc_,
      anchors_py_.tdesc_, anchors_py_.mdesc_,
      anchors_pz_.tdesc_, anchors_pz_.mdesc_,
      anchors_dx_.tdesc_, anchors_dx_.mdesc_,
      anchors_dy_.tdesc_, anchors_dy_.mdesc_,
      anchors_dz_.tdesc_, anchors_dz_.mdesc_,
      anchors_ro_.tdesc_, anchors_ro_.mdesc_,
      anchors_diagonal_.tdesc_, anchors_diagonal_.mdesc_,
      out_box_.tdesc_, out_box_.mdesc_,
      out_box_num_.tdesc_, out_box_num_.mdesc_,
      detection_workspace_,
      bbox_scalar_.tdesc_, bbox_scalar_.mdesc_,
      scores_scalar_.tdesc_, scores_scalar_.mdesc_,
      preds_scalar_.tdesc_, preds_scalar_.mdesc_,
      kNumClass, kNumAnchor, kNumOutputBoxFeature, kNumBoxCorners,
      score_threshold, nms_overlap_threshold);
  }

  void Sync() {
    hdplStreamSynchronize(stream_);
  }

  void CheckResult() {
    int output_num[1] = {0};
    hdplMemcpy(output_num, out_box_num_.data_, 4, hdplMemcpyDeviceToHost);
    printf("detection output_num %d\n", output_num[0]);
    float rvv_out_detections[1024 * 9] = {0};
    hdplMemcpy(rvv_out_detections, out_box_.data_, 1024 * 9 * 4,
              hdplMemcpyDeviceToHost);
    std::vector<std::string> classes({"Car", "Pedestrian", "Cyclist"});
    int num_objects = output_num[0];
    for (int i = 0; i < num_objects; ++i) {
      // read params of bounding box
      // normal boxes: x, y, z, l, w, h, r, 注意pp的后处理cu代码里是w,l,h。
      // 在PointPillarsDetector里又弄成l,w,h, 并对yaw方向取反
      // 这里直接处理成l,w,h并不对yaw + pi/2符合mmdetect3d定义的雷达坐标系。
      float x = rvv_out_detections[i * 9 + 0];
      float y = rvv_out_detections[i * 9 + 1];
      float z = rvv_out_detections[i * 9 + 2];
      float dl = rvv_out_detections[i * 9 + 4];
      float dw = rvv_out_detections[i * 9 + 3];
      float dh = rvv_out_detections[i * 9 + 5];
      float yaw = rvv_out_detections[i * 9 + 6];

      yaw += M_PI / 2;
      yaw += M_PI;  // 对齐到mmdet3d的雷达坐标系
      yaw = std::atan2(sinf(yaw), cosf(yaw));
      yaw = -yaw;

      float label = static_cast<int>(rvv_out_detections[i * 9 + 7]);
      // out_labels->push_back(label);
      float score = rvv_out_detections[i * 9 + 8];
      std::string name = "UNKNOWN";
      if (label < 3) {
        name = classes[label];
      }
      std::cout << name << " " << dl << " " << dw << " " << dh << " " << x << " "
                << y << " " << z << " " << yaw << " " << score << std::endl;
    }
    if (!(num_objects >= 41 && num_objects <= 42)) {
      printf("============point_pillars result error===========\n");
    } else {
      printf("============point_pillars result ok===========\n");
    }
    // assert(num_objects >= 35 && num_objects <= 36);
  }

 protected:
  idnnlHandle_t handle_;
  hdplStream_t stream_;
  // voxelization
  float quant_scale = 144.66707611634487680628761143145;
  IdnnlDesc points_;
  IdnnlDesc coors_;
  IdnnlDesc num_per_pillar_;
  IdnnlDesc voxel_;
  idnnlWorkspaceDescriptor_t voxelization_workspace_;
  void* voxelization_wsdata_ = nullptr;
  // pfe1
  tcim::Module pfe1_module_;
  // scatter
  IdnnlDesc pfe1_out_;
  IdnnlDesc scatter_out_;
  IdnnlDesc active_pillar_;
  int left_pillar_count_;
  // rpn
  tcim::Module rpn_module_;
  // pillarsdetection
  IdnnlDesc anchors_px_;
  IdnnlDesc anchors_py_;
  IdnnlDesc anchors_pz_;
  IdnnlDesc anchors_dx_;
  IdnnlDesc anchors_dy_;
  IdnnlDesc anchors_dz_;
  IdnnlDesc anchors_ro_;
  IdnnlDesc anchors_diagonal_;
  IdnnlDesc preds_scalar_;
  IdnnlDesc scores_scalar_;
  IdnnlDesc bbox_scalar_;
  IdnnlDesc bbox_preds_;
  IdnnlDesc cls_scores_;
  IdnnlDesc dir_cls_preds_;
  IdnnlDesc out_box_;
  IdnnlDesc out_box_num_;
  idnnlWorkspaceDescriptor_t detection_workspace_;
  void* detection_wsdata_ = nullptr;
};


typedef struct {
  // std::map<std::string, tcim::Tensor> tmap;
  uint64_t req_id;
} TaskInfo;


typedef struct {
  std::queue<TaskInfo> queue;
  std::mutex mutex;
  std::condition_variable cond;
} TaskQueue;


typedef struct {
  int sample_cnt = 0;
  uint32_t max_cost = 0;
  uint32_t total_cost = 0;
} ThreadInfo;


int main(int argc, char* argv[]) {
  int point_num = 54908;
  size_t test_count = 1;
  int thread_num = 1;
  if (argc >= 2) {
    thread_num = atoi(argv[1]);
  }
  if (argc >= 3) {
    test_count = atoi(argv[2]);
  }
  std::vector<std::thread> threads;
  printf("thread_num: %d, test_num: %d\n\n", thread_num, test_count);
  std::vector<std::shared_ptr<PointPillars>> pps;
  for (int i = 0; i < thread_num; i++) {
    std::shared_ptr<PointPillars> pp(new PointPillars(point_num));
    pps.push_back(pp);
  }

  // prepare task
  TaskQueue qin;
  for (int i = 0; i < test_count; i++) {
    TaskInfo tinfo;
    tinfo.req_id = i;
    qin.queue.push(tinfo);
  }
  std::cout << "sample queue size is " << qin.queue.size() << std::endl;

  ThreadInfo threads_info[thread_num];
  Barrier barrier(thread_num);
  auto thread_func = [](int tid, std::shared_ptr<PointPillars>& pp, TaskQueue& qin, ThreadInfo& thread_info, Barrier& barrier) {
    // wait until all threads ready
    barrier.barrier();
    printf("\n===> thread %d infer start...\n", tid);

    while (true) {
      // get data from the task queue
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      if (qin.queue.empty()) {
        lock_in.unlock();
        break;
      }
      auto req_id = qin.queue.front().req_id;
      qin.queue.pop();
      lock_in.unlock();

      auto start = GET_TIME();
      pp->Run();
      auto end = GET_TIME();
      auto cost = GET_COST(start, end);
      thread_info.total_cost += cost;
      if (thread_info.max_cost < cost) thread_info.max_cost = cost;
      thread_info.sample_cnt++;
    }
    pp->Sync();

    printf("\n===> thread %d completed. %d sampels tested.\n", tid, thread_info.sample_cnt);
    barrier.barrier();
  };

  for (int i = 0; i < thread_num; i++) {
    threads.push_back(std::thread(thread_func, i, std::ref(pps[i]), std::ref(qin),
                      std::ref(threads_info[i]), std::ref(barrier)));
  }

  barrier.wait();
  barrier.reset();
  auto start = GET_TIME();
  barrier.wait();
  auto end = GET_TIME();

  // 6. wait all threads done
  for (auto & t: threads) {
    t.join();
  }

  for (int i = 0; i < thread_num; i++) {
    printf("\nthread %d check result:\n", i);
    pps[i]->CheckResult();
  }

  float total_cost = GET_COST(start, end) / 1000.0;
  printf("\npointpillars test_num %u cost %fms, thread_num %d qps %f\n",
          test_count, total_cost, thread_num,
          1000 / (total_cost / test_count));

  return 0;
}
