// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "hdpl/hdpl_runtime.h"
#include "idnnl/idnnl_utils.h"

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
#define RUN_VOXEL 1
#define RUN_PFE 1
#define RUN_SCATTER 1
#define RUN_RPN 1
#define RUN_POST_PROCESS 1

static tvm::hdpl::Module pfe1_aot_module = tvm::hdpl::LoadModelPackage(
    "../compile_model/tcim_pointpillars_pfe", "aot");
static tvm::hdpl::Module rpn_stream0 = tvm::hdpl::LoadModelPackage(
    "../compile_model/tcim_pointpillars_rpn", "aot");

// run voxelization
void runVoxelization(int point_num) {
  int point_dim_num = kNumPointFeature;
  // build points
  void* host_point = malloc(point_num * point_dim_num * sizeof(point_dim_num));
  std::string point_file = "point_464.txt";
  readPointFile<float>(point_file, point_dim_num, host_point);
  printf("point[0] %f point[54908] %f \n",
         (reinterpret_cast<float*>(host_point))[0],
         (reinterpret_cast<float*>(host_point))[54907 * 5]);
  // build op input and output params
  idnnlHandle_t handle0;
  idnnlCreate(&handle0);
  idnnlWorkspaceDescriptor_t workspace;
  idnnlCreateWorkspaceDescriptor(&workspace);
  idnnlMemoryDescriptor_t point_ddr;
  idnnlCreateMemoryDescriptor(&point_ddr);
  idnnlTensorDescriptor_t point_desc;
  idnnlCreateTensorDescriptor(&point_desc);
  const int point_shape[2] = {point_num, point_dim_num};
  idnnlSetTensorNdDescriptor(point_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             point_shape, 2);
  void* point_device = nullptr;
  hdplMalloc(&point_device, point_num * point_dim_num * sizeof(float));
  void* point_ptr[] = {point_device};
  idnnlSetMemoryDescriptor(point_ddr, point_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(point_device, host_point,
             point_num * point_dim_num * sizeof(float), hdplMemcpyHostToDevice);
  free(host_point);
  // voxel
  idnnlTensorDescriptor_t voxel_desc;
  idnnlCreateTensorDescriptor(&voxel_desc);
  const int voxel_shape[3] = {kMaxNumPillars, kMaxNumPointsPerPillar,
                              point_dim_num};
  idnnlSetTensorNdDescriptor(voxel_desc, IDNNL_TENSOR_ND, IDNNL_DATA_INT8,
                             voxel_shape, 3);
  idnnlMemoryDescriptor_t voxel_ddr;
  idnnlCreateMemoryDescriptor(&voxel_ddr);
  void* voxel_device = nullptr;
  hdplMalloc(&voxel_device, kMaxNumPillars * kMaxNumPointsPerPillar *
                                ROUND_UP(sizeof(int8_t) * point_dim_num, 64));
  void* voxel_ptr[] = {voxel_device};
  idnnlSetMemoryDescriptor(voxel_ddr, voxel_ptr, 1, IDNNL_MEM_GM);
  // coors
  idnnlTensorDescriptor_t coors_desc;
  idnnlCreateTensorDescriptor(&coors_desc);
  const int coors_shape[] = {kMaxNumPillars, 4};
  idnnlSetTensorNdDescriptor(coors_desc, IDNNL_TENSOR_ND, IDNNL_DATA_INT16,
                             coors_shape, 2);
  idnnlMemoryDescriptor_t coors_ddr;
  idnnlCreateMemoryDescriptor(&coors_ddr);
  void* coors_device = nullptr;
  hdplMalloc(&coors_device, kMaxNumPillars * ROUND_UP(4 * sizeof(int16_t), 64));
  void* coors_ptr[] = {coors_device};
  idnnlSetMemoryDescriptor(coors_ddr, coors_ptr, 1, IDNNL_MEM_GM);
  // num_per_pillar
  idnnlTensorDescriptor_t num_per_pillar_desc;
  idnnlCreateTensorDescriptor(&num_per_pillar_desc);
  const int num_per_pillar_shape[] = {kMaxNumPillars};
  idnnlSetTensorNdDescriptor(num_per_pillar_desc, IDNNL_TENSOR_ND,
                             IDNNL_DATA_INT16, num_per_pillar_shape, 1);
  idnnlMemoryDescriptor_t num_per_pillar_ddr;
  idnnlCreateMemoryDescriptor(&num_per_pillar_ddr);
  void* num_per_pillar_device = nullptr;
  hdplMalloc(&num_per_pillar_device, kMaxNumPillars * sizeof(int16_t));
  void* num_per_pillar_ptr[] = {num_per_pillar_device};
  idnnlSetMemoryDescriptor(num_per_pillar_ddr, num_per_pillar_ptr, 1,
                           IDNNL_MEM_GM);
  // get gm workspace size
  printf(
      "voxelization workspace h %d, w %d , kMaxNumPointsPerPillar %d, "
      "kMaxNumPillars %d \n,",
      h, w, kMaxNumPointsPerPillar, kMaxNumPillars);
  idnnlGetVoxelizationWorkspace(handle0, h, w, kMaxNumPointsPerPillar,
                                kMaxNumPillars, workspace);
  int size = 150 * 1024 * 1024;
  idnnlGetGMWorkspaceSize(workspace, &size);
  void* workspace_gm_addr = nullptr;
  int stream_num = 1;
  hdplMalloc(&workspace_gm_addr, size * stream_num);
  std::cout << "workspace size " << size * stream_num / 1024 / 1024
            << std::endl;
  idnnlSetGMWorkspace(workspace, workspace_gm_addr);
  // tmp quant_scale
  float quant_scale = 142.21418261353976512690209203624;
  // pfe part1 ddr
  void* pfe1_output_addr = nullptr;
  hdplMalloc(&pfe1_output_addr, 12032 * 64);
  void* pfe1_output_ptr[] = {pfe1_output_addr};
  // pfe part2 ddr
  void* pfe2_output_addr = nullptr;
  void* pfe2_output_ptr[] = {pfe2_output_addr};

  void* scatter_out_ddr = nullptr;
  int scatter_output_size = h * w * 64;
  hdplError_t ret22 = hdplMalloc(&scatter_out_ddr, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_out_ddr return %d\r\n", ret22);
    return;
  }
  void* scatter_in_ddr = nullptr;
  ret22 = hdplMalloc(&scatter_in_ddr, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_in_ddr return %d\r\n", ret22);
    return;
  }

  void* dir_cls_preds_ddr = nullptr;
  void* bbox_preds_ddr = nullptr;
  void* cls_scores_ddr = nullptr;
  int rpn_output_size = 116 * 116 * 64;
  hdplMalloc(&dir_cls_preds_ddr, rpn_output_size);
  hdplMalloc(&bbox_preds_ddr, rpn_output_size);
  hdplMalloc(&cls_scores_ddr, rpn_output_size);

  idnnlWorkspaceDescriptor_t post_workspace;
  idnnlCreateWorkspaceDescriptor(&post_workspace);
  idnnlGetPillarsDetectionWorkspaceSize(nullptr, kNumAnchor, kNumClass,
                                        kNumOutputBoxFeature, rpnOutputHeight,
                                        rpnOutputWidth, post_workspace);
  int post_workspace_size = 0;
  idnnlGetGMWorkspaceSize(post_workspace, &post_workspace_size);
  std::cout << "post workspace size " << post_workspace_size << std::endl;
  void* post_workspace_addr = nullptr;
  hdplMalloc(&post_workspace_addr, post_workspace_size);
  idnnlSetGMWorkspace(post_workspace, post_workspace_addr);
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
      0.05936046317219734,  0.05953904241323471,  0.06364354491233826,
      0.062169790267944336, 0.03634127229452133,  0.03676672279834747,
      0.036799922585487366, 0.037349436432123184, 0.03381054103374481,
      0.03010406531393528,  0.03535057604312897,  0.03503552824258804};
  float bbox_preds_scale[42] = {
      0.002339052502065897,  0.003147886833176017,  0.090100958943367,
      0.004270119126886129,  0.003515428863465786,  0.006119038909673691,
      0.018384935334324837,  0.0025356377009302378, 0.0020904794801026583,
      0.07425294816493988,   0.004587132483720779,  0.00309442775323987,
      0.006698254030197859,  0.06386694312095642,   0.015490180812776089,
      0.010868524201214314,  0.10662107169628143,   0.07452819496393204,
      0.041057903319597244,  0.00815042294561863,   0.08061078935861588,
      0.0034649132285267115, 0.008832905441522598,  0.1265384405851364,
      0.06770142912864685,   0.02310025878250599,   0.009206191636621952,
      0.12291557341814041,   0.011366584338247776,  0.007686358876526356,
      0.09641500562429428,   0.05220061168074608,   0.015181024558842182,
      0.005787280388176441,  0.10884343832731247,   0.0038979898672550917,
      0.012594811618328094,  0.10921623557806015,   0.05207594111561775,
      0.011651678010821342,  0.009798778221011162,  0.11718147993087769};
  float cls_scores_scale[18] = {
      0.09308841079473495, 0.34382420778274536, 0.3854471445083618,
      0.08417491614818573, 0.3102761209011078,  0.39853090047836304,
      0.27744877338409424, 0.21648313105106354, 0.2122734934091568,
      0.23946698009967804, 0.21223139762878418, 0.22136308252811432,
      0.24621863663196564, 0.2012709379196167,  0.21804001927375793,
      0.2959044277667999,  0.21199274063110352, 0.2264881581068039};

  // init post_process params
  // 初始化anchor
  float* anchors_px_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  float* anchors_py_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  float* anchors_pz_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  float* anchors_dx_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  float* anchors_dy_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  float* anchors_dz_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  float* anchors_ro_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  float* anchors_diagonal_ = reinterpret_cast<float*>(malloc(kNumAnchor * 4));
  for (int i = 0; i < kNumAnchor; ++i) {
    anchors_px_[i] = 0;
    anchors_py_[i] = 0;
    anchors_pz_[i] = 0;
    anchors_dx_[i] = 0;
    anchors_dy_[i] = 0;
    anchors_dz_[i] = 0;
    anchors_ro_[i] = 0;
    anchors_diagonal_[i] = 0;
  }
  // printf("anchors_px_=0x%x...\r\n", anchors_px_);
  // printf("anchors_dx_=0x%x...\r\n", anchors_dx_);
  int ind = 0;
  for (size_t head = 0; head < kNumAnchorSets.size(); ++head) {
    float x_stride = kPillarXSize * kAnchorStrides[head];
    float y_stride = kPillarYSize * kAnchorStrides[head];
    int x_ind_start = kAnchorRanges[head * 4 + 0] / kAnchorStrides[head];
    int x_ind_end = kAnchorRanges[head * 4 + 1] / kAnchorStrides[head];
    int y_ind_start = kAnchorRanges[head * 4 + 2] / kAnchorStrides[head];
    int y_ind_end = kAnchorRanges[head * 4 + 3] / kAnchorStrides[head];
    // std::cout << "x_ind_start : " << x_ind_start << "| x_ind_end: " <<
    // x_ind_end; std::cout << "y_ind_start : " << y_ind_start << "| y_ind_end:
    // " << y_ind_end; coors of first anchor's center
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
            anchors_px_[ind] = anchor_x_count[x];
            anchors_py_[ind] = anchor_y_count[y];
            anchors_ro_[ind] = kAnchorRo[head][ro_count];
            anchors_pz_[ind] = kAnchorZCoors[head][c];
            anchors_dx_[ind] = kAnchorDxSizes[head][c];
            anchors_dy_[ind] = kAnchorDySizes[head][c];
            anchors_dz_[ind] = kAnchorDzSizes[head][c];
            anchors_diagonal_[ind] = sqrtf(anchors_dx_[ind] * anchors_dx_[ind] +
                                           anchors_dy_[ind] * anchors_dy_[ind]);
            ro_count++;
            ind++;
          }
        }
      }
    }
  }
  // anchors_px
  idnnlTensorDescriptor_t anchors_px_desc;
  idnnlCreateTensorDescriptor(&anchors_px_desc);
  const int anchors_px_shape[1] = {kNumAnchor};
  idnnlSetTensorNdDescriptor(anchors_px_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             anchors_px_shape, 1);
  idnnlMemoryDescriptor_t anchors_px_ddr;
  idnnlCreateMemoryDescriptor(&anchors_px_ddr);
  void* anchors_px_device = nullptr;
  hdplMalloc(&anchors_px_device, kNumAnchor * 4);
  void* anchors_px_ptr[] = {anchors_px_device};
  idnnlSetMemoryDescriptor(anchors_px_ddr, anchors_px_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(anchors_px_device, anchors_px_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // anchors_py
  idnnlTensorDescriptor_t anchors_py_desc;
  idnnlCreateTensorDescriptor(&anchors_py_desc);
  idnnlSetTensorNdDescriptor(anchors_py_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             anchors_px_shape, 1);
  idnnlMemoryDescriptor_t anchors_py_ddr;
  idnnlCreateMemoryDescriptor(&anchors_py_ddr);
  void* anchors_py_device = nullptr;
  hdplMalloc(&anchors_py_device, kNumAnchor * 4);
  void* anchors_py_ptr[] = {anchors_py_device};
  idnnlSetMemoryDescriptor(anchors_py_ddr, anchors_py_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(anchors_py_device, anchors_py_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // anchors_pz
  idnnlTensorDescriptor_t anchors_pz_desc;
  idnnlCreateTensorDescriptor(&anchors_pz_desc);
  const int anchors_pz_shape[1] = {kNumAnchor};
  idnnlSetTensorNdDescriptor(anchors_pz_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             anchors_pz_shape, 1);
  idnnlMemoryDescriptor_t anchors_pz_ddr;
  idnnlCreateMemoryDescriptor(&anchors_pz_ddr);
  void* anchors_pz_device = nullptr;
  hdplMalloc(&anchors_pz_device, kNumAnchor * 4);
  void* anchors_pz_ptr[] = {anchors_pz_device};
  idnnlSetMemoryDescriptor(anchors_pz_ddr, anchors_pz_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(anchors_pz_device, anchors_pz_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // anchors_dx
  idnnlTensorDescriptor_t anchors_dx_desc;
  idnnlCreateTensorDescriptor(&anchors_dx_desc);
  const int anchors_dx_shape[1] = {kNumAnchor};
  idnnlSetTensorNdDescriptor(anchors_dx_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             anchors_dx_shape, 1);
  idnnlMemoryDescriptor_t anchors_dx_ddr;
  idnnlCreateMemoryDescriptor(&anchors_dx_ddr);
  void* anchors_dx_device = nullptr;
  hdplMalloc(&anchors_dx_device, kNumAnchor * 4);
  void* anchors_dx_ptr[] = {anchors_dx_device};
  idnnlSetMemoryDescriptor(anchors_dx_ddr, anchors_dx_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(anchors_dx_device, anchors_dx_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // anchors_dy
  idnnlTensorDescriptor_t anchors_dy_desc;
  idnnlCreateTensorDescriptor(&anchors_dy_desc);
  const int anchors_dy_shape[1] = {kNumAnchor};
  idnnlSetTensorNdDescriptor(anchors_dy_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             anchors_dy_shape, 1);
  idnnlMemoryDescriptor_t anchors_dy_ddr;
  idnnlCreateMemoryDescriptor(&anchors_dy_ddr);
  void* anchors_dy_device = nullptr;
  hdplMalloc(&anchors_dy_device, kNumAnchor * 4);
  void* anchors_dy_ptr[] = {anchors_dy_device};
  idnnlSetMemoryDescriptor(anchors_dy_ddr, anchors_dy_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(anchors_dy_device, anchors_dy_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // anchors_dz
  idnnlTensorDescriptor_t anchors_dz_desc;
  idnnlCreateTensorDescriptor(&anchors_dz_desc);
  const int anchors_dz_shape[1] = {kNumAnchor};
  idnnlSetTensorNdDescriptor(anchors_dz_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             anchors_dz_shape, 1);
  idnnlMemoryDescriptor_t anchors_dz_ddr;
  idnnlCreateMemoryDescriptor(&anchors_dz_ddr);
  void* anchors_dz_device = nullptr;
  hdplMalloc(&anchors_dz_device, kNumAnchor * 4);
  void* anchors_dz_ptr[] = {anchors_dz_device};
  idnnlSetMemoryDescriptor(anchors_dz_ddr, anchors_dz_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(anchors_dz_device, anchors_dz_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // anchors_ro
  idnnlTensorDescriptor_t anchors_ro_desc;
  idnnlCreateTensorDescriptor(&anchors_ro_desc);
  const int anchors_ro_shape[1] = {kNumAnchor};
  idnnlSetTensorNdDescriptor(anchors_ro_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             anchors_ro_shape, 1);
  idnnlMemoryDescriptor_t anchors_ro_ddr;
  idnnlCreateMemoryDescriptor(&anchors_ro_ddr);
  void* anchors_ro_device = nullptr;
  hdplMalloc(&anchors_ro_device, kNumAnchor * 4);
  void* anchors_ro_ptr[] = {anchors_ro_device};
  idnnlSetMemoryDescriptor(anchors_ro_ddr, anchors_ro_ptr, 1, IDNNL_MEM_GM);
  printf("host anchors_ro_[0] %f\n", anchors_ro_[0]);
  hdplMemcpy(anchors_ro_device, anchors_ro_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // anchors_diagonal
  idnnlTensorDescriptor_t anchors_diagonal_desc;
  idnnlCreateTensorDescriptor(&anchors_diagonal_desc);
  const int anchors_diagonal_shape[1] = {kNumAnchor};
  idnnlSetTensorNdDescriptor(anchors_diagonal_desc, IDNNL_TENSOR_ND,
                             IDNNL_DATA_FLOAT, anchors_diagonal_shape, 1);
  idnnlMemoryDescriptor_t anchors_diagonal_ddr;
  idnnlCreateMemoryDescriptor(&anchors_diagonal_ddr);
  void* anchors_diagonal_device = nullptr;
  hdplMalloc(&anchors_diagonal_device, kNumAnchor * 4);
  void* anchors_diagonal_ptr[] = {anchors_diagonal_device};
  idnnlSetMemoryDescriptor(anchors_diagonal_ddr, anchors_diagonal_ptr, 1,
                           IDNNL_MEM_GM);
  hdplMemcpy(anchors_diagonal_device, anchors_diagonal_, kNumAnchor * 4,
             hdplMemcpyHostToDevice);
  // preds_scalar
  idnnlTensorDescriptor_t preds_scalar_desc;
  idnnlCreateTensorDescriptor(&preds_scalar_desc);
  const int preds_scalar_shape[1] = {64};
  idnnlSetTensorNdDescriptor(preds_scalar_desc, IDNNL_TENSOR_ND,
                             IDNNL_DATA_FLOAT, preds_scalar_shape, 1);
  idnnlMemoryDescriptor_t preds_scalar_ddr;
  idnnlCreateMemoryDescriptor(&preds_scalar_ddr);
  void* preds_scalar_device = nullptr;
  hdplMalloc(&preds_scalar_device, 64 * 4);
  void* preds_scalar_ptr[] = {preds_scalar_device};
  idnnlSetMemoryDescriptor(preds_scalar_ddr, preds_scalar_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(preds_scalar_device, dir_cls_preds_scale, 64 * 4,
             hdplMemcpyHostToDevice);
  // scores_scalar
  idnnlTensorDescriptor_t scores_scalar_desc;
  idnnlCreateTensorDescriptor(&scores_scalar_desc);
  const int scores_scalar_shape[1] = {64};
  idnnlSetTensorNdDescriptor(scores_scalar_desc, IDNNL_TENSOR_ND,
                             IDNNL_DATA_FLOAT, scores_scalar_shape, 1);
  idnnlMemoryDescriptor_t scores_scalar_ddr;
  idnnlCreateMemoryDescriptor(&scores_scalar_ddr);
  void* scores_scalar_device = nullptr;
  hdplMalloc(&scores_scalar_device, 64 * 4);
  void* scores_scalar_ptr[] = {scores_scalar_device};
  idnnlSetMemoryDescriptor(scores_scalar_ddr, scores_scalar_ptr, 1,
                           IDNNL_MEM_GM);
  hdplMemcpy(scores_scalar_device, cls_scores_scale, 64 * 4,
             hdplMemcpyHostToDevice);
  // bbox_scalar
  idnnlTensorDescriptor_t bbox_scalar_desc;
  idnnlCreateTensorDescriptor(&bbox_scalar_desc);
  // const int bbox_scalar_shape[1] = { sizeof(bbox_preds_scale) /
  // sizeof(bbox_preds_scale[0])  };
  const int bbox_scalar_shape[1] = {64};
  idnnlSetTensorNdDescriptor(bbox_scalar_desc, IDNNL_TENSOR_ND,
                             IDNNL_DATA_FLOAT, bbox_scalar_shape, 1);
  idnnlMemoryDescriptor_t bbox_scalar_ddr;
  idnnlCreateMemoryDescriptor(&bbox_scalar_ddr);
  void* bbox_scalar_device = nullptr;
  hdplMalloc(&bbox_scalar_device, 64 * 4);
  void* bbox_scalar_ptr[] = {bbox_scalar_device};
  idnnlSetMemoryDescriptor(bbox_scalar_ddr, bbox_scalar_ptr, 1, IDNNL_MEM_GM);
  hdplMemcpy(bbox_scalar_device, bbox_preds_scale,
             sizeof(bbox_preds_scale) / sizeof(bbox_preds_scale[0]) * 4,
             hdplMemcpyHostToDevice);

  int test_count = 1;

  tvm::runtime::NDArray::Container* data = new tvm::runtime::NDArray::Container(
      static_cast<void*>(reinterpret_cast<int8_t*>(voxel_device)),
      {1, 12032, 20, 64}, tvm::DataType::Int(8), {kDLHDPL, 0});
  // output
  tvm::runtime::NDArray::Container* pfe_part1_out =
      new tvm::runtime::NDArray::Container(
          static_cast<void*>(reinterpret_cast<int8_t*>(pfe1_output_addr)),
          {1, 12032, 1, 64}, tvm::DataType::Int(8), {kDLHDPL, 0});
  auto dev_139 = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(pfe_part1_out));
  // pfe1
  auto dev_voxel =
      tvm::runtime::NDArray(tvm::runtime::GetObjectPtr<tvm::Object>(data));

  pfe1_aot_module.SetInput("voxels", dev_voxel, "ND");
  pfe1_aot_module.SetOutput(pfe1_aot_module.GetOutputNameByIndex(0), dev_139);
  std::cout << "start run point_pillars" << std::endl;
  auto start = std::chrono::system_clock::now();
#if RUN_VOXEL
  for (int i = 0; i < test_count; i++) {
    auto ret = idnnlRvvVoxelization(
        handle0, point_desc, point_ddr, voxel_desc, voxel_ddr, coors_desc,
        coors_ddr, num_per_pillar_desc, num_per_pillar_ddr, workspace,
        gridXSize, gridYSize, gridZSize, kMinXRange, kMinYRange, kMinZRange,
        kPillarXSize, kPillarYSize, kPillarZSize, quant_scale);
  }
#endif
#if RUN_PFE
  pfe1_aot_module.Run();
#endif
#if RUN_SCATTER
  idnnlMemoryDescriptor_t indices_mem;
  idnnlMemoryDescriptor_t update_mem;
  idnnlMemoryDescriptor_t output_mem;
  idnnlMemoryDescriptor_t input_mem;
  idnnlCreateMemoryDescriptor(&indices_mem);
  idnnlCreateMemoryDescriptor(&update_mem);
  idnnlCreateMemoryDescriptor(&output_mem);
  idnnlCreateMemoryDescriptor(&input_mem);
  // void* indices_mem_ptr[] = { pfe2_output_addr };
  void* indices_mem_ptr[] = {coors_device};
  void* update_mem_ptr[] = {pfe1_output_addr};
  idnnlSetMemoryDescriptor(indices_mem, indices_mem_ptr, 1, IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(update_mem, update_mem_ptr, 1, IDNNL_MEM_GM);
  void* scatter_output_ptr[] = {scatter_out_ddr};
  void* scatter_input_ptr[] = {scatter_in_ddr};
  idnnlSetMemoryDescriptor(output_mem, scatter_output_ptr, 1, IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(input_mem, scatter_input_ptr, 1, IDNNL_MEM_GM);
  // input_220 = nox->pfe_out_1_220;
  // input_to8 = nox->pfe_out_2_to8;
  idnnlTensorDescriptor_t indices_desc;
  idnnlTensorDescriptor_t update_desc;
  idnnlTensorDescriptor_t output_desc;
  idnnlCreateTensorDescriptor(&indices_desc);
  idnnlCreateTensorDescriptor(&update_desc);
  idnnlCreateTensorDescriptor(&output_desc);
  // loop_time need check
  //   printf("loop time %d ======== tail pillar %d\n", loop_time, tail_pillar);
  idnnlSetTensor4dDescriptor(indices_desc, IDNNL_TENSOR_NCHW, IDNNL_DATA_INT16,
                             1, 1, 12032,
                             4);  // need to check
  idnnlSetTensor4dDescriptor(update_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                             1, 12032,
                             64);  // need to check
  idnnlSetTensor4dDescriptor(output_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                             h, w,
                             64);  // need to check
  idnnlCreateWorkspaceDescriptor(&workspace);
  // int left_pillar_count = tail_pillar;
  int left_pillar_count = 9226;
  idnnlScatterTranspose(handle0, indices_desc, indices_mem, left_pillar_count,
                        update_desc, update_mem, output_desc, input_mem,
                        output_mem, workspace);
#endif
#if RUN_RPN
  tvm::runtime::NDArray::Container* rpn_in =
      new tvm::runtime::NDArray::Container(static_cast<void*>(scatter_out_ddr),
                                           {1, h, w, 64}, tvm::DataType::Int(8),
                                           {kDLHDPL, 0});
  auto dev_rpn_in =
      tvm::runtime::NDArray(tvm::runtime::GetObjectPtr<tvm::Object>(rpn_in));
  // rpn output
  tvm::runtime::NDArray::Container* dir_cls_preds_in =
      new tvm::runtime::NDArray::Container(
          static_cast<void*>(dir_cls_preds_ddr), {1, 116, 116, 12},
          tvm::DataType::Int(8), {kDLHDPL, 0});
  tvm::runtime::NDArray::Container* bbox_preds_in =
      new tvm::runtime::NDArray::Container(static_cast<void*>(bbox_preds_ddr),
                                           {1, 116, 116, 42},
                                           tvm::DataType::Int(8), {kDLHDPL, 0});
  tvm::runtime::NDArray::Container* cls_scores_in =
      new tvm::runtime::NDArray::Container(static_cast<void*>(cls_scores_ddr),
                                           {1, 116, 116, 18},
                                           tvm::DataType::Int(8), {kDLHDPL, 0});
  auto dir_cls_preds_dev_in = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(dir_cls_preds_in));
  auto bbox_preds_dev_in = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(bbox_preds_in));
  auto cls_scores_dev_in = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(cls_scores_in));
  rpn_stream0.SetInput("features", dev_rpn_in);
  rpn_stream0.SetOutput(rpn_stream0.GetOutputNameByIndex(1), bbox_preds_dev_in);
  rpn_stream0.SetOutput(rpn_stream0.GetOutputNameByIndex(2),
                        dir_cls_preds_dev_in);
  rpn_stream0.SetOutput(rpn_stream0.GetOutputNameByIndex(0), cls_scores_dev_in);
  rpn_stream0.Run();
#endif
// post_process
#if RUN_POST_PROCESS
  // bbox_preds
  idnnlTensorDescriptor_t bbox_desc;
  idnnlCreateTensorDescriptor(&bbox_desc);
  idnnlSetTensor4dDescriptor(bbox_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                             116, 116, 42);
  idnnlMemoryDescriptor_t bbox_ddr;
  idnnlCreateMemoryDescriptor(&bbox_ddr);
  void* bbox_ptr[] = {bbox_preds_ddr};
  idnnlSetMemoryDescriptor(bbox_ddr, bbox_ptr, 1, IDNNL_MEM_GM);
  idnnlTensorDescriptor_t scores_desc;
  idnnlCreateTensorDescriptor(&scores_desc);
  idnnlSetTensor4dDescriptor(scores_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                             116, 116, 18);
  idnnlMemoryDescriptor_t scores_ddr;
  idnnlCreateMemoryDescriptor(&scores_ddr);
  void* scores_ptr[] = {cls_scores_ddr};
  idnnlSetMemoryDescriptor(scores_ddr, scores_ptr, 1, IDNNL_MEM_GM);
  idnnlTensorDescriptor_t preds_desc;
  idnnlCreateTensorDescriptor(&preds_desc);
  idnnlSetTensor4dDescriptor(preds_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                             116, 116, 12);
  idnnlMemoryDescriptor_t preds_mem;
  idnnlCreateMemoryDescriptor(&preds_mem);
  void* preds_desc_ptr[] = {dir_cls_preds_ddr};
  idnnlSetMemoryDescriptor(preds_mem, preds_desc_ptr, 1, IDNNL_MEM_GM);
  // post_process output
  // out box
  idnnlTensorDescriptor_t out_box_desc;
  idnnlCreateTensorDescriptor(&out_box_desc);
  const int out_box_shape[2] = {1024, 9};
  idnnlSetTensorNdDescriptor(out_box_desc, IDNNL_TENSOR_ND, IDNNL_DATA_FLOAT,
                             out_box_shape, 2);
  idnnlMemoryDescriptor_t out_box_ddr;
  idnnlCreateMemoryDescriptor(&out_box_ddr);
  void* out_box_device = nullptr;
  hdplMalloc(&out_box_device, 1024 * 9 * 4);
  void* out_box_ptr[] = {out_box_device};
  idnnlSetMemoryDescriptor(out_box_ddr, out_box_ptr, 1, IDNNL_MEM_GM);
  // out box_num
  idnnlTensorDescriptor_t out_box_num_desc;
  idnnlCreateTensorDescriptor(&out_box_num_desc);
  const int out_box_num_shape[1] = {1};
  idnnlSetTensorNdDescriptor(out_box_num_desc, IDNNL_TENSOR_ND,
                             IDNNL_DATA_INT32, out_box_num_shape, 1);
  idnnlMemoryDescriptor_t out_box_num_ddr;
  idnnlCreateMemoryDescriptor(&out_box_num_ddr);
  void* out_box_num_device = nullptr;
  hdplMalloc(&out_box_num_device, 4);
  void* out_box_num_ptr[] = {out_box_num_device};
  idnnlSetMemoryDescriptor(out_box_num_ddr, out_box_num_ptr, 1, IDNNL_MEM_GM);
  idnnlRvvPillarsDetection(
      handle0, bbox_desc, bbox_ddr, scores_desc, scores_ddr, preds_desc,
      preds_mem, anchors_px_desc, anchors_px_ddr, anchors_py_desc,
      anchors_py_ddr, anchors_pz_desc, anchors_pz_ddr, anchors_dx_desc,
      anchors_dx_ddr, anchors_dy_desc, anchors_dy_ddr, anchors_dz_desc,
      anchors_dz_ddr, anchors_ro_desc, anchors_ro_ddr, anchors_diagonal_desc,
      anchors_diagonal_ddr, out_box_desc, out_box_ddr, out_box_num_desc,
      out_box_num_ddr, post_workspace, bbox_scalar_desc, bbox_scalar_ddr,
      scores_scalar_desc, scores_scalar_ddr, preds_scalar_desc,
      preds_scalar_ddr, kNumClass, kNumAnchor, kNumOutputBoxFeature,
      kNumBoxCorners, score_threshold, nms_overlap_threshold);
#endif
  hdplDeviceSynchronize();
  auto finish = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  auto total_time = (duration.count());
  printf("point_pillars test count %d cost %ldus each time %ldus \n",
         test_count, total_time, total_time / test_count);
  std::cout << "end run point_pillars" << std::endl;
  int output_num[1] = {0};
  printf("out_box_num_ddr %p\n", out_box_num_device);
  hdplMemcpy(output_num, out_box_num_device, 4, hdplMemcpyDeviceToHost);
  printf("detection output_num %d out_box_num_device %p\n", output_num[0],
         out_box_num_device);
  float rvv_out_detections[1024 * 9] = {0};
  printf("out_box_num_ddr %p\n", out_box_device);
  hdplMemcpy(rvv_out_detections, out_box_device, 1024 * 9 * 4,
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
    float score = rvv_out_detections[i * 9 + 8];
    std::string name = "UNKNOWN";
    if (label < 3) {
      name = classes[label];
    }
    std::cout << name << " " << dl << " " << dw << " " << dh << " " << x << " "
              << y << " " << z << " " << yaw << " " << score << std::endl;
  }
  assert(num_objects >= 35 && num_objects <= 36);
  hdplFree(point_device);
  hdplFree(voxel_device);
  hdplFree(coors_device);
  hdplFree(num_per_pillar_device);
  hdplFree(workspace_gm_addr);
  hdplFree(pfe1_output_addr);
  hdplFree(scatter_out_ddr);
  hdplFree(scatter_in_ddr);
  hdplFree(dir_cls_preds_ddr);
  hdplFree(bbox_preds_ddr);
  hdplFree(cls_scores_ddr);
  hdplFree(post_workspace_addr);
  hdplFree(anchors_px_device);
  hdplFree(anchors_py_device);
  hdplFree(anchors_pz_device);
  hdplFree(anchors_dx_device);
  hdplFree(anchors_dy_device);
  hdplFree(anchors_dz_device);
  hdplFree(anchors_ro_device);
  hdplFree(anchors_diagonal_device);
  hdplFree(preds_scalar_device);
  hdplFree(scores_scalar_device);
  hdplFree(bbox_scalar_device);
  hdplFree(out_box_device);
  hdplFree(out_box_num_device);
}

int main() {
  int point_num = 54908;
  runVoxelization(point_num);
  return 0;
}
