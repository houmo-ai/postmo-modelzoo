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
#define VOXEL_S1 1
#define PFE_S1 1
#define SCATTER_S1 1
#define RPN_S1 1
#define POST_S1 1

static tvm::hdpl::Module pfe1_aot_module = tvm::hdpl::LoadModelPackage(
    "../pfe_1.hmm", "aot");
static tvm::hdpl::Module pfe1_aot_module_s1 = tvm::hdpl::LoadModelPackage(
    "../pfe_1.hmm", "aot");
static tvm::hdpl::Module pfe1_aot_module_s2 = tvm::hdpl::LoadModelPackage(
    "../pfe_1.hmm", "aot");
static tvm::hdpl::Module pfe1_aot_module_s3 = tvm::hdpl::LoadModelPackage(
    "../pfe_1.hmm", "aot");
static tvm::hdpl::Module rpn_stream0 = tvm::hdpl::LoadModelPackage(
    "../rpn.hmm", "aot");
static tvm::hdpl::Module rpn_stream1 = tvm::hdpl::LoadModelPackage(
    "../rpn.hmm", "aot");
static tvm::hdpl::Module rpn_stream2 = tvm::hdpl::LoadModelPackage(
    "../rpn.hmm", "aot");
static tvm::hdpl::Module rpn_stream3 = tvm::hdpl::LoadModelPackage(
    "../rpn.hmm", "aot");

// run voxelization
void runVoxelization(int point_num, size_t test_count) {
  int point_dim_num = kNumPointFeature;
  // build points
  void* host_point = malloc(point_num * point_dim_num * sizeof(point_dim_num));
  std::string point_file = "../data/point_464.txt";
  readPointFile<float>(point_file, point_dim_num, host_point);
  printf("point[0] %f point[54908] %f \n",
         (reinterpret_cast<float*>(host_point))[0],
         reinterpret_cast<float*>(host_point)[54907 * 5]);
  // build op input and output params
  idnnlHandle_t handle0;
  idnnlCreate(&handle0);
  idnnlHandle_t handle1;
  idnnlCreate(&handle1);
  idnnlHandle_t handle2;
  idnnlCreate(&handle2);
  idnnlHandle_t handle3;
  idnnlCreate(&handle3);
  idnnlMemoryDescriptor_t point_ddr;
  idnnlCreateMemoryDescriptor(&point_ddr);
  idnnlTensorDescriptor_t point_desc;
  idnnlCreateTensorDescriptor(&point_desc);
  hdplStream_t stream0_ = nullptr;
  hdplStream_t stream1_ = nullptr;
  hdplStream_t stream2_ = nullptr;
  hdplStream_t stream3_ = nullptr;
  hdplStreamCreate(&stream0_);
  hdplStreamCreate(&stream1_);
  hdplStreamCreate(&stream2_);
  hdplStreamCreate(&stream3_);
  idnnlSetStream(handle0, stream0_);
  idnnlSetStream(handle1, stream1_);
  idnnlSetStream(handle2, stream2_);
  idnnlSetStream(handle3, stream3_);
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

  idnnlMemoryDescriptor_t voxel_ddr_s1;
  idnnlCreateMemoryDescriptor(&voxel_ddr_s1);
  void* voxel_device_s1 = nullptr;
  hdplMalloc(&voxel_device_s1,
             kMaxNumPillars * kMaxNumPointsPerPillar *
                 ROUND_UP(sizeof(int8_t) * point_dim_num, 64));
  void* voxel_ptr_s1[] = {voxel_device_s1};
  idnnlSetMemoryDescriptor(voxel_ddr_s1, voxel_ptr_s1, 1, IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t voxel_ddr_s2;
  idnnlCreateMemoryDescriptor(&voxel_ddr_s2);
  void* voxel_device_s2 = nullptr;
  hdplMalloc(&voxel_device_s2,
             kMaxNumPillars * kMaxNumPointsPerPillar *
                 ROUND_UP(sizeof(int8_t) * point_dim_num, 64));
  void* voxel_ptr_s2[] = {voxel_device_s2};
  idnnlSetMemoryDescriptor(voxel_ddr_s2, voxel_ptr_s2, 1, IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t voxel_ddr_s3;
  idnnlCreateMemoryDescriptor(&voxel_ddr_s3);
  void* voxel_device_s3 = nullptr;
  hdplMalloc(&voxel_device_s3,
             kMaxNumPillars * kMaxNumPointsPerPillar *
                 ROUND_UP(sizeof(int8_t) * point_dim_num, 64));
  void* voxel_ptr_s3[] = {voxel_device_s3};
  idnnlSetMemoryDescriptor(voxel_ddr_s3, voxel_ptr_s3, 1, IDNNL_MEM_GM);

  // coors
  idnnlTensorDescriptor_t coors_desc;
  idnnlCreateTensorDescriptor(&coors_desc);
  const int coors_shape[] = {kMaxNumPillars, 4};
  idnnlSetTensorNdDescriptor(coors_desc, IDNNL_TENSOR_ND, IDNNL_DATA_INT16,
                             coors_shape, 2);
  idnnlMemoryDescriptor_t coors_ddr;
  idnnlCreateMemoryDescriptor(&coors_ddr);
  void* coors_device_s0 = nullptr;
  hdplMalloc(&coors_device_s0,
             kMaxNumPillars * ROUND_UP(4 * sizeof(int16_t), 64));
  void* coors_ptr[] = {coors_device_s0};
  idnnlSetMemoryDescriptor(coors_ddr, coors_ptr, 1, IDNNL_MEM_GM);
  void* coors_device_s1 = nullptr;
  hdplMalloc(&coors_device_s1,
             kMaxNumPillars * ROUND_UP(4 * sizeof(int16_t), 64));
  void* coors_ptr_s1[] = {coors_device_s1};
  idnnlMemoryDescriptor_t coors_ddr_s1;
  idnnlCreateMemoryDescriptor(&coors_ddr_s1);
  idnnlSetMemoryDescriptor(coors_ddr_s1, coors_ptr_s1, 1, IDNNL_MEM_GM);

  void* coors_device_s2 = nullptr;
  hdplMalloc(&coors_device_s2,
             kMaxNumPillars * ROUND_UP(4 * sizeof(int16_t), 64));
  void* coors_ptr_s2[] = {coors_device_s2};
  idnnlMemoryDescriptor_t coors_ddr_s2;
  idnnlCreateMemoryDescriptor(&coors_ddr_s2);
  idnnlSetMemoryDescriptor(coors_ddr_s2, coors_ptr_s2, 1, IDNNL_MEM_GM);

  void* coors_device_s3 = nullptr;
  hdplMalloc(&coors_device_s3,
             kMaxNumPillars * ROUND_UP(4 * sizeof(int16_t), 64));
  void* coors_ptr_s3[] = {coors_device_s3};
  idnnlMemoryDescriptor_t coors_ddr_s3;
  idnnlCreateMemoryDescriptor(&coors_ddr_s3);
  idnnlSetMemoryDescriptor(coors_ddr_s3, coors_ptr_s3, 1, IDNNL_MEM_GM);

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

  idnnlMemoryDescriptor_t num_per_pillar_ddr_s1;
  idnnlCreateMemoryDescriptor(&num_per_pillar_ddr_s1);
  void* num_per_pillar_device_s1 = nullptr;
  hdplMalloc(&num_per_pillar_device_s1, kMaxNumPillars * sizeof(int16_t));
  void* num_per_pillar_ptr_s1[] = {num_per_pillar_device_s1};
  idnnlSetMemoryDescriptor(num_per_pillar_ddr_s1, num_per_pillar_ptr_s1, 1,
                           IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t num_per_pillar_ddr_s2;
  idnnlCreateMemoryDescriptor(&num_per_pillar_ddr_s2);
  void* num_per_pillar_device_s2 = nullptr;
  hdplMalloc(&num_per_pillar_device_s2, kMaxNumPillars * sizeof(int16_t));
  void* num_per_pillar_ptr_s2[] = {num_per_pillar_device_s2};
  idnnlSetMemoryDescriptor(num_per_pillar_ddr_s2, num_per_pillar_ptr_s2, 1,
                           IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t num_per_pillar_ddr_s3;
  idnnlCreateMemoryDescriptor(&num_per_pillar_ddr_s3);
  void* num_per_pillar_device_s3 = nullptr;
  hdplMalloc(&num_per_pillar_device_s3, kMaxNumPillars * sizeof(int16_t));
  void* num_per_pillar_ptr_s3[] = {num_per_pillar_device_s3};
  idnnlSetMemoryDescriptor(num_per_pillar_ddr_s3, num_per_pillar_ptr_s3, 1,
                           IDNNL_MEM_GM);

  // get gm workspace size
  printf(
      "voxelization workspace h %d, w %d , kMaxNumPointsPerPillar %d, "
      "kMaxNumPillars %d \n,",
      h, w, kMaxNumPointsPerPillar, kMaxNumPillars);
  // tmp quant_scale
  // float quant_scale = 142.21418261353976512690209203624;
  float quant_scale = 144.66707611634487680628761143145;
  // pfe part1 ddr
  void* pfe1_output_addr = nullptr;
  hdplMalloc(&pfe1_output_addr, 12032 * 64);
  void* pfe1_output_ptr[] = {pfe1_output_addr};

  void* pfe1_output_addr_s1 = nullptr;
  hdplMalloc(&pfe1_output_addr_s1, 12032 * 64);
  void* pfe1_output_addr_s2 = nullptr;
  hdplMalloc(&pfe1_output_addr_s2, 12032 * 64);
  void* pfe1_output_addr_s3 = nullptr;
  hdplMalloc(&pfe1_output_addr_s3, 12032 * 64);

  void* scatter_out_ddr = nullptr;
  int scatter_output_size = h * w * 64;
  int ret22 = hdplMalloc(&scatter_out_ddr, scatter_output_size);
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

  void* scatter_out_ddr_s1 = nullptr;
  ret22 = hdplMalloc(&scatter_out_ddr_s1, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_out_ddr_s1 return %d\r\n", ret22);
    return;
  }
  void* scatter_in_ddr_s1 = nullptr;
  ret22 = hdplMalloc(&scatter_in_ddr_s1, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_in_ddr_s1 return %d\r\n", ret22);
    return;
  }

  void* scatter_out_ddr_s2 = nullptr;
  ret22 = hdplMalloc(&scatter_out_ddr_s2, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_out_ddr_s2 return %d\r\n", ret22);
    return;
  }
  void* scatter_in_ddr_s2 = nullptr;
  ret22 = hdplMalloc(&scatter_in_ddr_s2, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_in_ddr_s2 return %d\r\n", ret22);
    return;
  }

  void* scatter_out_ddr_s3 = nullptr;
  ret22 = hdplMalloc(&scatter_out_ddr_s3, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_out_ddr_s3 return %d\r\n", ret22);
    return;
  }
  void* scatter_in_ddr_s3 = nullptr;
  ret22 = hdplMalloc(&scatter_in_ddr_s3, scatter_output_size);
  if (ret22 != hdplError_t::hdplSuccess) {
    printf("hdplMalloc scatter_in_ddr_s3 return %d\r\n", ret22);
    return;
  }

  void* dir_cls_preds_ddr = nullptr;
  void* bbox_preds_ddr = nullptr;
  void* cls_scores_ddr = nullptr;
  int rpn_output_size = 116 * 116 * 64;
  hdplMalloc(&dir_cls_preds_ddr, rpn_output_size);
  hdplMalloc(&bbox_preds_ddr, rpn_output_size);
  hdplMalloc(&cls_scores_ddr, rpn_output_size);

  void* dir_cls_preds_ddr_s1 = nullptr;
  void* bbox_preds_ddr_s1 = nullptr;
  void* cls_scores_ddr_s1 = nullptr;
  hdplMalloc(&dir_cls_preds_ddr_s1, rpn_output_size);
  hdplMalloc(&bbox_preds_ddr_s1, rpn_output_size);
  hdplMalloc(&cls_scores_ddr_s1, rpn_output_size);

  void* dir_cls_preds_ddr_s2 = nullptr;
  void* bbox_preds_ddr_s2 = nullptr;
  void* cls_scores_ddr_s2 = nullptr;
  hdplMalloc(&dir_cls_preds_ddr_s2, rpn_output_size);
  hdplMalloc(&bbox_preds_ddr_s2, rpn_output_size);
  hdplMalloc(&cls_scores_ddr_s2, rpn_output_size);

  void* dir_cls_preds_ddr_s3 = nullptr;
  void* bbox_preds_ddr_s3 = nullptr;
  void* cls_scores_ddr_s3 = nullptr;
  hdplMalloc(&dir_cls_preds_ddr_s3, rpn_output_size);
  hdplMalloc(&bbox_preds_ddr_s3, rpn_output_size);
  hdplMalloc(&cls_scores_ddr_s3, rpn_output_size);

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

  idnnlWorkspaceDescriptor_t post_workspace_s1;
  idnnlCreateWorkspaceDescriptor(&post_workspace_s1);
  void* post_workspace_addr_s1 = nullptr;
  hdplMalloc(&post_workspace_addr_s1, post_workspace_size);
  idnnlSetGMWorkspace(post_workspace_s1, post_workspace_addr_s1);

  idnnlWorkspaceDescriptor_t post_workspace_s2;
  idnnlCreateWorkspaceDescriptor(&post_workspace_s2);
  void* post_workspace_addr_s2 = nullptr;
  hdplMalloc(&post_workspace_addr_s2, post_workspace_size);
  idnnlSetGMWorkspace(post_workspace_s2, post_workspace_addr_s2);

  idnnlWorkspaceDescriptor_t post_workspace_s3;
  idnnlCreateWorkspaceDescriptor(&post_workspace_s3);
  void* post_workspace_addr_s3 = nullptr;
  hdplMalloc(&post_workspace_addr_s3, post_workspace_size);
  idnnlSetGMWorkspace(post_workspace_s3, post_workspace_addr_s3);

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
  // w*h*6 总的anchor数量
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

  free(anchors_px_);
  free(anchors_py_);
  free(anchors_pz_);
  free(anchors_dx_);
  free(anchors_dy_);
  free(anchors_dz_);
  free(anchors_ro_);
  free(anchors_diagonal_);

  tvm::runtime::NDArray::Container* data = new tvm::runtime::NDArray::Container(
      static_cast<void*>(voxel_device), {1, 12032, 20, 64},
      tvm::DataType::Int(8), {kDLHDPL, 0});
  // output
  tvm::runtime::NDArray::Container* pfe_part1_out =
      new tvm::runtime::NDArray::Container(static_cast<void*>(pfe1_output_addr),
                                           {1, 12032, 1, 64},
                                           tvm::DataType::Int(8), {kDLHDPL, 0});
  auto dev_139 = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(pfe_part1_out));
  // pfe1
  auto dev_voxel =
      tvm::runtime::NDArray(tvm::runtime::GetObjectPtr<tvm::Object>(data));

  pfe1_aot_module.SetInput("voxels", dev_voxel, "ND");
  pfe1_aot_module.SetStream(stream0_);
  pfe1_aot_module.SetOutput(pfe1_aot_module.GetOutputNameByIndex(0), dev_139);

  // pfe stream 1
  tvm::runtime::NDArray::Container* data_s1 =
      new tvm::runtime::NDArray::Container(static_cast<void*>(voxel_device_s1),
                                           {1, 12032, 20, 64},
                                           tvm::DataType::Int(8), {kDLHDPL, 0});
  // output
  tvm::runtime::NDArray::Container* pfe_part1_out_s1 =
      new tvm::runtime::NDArray::Container(
          static_cast<void*>(pfe1_output_addr_s1), {1, 12032, 1, 64},
          tvm::DataType::Int(8), {kDLHDPL, 0});
  auto dev_139_s1 = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(pfe_part1_out_s1));
  // pfe1
  auto dev_voxel_s1 =
      tvm::runtime::NDArray(tvm::runtime::GetObjectPtr<tvm::Object>(data_s1));

  pfe1_aot_module_s1.SetInput("voxels", dev_voxel_s1, "ND");
  pfe1_aot_module_s1.SetStream(stream1_);
  pfe1_aot_module_s1.SetOutput(pfe1_aot_module_s1.GetOutputNameByIndex(0),
                               dev_139_s1);

  // pfe stream 2
  tvm::runtime::NDArray::Container* data_s2 =
      new tvm::runtime::NDArray::Container(
          reinterpret_cast<void*>(voxel_device_s2), {1, 12032, 20, 64},
          tvm::DataType::Int(8), {kDLHDPL, 0});
  // output
  tvm::runtime::NDArray::Container* pfe_part1_out_s2 =
      new tvm::runtime::NDArray::Container(
          reinterpret_cast<void*>(pfe1_output_addr_s2), {1, 12032, 1, 64},
          tvm::DataType::Int(8), {kDLHDPL, 0});
  auto dev_139_s2 = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(pfe_part1_out_s2));
  // pfe1
  auto dev_voxel_s2 =
      tvm::runtime::NDArray(tvm::runtime::GetObjectPtr<tvm::Object>(data_s2));

  pfe1_aot_module_s2.SetInput("voxels", dev_voxel_s2, "ND");
  pfe1_aot_module_s2.SetStream(stream2_);
  pfe1_aot_module_s2.SetOutput(pfe1_aot_module_s2.GetOutputNameByIndex(0),
                               dev_139_s2);

  // pfe stream 3
  tvm::runtime::NDArray::Container* data_s3 =
      new tvm::runtime::NDArray::Container(
          reinterpret_cast<void*>(voxel_device_s3), {1, 12032, 20, 64},
          tvm::DataType::Int(8), {kDLHDPL, 0});
  // output
  tvm::runtime::NDArray::Container* pfe_part1_out_s3 =
      new tvm::runtime::NDArray::Container(
          reinterpret_cast<void*>(pfe1_output_addr_s3), {1, 12032, 1, 64},
          tvm::DataType::Int(8), {kDLHDPL, 0});
  auto dev_139_s3 = tvm::runtime::NDArray(
      tvm::runtime::GetObjectPtr<tvm::Object>(pfe_part1_out_s3));
  // pfe1
  auto dev_voxel_s3 =
      tvm::runtime::NDArray(tvm::runtime::GetObjectPtr<tvm::Object>(data_s3));

  pfe1_aot_module_s3.SetInput("voxels", dev_voxel_s3, "ND");
  pfe1_aot_module_s3.SetStream(stream3_);
  pfe1_aot_module_s3.SetOutput(pfe1_aot_module_s3.GetOutputNameByIndex(0),
                               dev_139_s3);

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

  idnnlMemoryDescriptor_t out_box_ddr_s1;
  idnnlCreateMemoryDescriptor(&out_box_ddr_s1);
  void* out_box_device_s1 = nullptr;
  hdplMalloc(&out_box_device_s1, 1024 * 9 * 4);
  void* out_box_ptr_s1[] = {out_box_device_s1};
  idnnlSetMemoryDescriptor(out_box_ddr_s1, out_box_ptr_s1, 1, IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t out_box_ddr_s2;
  idnnlCreateMemoryDescriptor(&out_box_ddr_s2);
  void* out_box_device_s2 = nullptr;
  hdplMalloc(&out_box_device_s2, 1024 * 9 * 4);
  void* out_box_ptr_s2[] = {out_box_device_s2};
  idnnlSetMemoryDescriptor(out_box_ddr_s2, out_box_ptr_s2, 1, IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t out_box_ddr_s3;
  idnnlCreateMemoryDescriptor(&out_box_ddr_s3);
  void* out_box_device_s3 = nullptr;
  hdplMalloc(&out_box_device_s3, 1024 * 9 * 4);
  void* out_box_ptr_s3[] = {out_box_device_s3};
  idnnlSetMemoryDescriptor(out_box_ddr_s3, out_box_ptr_s3, 1, IDNNL_MEM_GM);

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

  idnnlMemoryDescriptor_t out_box_num_ddr_s1;
  idnnlCreateMemoryDescriptor(&out_box_num_ddr_s1);
  void* out_box_num_device_s1 = nullptr;
  hdplMalloc(&out_box_num_device_s1, 4);
  void* out_box_num_ptr_s1[] = {out_box_num_device_s1};
  idnnlSetMemoryDescriptor(out_box_num_ddr_s1, out_box_num_ptr_s1, 1,
                           IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t out_box_num_ddr_s2;
  idnnlCreateMemoryDescriptor(&out_box_num_ddr_s2);
  void* out_box_num_device_s2 = nullptr;
  hdplMalloc(&out_box_num_device_s2, 4);
  void* out_box_num_ptr_s2[] = {out_box_num_device_s2};
  idnnlSetMemoryDescriptor(out_box_num_ddr_s2, out_box_num_ptr_s2, 1,
                           IDNNL_MEM_GM);

  idnnlMemoryDescriptor_t out_box_num_ddr_s3;
  idnnlCreateMemoryDescriptor(&out_box_num_ddr_s3);
  void* out_box_num_device_s3 = nullptr;
  hdplMalloc(&out_box_num_device_s3, 4);
  void* out_box_num_ptr_s3[] = {out_box_num_device_s3};
  idnnlSetMemoryDescriptor(out_box_num_ddr_s3, out_box_num_ptr_s3, 1,
                           IDNNL_MEM_GM);

  idnnlWorkspaceDescriptor_t workspace;
  idnnlCreateWorkspaceDescriptor(&workspace);
  idnnlGetVoxelizationWorkspace(handle0, h, w, kMaxNumPointsPerPillar,
                                kMaxNumPillars, workspace);
  int size = 0;
  idnnlGetGMWorkspaceSize(workspace, &size);
  printf("GMWorkspaceSize: %d\n", size);
  // size = 105 * 1024 * 1024;  // todo need fix
  void* workspace_gm_addr = nullptr;
  // uint64_t ddr_reserve_start = 0;
  // uint64_t ddr_free_start = 0;
  // uint64_t ddr_gm_size = 0;
  // hdplReservedMemGetInfo(&ddr_reserve_start, &ddr_free_start, &ddr_gm_size);
  // workspace_gm_addr =
  //     reinterpret_cast<void*>(ddr_reserve_start + 580 * 1024 * 1024);
  // std::cout << "workspace size s0 " << 4 * size / 1024 / 1024 << std::endl;
  // printf("workspace_gm_addr 0x%p\n", workspace_gm_addr);
  hdplReservedMemAlloc(&workspace_gm_addr, size);
  idnnlSetGMWorkspace(workspace, workspace_gm_addr);

  void* workspace_gm_addr_s1 = nullptr;
      // reinterpret_cast<char*>(workspace_gm_addr) + 105 * 1024 * 1024;
  idnnlWorkspaceDescriptor_t workspace_s1;
  idnnlCreateWorkspaceDescriptor(&workspace_s1);
  hdplReservedMemAlloc(&workspace_gm_addr_s1, size);
  // hdplMalloc(&workspace_gm_addr_s1, size);
  idnnlSetGMWorkspace(workspace_s1, workspace_gm_addr_s1);

  void* workspace_gm_addr_s2 = nullptr;
      // reinterpret_cast<char*>(workspace_gm_addr_s1) + 105 * 1024 * 1024;
  idnnlWorkspaceDescriptor_t workspace_s2;
  idnnlCreateWorkspaceDescriptor(&workspace_s2);
  hdplReservedMemAlloc(&workspace_gm_addr_s2, size);
  // hdplMalloc(&workspace_gm_addr_s2, size);
  idnnlSetGMWorkspace(workspace_s2, workspace_gm_addr_s2);

  void* workspace_gm_addr_s3 = nullptr;
      // reinterpret_cast<char*>(workspace_gm_addr_s2) + 105 * 1024 * 1024;
  idnnlWorkspaceDescriptor_t workspace_s3;
  idnnlCreateWorkspaceDescriptor(&workspace_s3);
  hdplReservedMemAlloc(&workspace_gm_addr_s3, size);
  // hdplMalloc(&workspace_gm_addr_s2, size);
  // printf("workspace_gm_addr_s3 %p\n", workspace_gm_addr_s3);
  idnnlSetGMWorkspace(workspace_s3, workspace_gm_addr_s3);

  // scatter_stream0
  idnnlMemoryDescriptor_t indices_mem;
  idnnlMemoryDescriptor_t update_mem;
  idnnlMemoryDescriptor_t output_mem;
  idnnlMemoryDescriptor_t input_mem;
  idnnlCreateMemoryDescriptor(&indices_mem);
  idnnlCreateMemoryDescriptor(&update_mem);
  idnnlCreateMemoryDescriptor(&output_mem);
  idnnlCreateMemoryDescriptor(&input_mem);
  void* indices_mem_ptr[] = {coors_device_s0};
  void* update_mem_ptr[] = {pfe1_output_addr};
  idnnlSetMemoryDescriptor(indices_mem, indices_mem_ptr, 1, IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(update_mem, update_mem_ptr, 1, IDNNL_MEM_GM);
  void* scatter_output_ptr[] = {scatter_out_ddr};
  void* scatter_input_ptr[] = {scatter_in_ddr};
  idnnlSetMemoryDescriptor(output_mem, scatter_output_ptr, 1, IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(input_mem, scatter_input_ptr, 1, IDNNL_MEM_GM);

  // scatter_stream1
  idnnlMemoryDescriptor_t indices_mem_s1;
  idnnlMemoryDescriptor_t update_mem_s1;
  idnnlMemoryDescriptor_t output_mem_s1;
  idnnlMemoryDescriptor_t input_mem_s1;
  idnnlCreateMemoryDescriptor(&indices_mem_s1);
  idnnlCreateMemoryDescriptor(&update_mem_s1);
  idnnlCreateMemoryDescriptor(&output_mem_s1);
  idnnlCreateMemoryDescriptor(&input_mem_s1);
  void* indices_mem_ptr_s1[] = {coors_device_s1};
  void* update_mem_ptr_s1[] = {pfe1_output_addr_s1};
  idnnlSetMemoryDescriptor(indices_mem_s1, indices_mem_ptr_s1, 1, IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(update_mem_s1, update_mem_ptr_s1, 1, IDNNL_MEM_GM);
  void* scatter_output_ptr_s1[] = {scatter_out_ddr_s1};
  void* scatter_input_ptr_s1[] = {scatter_in_ddr_s1};
  idnnlSetMemoryDescriptor(output_mem_s1, scatter_output_ptr_s1, 1,
                           IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(input_mem_s1, scatter_input_ptr_s1, 1, IDNNL_MEM_GM);

  // scatter_stream2
  idnnlMemoryDescriptor_t indices_mem_s2;
  idnnlMemoryDescriptor_t update_mem_s2;
  idnnlMemoryDescriptor_t output_mem_s2;
  idnnlMemoryDescriptor_t input_mem_s2;
  idnnlCreateMemoryDescriptor(&indices_mem_s2);
  idnnlCreateMemoryDescriptor(&update_mem_s2);
  idnnlCreateMemoryDescriptor(&output_mem_s2);
  idnnlCreateMemoryDescriptor(&input_mem_s2);
  void* indices_mem_ptr_s2[] = {coors_device_s2};
  void* update_mem_ptr_s2[] = {pfe1_output_addr_s2};
  idnnlSetMemoryDescriptor(indices_mem_s2, indices_mem_ptr_s2, 1, IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(update_mem_s2, update_mem_ptr_s2, 1, IDNNL_MEM_GM);
  void* scatter_output_ptr_s2[] = {scatter_out_ddr_s2};
  void* scatter_input_ptr_s2[] = {scatter_in_ddr_s2};
  idnnlSetMemoryDescriptor(output_mem_s2, scatter_output_ptr_s2, 1,
                           IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(input_mem_s2, scatter_input_ptr_s2, 1, IDNNL_MEM_GM);

  // scatter_stream3
  idnnlMemoryDescriptor_t indices_mem_s3;
  idnnlMemoryDescriptor_t update_mem_s3;
  idnnlMemoryDescriptor_t output_mem_s3;
  idnnlMemoryDescriptor_t input_mem_s3;
  idnnlCreateMemoryDescriptor(&indices_mem_s3);
  idnnlCreateMemoryDescriptor(&update_mem_s3);
  idnnlCreateMemoryDescriptor(&output_mem_s3);
  idnnlCreateMemoryDescriptor(&input_mem_s3);
  void* indices_mem_ptr_s3[] = {coors_device_s3};
  void* update_mem_ptr_s3[] = {pfe1_output_addr_s3};
  idnnlSetMemoryDescriptor(indices_mem_s3, indices_mem_ptr_s3, 1, IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(update_mem_s3, update_mem_ptr_s3, 1, IDNNL_MEM_GM);
  void* scatter_output_ptr_s3[] = {scatter_out_ddr_s3};
  void* scatter_input_ptr_s3[] = {scatter_in_ddr_s3};
  idnnlSetMemoryDescriptor(output_mem_s3, scatter_output_ptr_s3, 1,
                           IDNNL_MEM_GM);
  idnnlSetMemoryDescriptor(input_mem_s3, scatter_input_ptr_s3, 1, IDNNL_MEM_GM);

  // input_220 = nox->pfe_out_1_220;
  // input_to8 = nox->pfe_out_2_to8;
  idnnlTensorDescriptor_t indices_desc;
  idnnlTensorDescriptor_t update_desc;
  idnnlTensorDescriptor_t output_desc;
  idnnlCreateTensorDescriptor(&indices_desc);
  idnnlCreateTensorDescriptor(&update_desc);
  idnnlCreateTensorDescriptor(&output_desc);
  // loop_time need check
  idnnlSetTensor4dDescriptor(indices_desc, IDNNL_TENSOR_NCHW, IDNNL_DATA_INT16,
                             1, 1, 12032,
                             4);  // need to check
  idnnlSetTensor4dDescriptor(update_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                             1, 12032,
                             64);  // need to check
  idnnlSetTensor4dDescriptor(output_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8, 1,
                             h, w, 64);  // need to check

  int thread_num = 4;
  uint32_t* pillar_point_index_ddr =
      reinterpret_cast<uint32_t*>(workspace_gm_addr) + thread_num * h * w;
  uint32_t* active_pillar_point_count_ddr =
      pillar_point_index_ddr + thread_num * h * w * kMaxNumPointsPerPillar;
  uint32_t* active_pillar_point_index_ddr =
      active_pillar_point_count_ddr + h * w * kMaxNumPointsPerPillar;
  uint32_t* x_ddr = active_pillar_point_index_ddr + h * w;
  uint32_t* active_pillar_addr = x_ddr + kMaxNumPillars;
  // idnnlCreateWorkspaceDescriptor(&workspace);
  int left_pillar_count = 9226;

  // auto pfe_start = std::chrono::system_clock::now();
  std::cout << "start run point_pillars" << std::endl;
  auto start = std::chrono::system_clock::now();
#if RUN_VOXEL
  for (int i = 0; i < test_count; i++) {
    auto ret = idnnlRvvVoxelization(
        handle0, point_desc, point_ddr, voxel_desc, voxel_ddr, coors_desc,
        coors_ddr, num_per_pillar_desc, num_per_pillar_ddr, workspace,
        gridXSize, gridYSize, gridZSize, kMinXRange, kMinYRange, kMinZRange,
        kPillarXSize, kPillarYSize, kPillarZSize, quant_scale);
#if VOXEL_S1
    idnnlRvvVoxelization(
        handle1, point_desc, point_ddr, voxel_desc, voxel_ddr_s1, coors_desc,
        coors_ddr_s1, num_per_pillar_desc, num_per_pillar_ddr_s1, workspace_s1,
        gridXSize, gridYSize, gridZSize, kMinXRange, kMinYRange, kMinZRange,
        kPillarXSize, kPillarYSize, kPillarZSize, quant_scale);
    idnnlRvvVoxelization(
        handle2, point_desc, point_ddr, voxel_desc, voxel_ddr_s2, coors_desc,
        coors_ddr_s2, num_per_pillar_desc, num_per_pillar_ddr_s2, workspace_s2,
        gridXSize, gridYSize, gridZSize, kMinXRange, kMinYRange, kMinZRange,
        kPillarXSize, kPillarYSize, kPillarZSize, quant_scale);
    idnnlRvvVoxelization(
        handle3, point_desc, point_ddr, voxel_desc, voxel_ddr_s3, coors_desc,
        coors_ddr_s3, num_per_pillar_desc, num_per_pillar_ddr_s3, workspace_s3,
        gridXSize, gridYSize, gridZSize, kMinXRange, kMinYRange, kMinZRange,
        kPillarXSize, kPillarYSize, kPillarZSize, quant_scale);
#endif
#endif
#if RUN_PFE
    pfe1_aot_module.Run();
#if PFE_S1
    pfe1_aot_module_s1.Run();
    pfe1_aot_module_s2.Run();
    pfe1_aot_module_s3.Run();
#endif
#endif
#if RUN_SCATTER
    idnnlMemoryDescriptor_t active_pillar_mem;
    idnnlCreateMemoryDescriptor(&active_pillar_mem);
    void* active_pillar_ptr[] = {active_pillar_addr};
    idnnlSetMemoryDescriptor(active_pillar_mem, active_pillar_ptr, 1,
                             IDNNL_MEM_GM);
    idnnlScatterTranspose(handle0, indices_desc, indices_mem, active_pillar_mem,
                          left_pillar_count, update_desc, update_mem,
                          output_desc, input_mem, output_mem, workspace);
#if SCATTER_S1
    idnnlScatterTranspose(handle1, indices_desc, indices_mem_s1,
                          active_pillar_mem, left_pillar_count, update_desc,
                          update_mem_s1, output_desc, input_mem_s1,
                          output_mem_s1, workspace_s1);
    idnnlScatterTranspose(handle2, indices_desc, indices_mem_s2,
                          active_pillar_mem, left_pillar_count, update_desc,
                          update_mem_s2, output_desc, input_mem_s2,
                          output_mem_s2, workspace_s2);
    idnnlScatterTranspose(handle3, indices_desc, indices_mem_s3,
                          active_pillar_mem, left_pillar_count, update_desc,
                          update_mem_s3, output_desc, input_mem_s3,
                          output_mem_s3, workspace_s3);
#endif
#endif
#if RUN_RPN
    tvm::runtime::NDArray::Container* rpn_in =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(scatter_out_ddr), {1, h, w, 64},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dev_rpn_in =
        tvm::runtime::NDArray(tvm::runtime::GetObjectPtr<tvm::Object>(rpn_in));
    // rpn output
    tvm::runtime::NDArray::Container* dir_cls_preds_in =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(dir_cls_preds_ddr), {1, 116, 116, 12},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* bbox_preds_in =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(bbox_preds_ddr), {1, 116, 116, 42},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* cls_scores_in =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(cls_scores_ddr), {1, 116, 116, 18},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dir_cls_preds_dev_in = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(dir_cls_preds_in));
    auto bbox_preds_dev_in = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(bbox_preds_in));
    auto cls_scores_dev_in = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(cls_scores_in));
    rpn_stream0.SetInput("features", dev_rpn_in);
    rpn_stream0.SetOutput(rpn_stream0.GetOutputNameByIndex(1),
                          bbox_preds_dev_in);
    rpn_stream0.SetOutput(rpn_stream0.GetOutputNameByIndex(2),
                          dir_cls_preds_dev_in);
    rpn_stream0.SetOutput(rpn_stream0.GetOutputNameByIndex(0),
                          cls_scores_dev_in);

    // s1
    tvm::runtime::NDArray::Container* rpn_in_s1 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(scatter_out_ddr_s1), {1, h, w, 64},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dev_rpn_in_s1 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(rpn_in_s1));
    // rpn output
    tvm::runtime::NDArray::Container* dir_cls_preds_in_s1 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(dir_cls_preds_ddr_s1), {1, 116, 116, 12},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* bbox_preds_in_s1 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(bbox_preds_ddr_s1), {1, 116, 116, 42},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* cls_scores_in_s1 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(cls_scores_ddr_s1), {1, 116, 116, 18},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dir_cls_preds_dev_in_s1 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(dir_cls_preds_in_s1));
    auto bbox_preds_dev_in_s1 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(bbox_preds_in_s1));
    auto cls_scores_dev_in_s1 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(cls_scores_in_s1));
    rpn_stream1.SetInput("features", dev_rpn_in_s1);
    rpn_stream1.SetOutput(rpn_stream1.GetOutputNameByIndex(1),
                          bbox_preds_dev_in_s1);
    rpn_stream1.SetOutput(rpn_stream1.GetOutputNameByIndex(2),
                          dir_cls_preds_dev_in_s1);
    rpn_stream1.SetOutput(rpn_stream1.GetOutputNameByIndex(0),
                          cls_scores_dev_in_s1);

    // rpn s2
    tvm::runtime::NDArray::Container* rpn_in_s2 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(scatter_out_ddr_s2), {1, h, w, 64},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dev_rpn_in_s2 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(rpn_in_s2));
    // rpn output
    tvm::runtime::NDArray::Container* dir_cls_preds_in_s2 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(dir_cls_preds_ddr_s2), {1, 116, 116, 12},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* bbox_preds_in_s2 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(bbox_preds_ddr_s2), {1, 116, 116, 42},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* cls_scores_in_s2 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(cls_scores_ddr_s2), {1, 116, 116, 18},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dir_cls_preds_dev_in_s2 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(dir_cls_preds_in_s2));
    auto bbox_preds_dev_in_s2 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(bbox_preds_in_s2));
    auto cls_scores_dev_in_s2 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(cls_scores_in_s2));
    rpn_stream2.SetInput("features", dev_rpn_in_s2);
    rpn_stream2.SetOutput(rpn_stream2.GetOutputNameByIndex(1),
                          bbox_preds_dev_in_s2);
    rpn_stream2.SetOutput(rpn_stream2.GetOutputNameByIndex(2),
                          dir_cls_preds_dev_in_s2);
    rpn_stream2.SetOutput(rpn_stream2.GetOutputNameByIndex(0),
                          cls_scores_dev_in_s2);

    // rpn s3
    tvm::runtime::NDArray::Container* rpn_in_s3 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(scatter_out_ddr_s3), {1, h, w, 64},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dev_rpn_in_s3 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(rpn_in_s3));
    // rpn output
    tvm::runtime::NDArray::Container* dir_cls_preds_in_s3 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(dir_cls_preds_ddr_s3), {1, 116, 116, 12},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* bbox_preds_in_s3 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(bbox_preds_ddr_s3), {1, 116, 116, 42},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    tvm::runtime::NDArray::Container* cls_scores_in_s3 =
        new tvm::runtime::NDArray::Container(
            static_cast<void*>(cls_scores_ddr_s3), {1, 116, 116, 18},
            tvm::DataType::Int(8), {kDLHDPL, 0});
    auto dir_cls_preds_dev_in_s3 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(dir_cls_preds_in_s3));
    auto bbox_preds_dev_in_s3 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(bbox_preds_in_s3));
    auto cls_scores_dev_in_s3 = tvm::runtime::NDArray(
        tvm::runtime::GetObjectPtr<tvm::Object>(cls_scores_in_s3));
    rpn_stream3.SetInput("features", dev_rpn_in_s3);
    rpn_stream3.SetOutput(rpn_stream3.GetOutputNameByIndex(1),
                          bbox_preds_dev_in_s3);
    rpn_stream3.SetOutput(rpn_stream3.GetOutputNameByIndex(2),
                          dir_cls_preds_dev_in_s3);
    rpn_stream3.SetOutput(rpn_stream3.GetOutputNameByIndex(0),
                          cls_scores_dev_in_s3);
    rpn_stream0.SetStream(stream0_);
    rpn_stream0.Run();
    rpn_stream1.SetStream(stream1_);
    rpn_stream2.SetStream(stream2_);
    rpn_stream3.SetStream(stream3_);
#ifdef RPN_S1
    rpn_stream1.Run();
    rpn_stream2.Run();
    rpn_stream3.Run();
#endif
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
    idnnlMemoryDescriptor_t bbox_ddr_s1;
    idnnlCreateMemoryDescriptor(&bbox_ddr_s1);
    void* bbox_ptr_s1[] = {bbox_preds_ddr_s1};
    idnnlSetMemoryDescriptor(bbox_ddr_s1, bbox_ptr_s1, 1, IDNNL_MEM_GM);
    idnnlMemoryDescriptor_t bbox_ddr_s2;
    idnnlCreateMemoryDescriptor(&bbox_ddr_s2);
    void* bbox_ptr_s2[] = {bbox_preds_ddr_s2};
    idnnlSetMemoryDescriptor(bbox_ddr_s2, bbox_ptr_s2, 1, IDNNL_MEM_GM);
    idnnlMemoryDescriptor_t bbox_ddr_s3;
    idnnlCreateMemoryDescriptor(&bbox_ddr_s3);
    void* bbox_ptr_s3[] = {bbox_preds_ddr_s3};
    idnnlSetMemoryDescriptor(bbox_ddr_s3, bbox_ptr_s3, 1, IDNNL_MEM_GM);
    idnnlTensorDescriptor_t scores_desc;
    idnnlCreateTensorDescriptor(&scores_desc);
    idnnlSetTensor4dDescriptor(scores_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8,
                               1, 116, 116, 18);
    idnnlMemoryDescriptor_t scores_ddr;
    idnnlCreateMemoryDescriptor(&scores_ddr);
    void* scores_ptr[] = {cls_scores_ddr};
    idnnlSetMemoryDescriptor(scores_ddr, scores_ptr, 1, IDNNL_MEM_GM);

    idnnlMemoryDescriptor_t scores_ddr_s1;
    idnnlCreateMemoryDescriptor(&scores_ddr_s1);
    void* scores_ptr_s1[] = {cls_scores_ddr_s1};
    idnnlSetMemoryDescriptor(scores_ddr_s1, scores_ptr_s1, 1, IDNNL_MEM_GM);

    idnnlMemoryDescriptor_t scores_ddr_s2;
    idnnlCreateMemoryDescriptor(&scores_ddr_s2);
    void* scores_ptr_s2[] = {cls_scores_ddr_s2};
    idnnlSetMemoryDescriptor(scores_ddr_s2, scores_ptr_s2, 1, IDNNL_MEM_GM);

    idnnlMemoryDescriptor_t scores_ddr_s3;
    idnnlCreateMemoryDescriptor(&scores_ddr_s3);
    void* scores_ptr_s3[] = {cls_scores_ddr_s3};
    idnnlSetMemoryDescriptor(scores_ddr_s3, scores_ptr_s3, 1, IDNNL_MEM_GM);

    idnnlTensorDescriptor_t preds_desc;
    idnnlCreateTensorDescriptor(&preds_desc);
    idnnlSetTensor4dDescriptor(preds_desc, IDNNL_TENSOR_NHWC, IDNNL_DATA_INT8,
                               1, 116, 116, 12);
    idnnlMemoryDescriptor_t preds_mem;
    idnnlCreateMemoryDescriptor(&preds_mem);
    void* preds_desc_ptr[] = {dir_cls_preds_ddr};
    idnnlSetMemoryDescriptor(preds_mem, preds_desc_ptr, 1, IDNNL_MEM_GM);

    idnnlMemoryDescriptor_t preds_mem_s1;
    idnnlCreateMemoryDescriptor(&preds_mem_s1);
    void* preds_desc_ptr_s1[] = {dir_cls_preds_ddr_s1};
    idnnlSetMemoryDescriptor(preds_mem_s1, preds_desc_ptr_s1, 1, IDNNL_MEM_GM);
    idnnlMemoryDescriptor_t preds_mem_s2;
    idnnlCreateMemoryDescriptor(&preds_mem_s2);
    void* preds_desc_ptr_s2[] = {dir_cls_preds_ddr_s2};
    idnnlSetMemoryDescriptor(preds_mem_s2, preds_desc_ptr_s2, 1, IDNNL_MEM_GM);
    idnnlMemoryDescriptor_t preds_mem_s3;
    idnnlCreateMemoryDescriptor(&preds_mem_s3);
    void* preds_desc_ptr_s3[] = {dir_cls_preds_ddr_s3};
    idnnlSetMemoryDescriptor(preds_mem_s3, preds_desc_ptr_s3, 1, IDNNL_MEM_GM);
    // printf("start run detection\n");
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
#if POST_S1
    idnnlRvvPillarsDetection(
        handle1, bbox_desc, bbox_ddr_s1, scores_desc, scores_ddr_s1, preds_desc,
        preds_mem_s1, anchors_px_desc, anchors_px_ddr, anchors_py_desc,
        anchors_py_ddr, anchors_pz_desc, anchors_pz_ddr, anchors_dx_desc,
        anchors_dx_ddr, anchors_dy_desc, anchors_dy_ddr, anchors_dz_desc,
        anchors_dz_ddr, anchors_ro_desc, anchors_ro_ddr, anchors_diagonal_desc,
        anchors_diagonal_ddr, out_box_desc, out_box_ddr_s1, out_box_num_desc,
        out_box_num_ddr_s1, post_workspace_s1, bbox_scalar_desc,
        bbox_scalar_ddr, scores_scalar_desc, scores_scalar_ddr,
        preds_scalar_desc, preds_scalar_ddr, kNumClass, kNumAnchor,
        kNumOutputBoxFeature, kNumBoxCorners, score_threshold,
        nms_overlap_threshold);
    idnnlRvvPillarsDetection(
        handle2, bbox_desc, bbox_ddr_s2, scores_desc, scores_ddr_s2, preds_desc,
        preds_mem_s2, anchors_px_desc, anchors_px_ddr, anchors_py_desc,
        anchors_py_ddr, anchors_pz_desc, anchors_pz_ddr, anchors_dx_desc,
        anchors_dx_ddr, anchors_dy_desc, anchors_dy_ddr, anchors_dz_desc,
        anchors_dz_ddr, anchors_ro_desc, anchors_ro_ddr, anchors_diagonal_desc,
        anchors_diagonal_ddr, out_box_desc, out_box_ddr_s2, out_box_num_desc,
        out_box_num_ddr_s2, post_workspace_s2, bbox_scalar_desc,
        bbox_scalar_ddr, scores_scalar_desc, scores_scalar_ddr,
        preds_scalar_desc, preds_scalar_ddr, kNumClass, kNumAnchor,
        kNumOutputBoxFeature, kNumBoxCorners, score_threshold,
        nms_overlap_threshold);
    idnnlRvvPillarsDetection(
        handle3, bbox_desc, bbox_ddr_s3, scores_desc, scores_ddr_s3, preds_desc,
        preds_mem_s3, anchors_px_desc, anchors_px_ddr, anchors_py_desc,
        anchors_py_ddr, anchors_pz_desc, anchors_pz_ddr, anchors_dx_desc,
        anchors_dx_ddr, anchors_dy_desc, anchors_dy_ddr, anchors_dz_desc,
        anchors_dz_ddr, anchors_ro_desc, anchors_ro_ddr, anchors_diagonal_desc,
        anchors_diagonal_ddr, out_box_desc, out_box_ddr_s3, out_box_num_desc,
        out_box_num_ddr_s3, post_workspace_s3, bbox_scalar_desc,
        bbox_scalar_ddr, scores_scalar_desc, scores_scalar_ddr,
        preds_scalar_desc, preds_scalar_ddr, kNumClass, kNumAnchor,
        kNumOutputBoxFeature, kNumBoxCorners, score_threshold,
        nms_overlap_threshold);
#endif
  }
#endif
  hdplDeviceSynchronize();
  auto finish = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  auto total_time = (duration.count());
  printf(
      "point_pillars test count %zu cost %fms each time stream_num %d fps %d "
      "\n",
      test_count, total_time / 1000.0 / test_count, 4,
      1000 / (total_time / 1000 / test_count) * 4);
  std::cout << "end run point_pillars" << std::endl;
  int output_num[1] = {0};
  int output_num_s1[1] = {0};
  int output_num_s2[1] = {0};
  int output_num_s3[1] = {0};
  hdplMemcpy(output_num, out_box_num_device, 4, hdplMemcpyDeviceToHost);
  hdplMemcpy(output_num_s1, out_box_num_device_s1, 4, hdplMemcpyDeviceToHost);
  hdplMemcpy(output_num_s2, out_box_num_device_s2, 4, hdplMemcpyDeviceToHost);
  hdplMemcpy(output_num_s3, out_box_num_device_s3, 4, hdplMemcpyDeviceToHost);
  printf("detection stream0 output_num %d out_box_num_device %p\n",
         output_num[0], out_box_num_device);
  printf("detection stream1 output_num %d out_box_num_device %p\n",
         output_num_s1[0], out_box_num_device_s1);
  printf("detection stream2 output_num %d out_box_num_device %p\n",
         output_num_s2[0], out_box_num_device_s2);
  printf("detection stream3 output_num %d out_box_num_device %p\n",
         output_num_s3[0], out_box_num_device_s3);
  float rvv_out_detections[1024 * 9] = {0};
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
  hdplFree(point_device);
  hdplFree(voxel_device);
  hdplFree(voxel_device_s1);
  hdplFree(voxel_device_s2);
  hdplFree(voxel_device_s3);
  hdplFree(coors_device_s0);
  hdplFree(coors_device_s1);
  hdplFree(coors_device_s2);
  hdplFree(coors_device_s3);
  hdplFree(num_per_pillar_device);
  hdplFree(num_per_pillar_device_s1);
  hdplFree(num_per_pillar_device_s2);
  hdplFree(num_per_pillar_device_s3);
  hdplReservedMemFree(workspace_gm_addr);
  hdplReservedMemFree(workspace_gm_addr_s1);
  hdplReservedMemFree(workspace_gm_addr_s2);
  hdplReservedMemFree(workspace_gm_addr_s3);
  hdplFree(pfe1_output_addr);
  hdplFree(pfe1_output_addr_s1);
  hdplFree(pfe1_output_addr_s2);
  hdplFree(pfe1_output_addr_s3);
  hdplFree(scatter_out_ddr);
  hdplFree(scatter_out_ddr_s1);
  hdplFree(scatter_out_ddr_s2);
  hdplFree(scatter_out_ddr_s3);
  hdplFree(scatter_in_ddr);
  hdplFree(scatter_in_ddr_s1);
  hdplFree(scatter_in_ddr_s2);
  hdplFree(scatter_in_ddr_s3);
  hdplFree(bbox_preds_ddr);
  hdplFree(bbox_preds_ddr_s1);
  hdplFree(bbox_preds_ddr_s2);
  hdplFree(bbox_preds_ddr_s3);
  hdplFree(cls_scores_ddr);
  hdplFree(cls_scores_ddr_s1);
  hdplFree(cls_scores_ddr_s2);
  hdplFree(cls_scores_ddr_s3);
  hdplFree(dir_cls_preds_ddr);
  hdplFree(dir_cls_preds_ddr_s1);
  hdplFree(dir_cls_preds_ddr_s2);
  hdplFree(dir_cls_preds_ddr_s3);
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
  hdplStreamDestroy(stream0_);
  hdplStreamDestroy(stream1_);
  hdplStreamDestroy(stream2_);
  hdplStreamDestroy(stream3_);
  idnnlDestroy(handle0);
  idnnlDestroy(handle1);
  idnnlDestroy(handle2);
  idnnlDestroy(handle3);
}

int main(int argc, char* argv[]) {
  int point_num = 54908;
  size_t test_count = 1;
  if (argc == 2) {
    test_count = atoi(argv[1]);
  }
  runVoxelization(point_num, test_count);
  return 0;
}
