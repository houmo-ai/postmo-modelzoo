#pragma once
#include "tcim/tcim_runtime.h"
#include <cassert>
#include <cstdint>

// 合并后的转换函数
static void BGR2YUVSP(const uint8_t *src, uint8_t *dst, int height, int width, tcim::DataFmt format) {
    assert(src != nullptr && dst != nullptr);
    assert(height > 0 && width > 0);

    // 根据格式验证尺寸要求
    if (format == tcim::DataFmt::YUV420SP) {
        assert((height % 2) == 0 && (width % 2) == 0);  // 420需双数尺寸
    } else if (format == tcim::DataFmt::YUV420SP) {
        assert((width % 2) == 0);  // 422需宽度为双数
    }

    const int64_t total_pixels = static_cast<int64_t>(width) * height;
    const uint8_t *bgr = src;
    uint8_t *y_plane = dst;
    uint8_t *uv_plane = dst + total_pixels;  // UV平面起始位置

    // 公共的Y分量计算（所有格式相同）
    for (int h = 0; h < height; ++h) {
        const uint8_t *row = bgr + static_cast<int64_t>(h) * width * 3;
        uint8_t *y_row = y_plane + static_cast<int64_t>(h) * width;
        for (int w = 0; w < width; ++w) {
            int idx = w * 3;
            int b = row[idx + 0];
            int g = row[idx + 1];
            int r = row[idx + 2];
            int y = (76 * r + 150 * g + 29 * b) >> 8;  // BT.601标准
            y_row[w] = static_cast<uint8_t>(y < 0 ? 0 : (y > 255 ? 255 : y));
        }
    }

    // 根据格式处理UV分量
    switch (format) {
    case tcim::DataFmt::YUV420SP: {     // NV12格式（4:2:0采样）
        const int rounding = (1 << 9);  // 舍入位
        for (int h = 0; h < height; h += 2) {
            const uint8_t *row0 = bgr + static_cast<int64_t>(h) * width * 3;
            const uint8_t *row1 = bgr + static_cast<int64_t>(h + 1) * width * 3;
            uint8_t *uv_row = uv_plane + static_cast<int64_t>(h / 2) * width;

            for (int w = 0; w < width; w += 2) {
                // 读取2x2像素块
                int idx = w * 3;
                int r00 = row0[idx + 2], g00 = row0[idx + 1], b00 = row0[idx + 0];
                int r01 = row0[idx + 5], g01 = row0[idx + 4], b01 = row0[idx + 3];
                int r10 = row1[idx + 2], g10 = row1[idx + 1], b10 = row1[idx + 0];
                int r11 = row1[idx + 5], g11 = row1[idx + 4], b11 = row1[idx + 3];

                // 计算RGB分量平均值
                int64_t sumR = r00 + r01 + r10 + r11;
                int64_t sumG = g00 + g01 + g10 + g11;
                int64_t sumB = b00 + b01 + b10 + b11;

                // 计算UV分量
                int u = ((-38 * sumR - 74 * sumG + 112 * sumB + rounding) >> 10) + 128;
                int v = ((112 * sumR - 94 * sumG - 18 * sumB + rounding) >> 10) + 128;

                // 存储交错UV
                uv_row[w] = static_cast<uint8_t>(u < 0 ? 0 : (u > 255 ? 255 : u));
                uv_row[w + 1] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
        break;
    }

    case tcim::DataFmt::YUV422SP: {     // NV16格式（4:2:2采样）
        const int rounding = (1 << 8);  // 舍入位
        for (int h = 0; h < height; ++h) {
            const uint8_t *row = bgr + static_cast<int64_t>(h) * width * 3;
            uint8_t *uv_row = uv_plane + static_cast<int64_t>(h) * width;

            for (int w = 0; w < width; w += 2) {
                int idx0 = w * 3;
                int idx1 = (w + 1) * 3;

                // 读取水平相邻的两个像素
                int r0 = row[idx0 + 2], g0 = row[idx0 + 1], b0 = row[idx0 + 0];
                int r1 = row[idx1 + 2], g1 = row[idx1 + 1], b1 = row[idx1 + 0];

                // 水平平均RGB分量
                int avgR = (r0 + r1 + 1) >> 1;
                int avgG = (g0 + g1 + 1) >> 1;
                int avgB = (b0 + b1 + 1) >> 1;

                // 计算UV分量
                int u = ((-38 * avgR - 74 * avgG + 112 * avgB + rounding) >> 9) + 128;
                int v = ((112 * avgR - 94 * avgG - 18 * avgB + rounding) >> 9) + 128;

                // 存储交错UV
                uv_row[w] = static_cast<uint8_t>(u < 0 ? 0 : (u > 255 ? 255 : u));
                uv_row[w + 1] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
        break;
    }

    case tcim::DataFmt::YUV444SP: {         // YUV444格式（无下采样）
        const int round_offset = (1 << 7);  // 舍入位
        for (int h = 0; h < height; ++h) {
            const uint8_t *row = bgr + static_cast<int64_t>(h) * width * 3;
            uint8_t *uv_row = uv_plane + static_cast<int64_t>(h) * width * 2;

            for (int w = 0; w < width; ++w) {
                int idx = w * 3;
                int b = row[idx + 0];
                int g = row[idx + 1];
                int r = row[idx + 2];

                // 计算UV分量（每个像素独立计算）
                int u = ((-38 * r - 74 * g + 112 * b + round_offset) >> 8) + 128;
                int v = ((112 * r - 94 * g - 18 * b + round_offset) >> 8) + 128;

                // 存储交错UV
                uv_row[2 * w] = static_cast<uint8_t>(u < 0 ? 0 : (u > 255 ? 255 : u));
                uv_row[2 * w + 1] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
        break;
    }
    }
}
