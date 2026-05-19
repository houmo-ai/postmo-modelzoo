/*
 * Copyright (c) 2026 HOUMO AI
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * File: miniaudio_impl.cpp
 * Description: miniaudio library implementation entry point
 *
 * MINIAUDIO_IMPLEMENTATION must be defined in exactly one translation unit.
 */

#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"