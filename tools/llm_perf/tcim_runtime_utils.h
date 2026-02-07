
#ifndef __TCIM_RUNTIME_UTILS_H__
#define __TCIM_RUNTIME_UTILS_H__

#include "tcim/tcim_runtime.h"
inline void CheckTcimRetStatus(const tcim::Status &status) {
  if (status != tcim::Status::OK) {
    throw std::runtime_error(
        "tcim_runtime ret Status is not OK, current ret Status is " +
        std::to_string(status));
  }

  return;
}
#endif