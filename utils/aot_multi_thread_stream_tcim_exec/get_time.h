// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

#ifndef TIME_H_
#define TIME_H_

#include <chrono>

using namespace std::chrono;

class GetTime {
  public:
    GetTime() {};
    ~GetTime() {};

    static high_resolution_clock::time_point Now() {
        return high_resolution_clock::now();
    }

    static double DurationS(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
        duration<double,std::ratio<1,1>> duration_s(t2 - t1);
        return duration_s.count();
    }

    static double DurationMs(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
        duration<double,std::ratio<1,1000>> duration_ms(t2 - t1);
        return duration_ms.count();
    }

    static double DurationUs(high_resolution_clock::time_point t1, high_resolution_clock::time_point t2) {
        duration<double,std::ratio<1,1000000>> duration_us(t2 - t1);
        return duration_us.count();
    }

};


#endif