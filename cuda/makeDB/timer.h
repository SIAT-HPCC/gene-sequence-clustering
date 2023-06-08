/*
  timer.h
  chrono库实现的计时器 支持时间戳 中断计时
  用法如下
  TImer timer;  // 声明
  timer.getTimeNow();  // 打时间戳
  timer.start();  // 开始计时
  timer.pause();  // 暂停计时
  timer.resume();  // 恢复计时
  timer.getDuration();  // 统计耗时
  2022/02/13 by 鞠震
*/

#include <iostream>  // cout
#include <chrono>  // system_clock
#include <ctime>  // time_t
#ifndef __TIMERH__  // 防御声明 pragma once 通用性有问题
#define __TIMERH__
//--------类声明--------//
class Timer {  // 计时器
  private:
    std::chrono::system_clock::time_point now;  // 当前时间
    std::time_t time_now;  // 时间戳的格式化版
    std::chrono::steady_clock::time_point t1;  // 计时开始时刻
    std::chrono::steady_clock::time_point t2;  // 计时结束时刻
    std::chrono::duration<double> duration;  // 耗时
    int p;  // 是否暂停了
  public:
    void getTimeNow() {  // 输出当前时间戳
      now = std::chrono::system_clock::now();
      time_now = std::chrono::system_clock::to_time_t(now);
      std::cout << ctime(&time_now);
    }
    void start() {  // 开始计时
      t1 = std::chrono::steady_clock::now();
      duration = std::chrono::duration<double>(0);
      p = 0;  // 不暂停
    }
    void pause() {  // 暂停计时
      t2 = std::chrono::steady_clock::now();
      duration += std::chrono::duration_cast
        <std::chrono::duration<double>>(t2-t1);
      p = 1;
    }
    void resume() {  // 恢复计时
      t1 = std::chrono::steady_clock::now();
      p = 0;
    }
    void getDuration() {  // 输出耗时
      if (p == 0) pause();  // 没暂停需要先暂停
      std::cout << duration.count() << " seconds.\n";
    }
};
#endif  // __TIMERH__
