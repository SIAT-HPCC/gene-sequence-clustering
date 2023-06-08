#include "func.h"  // 数据结构与函数
#include "timer.h"  // 计时器

int main(int argc, char **argv) {
  Timer timer; timer.start();  // 计时
  Option option;  // 输入参数
  init(argc, argv, option);  // 读输入 初始化显卡
  Data data;
  readData(option, data);  // 读数据
  clustering(option, data);  // 聚类
  saveResult(option, data);  // 保存结果
  timer.getDuration();  // 耗时
}
