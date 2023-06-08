#ifndef __FUNCH__
#define __FUNCH__

#include <string>  // std::string
#include <vector>  // vector

//--------数据结构--------//
struct Option {  // 输入参数
  std::string packedFile;  // packed文件
  std::string resultFile;  // result文件
  float similarity;  // 相似度
  int32_t mode;  // 运行模式 0-fast 1-precise
  int32_t rank;  // 线程编号
  int32_t size;  // 线程数
};

struct Data {  // 读入数据
  int32_t readsCount;  // 序列数
  int32_t longest;  // 最长序列
  int32_t shortest;  // 最短序列
  int32_t type;  // 序列种类 0:gene 1:protein
  std::vector<int32_t> nameLengths;  // 序列名长度
  std::vector<int32_t> readLengths;  // 序列长度
  std::vector<std::string> names;  // 序列名
  uint32_t *bufferH;  // 序列数据H
  uint32_t *bufferD;  // 序列数据D
  uint32_t **readsH;  // 序列索引H
  uint32_t **readsD;  // 序列索引D
  std::vector<int32_t> result;  // 聚类结果
};

//--------声明函数--------//
void init(int argc, char **argv, Option &option);  // 读输入
void readData(const Option &option, Data &data);  // 读数据
void clustering(const Option &option, Data &data);  // 聚类
void saveResult(Option &option, Data &data);  // 保存结果
#endif  // __FUNCH__
