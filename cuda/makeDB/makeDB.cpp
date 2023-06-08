/*
makeDB.cpp
data.packed:
  int 序列数
  int 序列最长
  int 序列最短
  int 序列类型 0:gene 1:protein
  vector<int> 序列名长度
  vector<int> 序列长度
  vector<long> 数据偏移
  压缩数据
    string 文件名
    unsigned int 长度
    unsigned int 净长度
    unsigned int 压缩数据
用法:
makeDB -f fasta文件 -p packed文件 -t (0:基因 1:蛋白序)
2023-04-07 by 鞠震
*/
// 为支持TB大小数据集 不能把数据都读入内存 要先生成索引

#include <iostream>  // cout
#include <fstream>  // fstream
#include <string>  // string
#include <vector>  // vector
#include <algorithm>  // sort
#include <climits>  // INT_MAX
#include <unordered_map>  // unordered_map
#include <omp.h>  // openmp
#include "timer.h"  // 计时器
#include "cmdline.h"  // 解析器

//--------数据--------//
struct Option {  // 输入选项
  std::string fastaFile;  // 输入文件
  std::string packedFile;  // 输出路径
  int type;  // 数据类型 0基因 1蛋白
};
struct Read {  // 记录一条序列的位置 为了排序
  int nameLength;  // 序列名长
  int readLength;  // 序列长 最长65536
  long offset;  // 序列的起始位置
};

//--------函数--------//
// parse 解析输入选项
void parse(int argc, char **argv, Option &option) {
  cmdline::parser parser;  // 解析器
  parser.add<std::string>("fasta", 'f', "fasta file", true, "");  // 输入
  parser.add<std::string>("packed", 'p', "packed file", true, "");  // 输出
  parser.add<int>("type", 't', "fasta type 0:gene 1:protein", true, 0,
    cmdline::range<int>(0, 1));  // 数据类型
  parser.parse_check(argc, argv);
  option.fastaFile = parser.get<std::string>("fasta");
  option.packedFile = parser.get<std::string>("packed");
  option.type = parser.get<int>("type");
  std::cout << "fasta:\t\t" << option.fastaFile << "\n";
  std::cout << "packed:\t\t" << option.packedFile << "\n";
  std::cout << "type:\t\t" << option.type << " (0:gene 1:protein)\n";
}

// makeIndex 生成文件索引
void makeIndex(const Option &option, std::vector<Read> &reads) {
  std::ifstream fastaFile(option.fastaFile);  // 打开输入
  std::string line;  // 读入的一行
  Read read;  // 一条数据
  long progress = 0;  // 当前进度
  while(fastaFile.peek() != EOF) {  // 读到文件结束
    read.offset = fastaFile.tellg();  // 序列起始位置
    getline(fastaFile, line);  // 读序列名
    if (line.back() == '\r') line.pop_back();  // 去除\r
    read.nameLength = line.size();
    read.readLength = 0;  // 序列数据长度清零
    while (fastaFile.peek() != EOF && fastaFile.peek() != '>') {  // 读序列数据
      getline(fastaFile, line);
      if (line.back() == '\r') line.pop_back();  // 去除\r
      read.readLength += line.size();
    }
    // 写入节点 打印进度
    if (read.readLength <= USHRT_MAX) reads.push_back(read);  // 序列最长65535
    if(progress%(1024*1024) == 0) std::cout << ". " << std::flush;
    progress += 1;
  }
  fastaFile.close();
  std::cout << "\nfind " << progress << " sequences\n";
  if (reads.size() > INT_MAX) reads.resize(INT_MAX);  // 不超2147483647
  // 排序
  std::stable_sort(reads.begin(), reads.end(),
  [](const Read &a, const Read &b) {return a.readLength > b.readLength;});
  std::cout << "longest:\t" << reads.front().readLength << "\n";
  std::cout << "shortest:\t" << reads.back().readLength << "\n";
}

std::unordered_map<char, unsigned int> transTableGen = {  // 基因转码表
  {'a',0}, {'c',1}, {'g',2}, {'t',3}, {'u',3},
  {'A',0}, {'C',1}, {'G',2}, {'T',3}, {'U',3}
};  // 只用于makeData函数
std::unordered_map<char, unsigned int> transTablePro = {  // 蛋白转码表
  {'a', 0}, {'b', 1}, {'c', 2}, {'d', 3}, {'e', 4}, {'f', 5}, {'g', 6},
  {'A', 0}, {'B', 1}, {'C', 2}, {'D', 3}, {'E', 4}, {'F', 5}, {'G', 6},
  {'h', 7}, {'i', 8}, {'j', 9}, {'k',10}, {'l',11}, {'m',12}, {'n',13},
  {'H', 7}, {'I', 8}, {'J', 9}, {'K',10}, {'L',11}, {'M',12}, {'N',13},
  {'o',14}, {'p',15}, {'q',16}, {'r',17}, {'s',18}, {'t',19}, {'u',20},
  {'O',14}, {'P',15}, {'Q',16}, {'R',17}, {'S',18}, {'T',19}, {'U',20},
  {'v',21}, {'w',22}, {'x',23}, {'y',24}, {'z',25}, {'*',26}, {'-',27},
  {'V',21}, {'W',22}, {'X',23}, {'Y',24}, {'Z',25}
};  // 只用于makeData函数

// makeData 生成数据
template <int entropy>  // 基因熵2 蛋白熵5
void makeData(const std::string &read, std::vector<unsigned int> &buffer) {
  int length = read.size();  // 长度
  buffer.assign(2+(length+31)/32*entropy, 0);  // 初始化
  unsigned int packs[entropy] = {0};  // 打包后数据 编译会展开成寄存器
  unsigned int *packed = buffer.data()+2;  // 打包后数据存储位置
  int netLength = 0;  // 净长度
  std::unordered_map<char, unsigned int> *transTable;  // 转码表
  transTable = entropy==2?&transTableGen:&transTablePro;
  for (auto base:read) {
    auto iterator = (*transTable).find(base);  // 查找结果
    if (iterator == (*transTable).end()) continue;  // 未知碱基/氨基酸
    unsigned int pack = iterator->second;  // 编码
    for (int e=0; e<entropy; e++) packs[e] >>= 1;
    for (int e=0; e<entropy; e++) packs[e] += (pack>>e&1)<<31;
    netLength += 1;
    if (netLength%32 == 0) {  // 每32个氨基酸存储一次
      for (int e=0; e<entropy; e++) *(packed+e) = packs[e];
      packed += entropy;
    }
  }
  if (netLength%32 > 0) {  // 需要补齐
    for (int e=0; e<entropy; e++) packs[e] >>= (32-netLength%32);
    for (int e=0; e<entropy; e++) *(packed+e) = packs[e];
  }
  buffer[0] = length;  // 长度
  buffer[1] = netLength;  // 净长度
}

// makeDB 生成数据库
void makeDB(const Option &option, std::vector<Read> &reads) {
  // 常用数据
  int entropy = option.type==0?2:5;  // 熵
  std::ofstream packedFile(option.packedFile);  // 输出文件
  int readsCount = reads.size();  // 序列数
  std::vector<int> nameLengths(readsCount);  // 序列名长度
  std::vector<int> readLengths(readsCount);  // 序列长度
  std::vector<long> fastaOffsets(readsCount);  // fasta偏移
  std::vector<long> packedOffsets;  // packed偏移 释放reads后再申请空间 省内存
  {  // 写入: 序列种类 序列数 最长 最短 序列名长度 序列长度
    packedFile.write((char*)&readsCount, sizeof(int));  // 序列数
    packedFile.write((char*)&reads.front().readLength, sizeof(int));  // 最长
    packedFile.write((char*)&reads.back().readLength, sizeof(int));  // 最短
    packedFile.write((char*)&option.type, sizeof(int));  // 序列类型
    #pragma omp parallel for
    for (int i=0; i<readsCount; i++) {
      nameLengths[i] = reads[i].nameLength;  // 序列名长度
      readLengths[i] = reads[i].readLength;  // 序列长度
      fastaOffsets[i] = reads[i].offset;  // fasta偏移
    }
    packedFile.write((char*)nameLengths.data(), sizeof(int)*readsCount);
    packedFile.write((char*)readLengths.data(), sizeof(int)*readsCount);
    reads.resize(0); reads.shrink_to_fit();  // 清空省内存
  }
  {  // 计算偏移
    packedOffsets.resize(readsCount);  // packed偏移
    packedOffsets[0] = sizeof(int)*4+(sizeof(int)*2+sizeof(long))*readsCount;
    for (int i=0; i<readsCount-1; i++) {
      int temp = sizeof(char)*nameLengths[i];  // 序列名长度
      temp += sizeof(int)*((readLengths[i]+31)/32*entropy+2);  // 序列数据
      packedOffsets[i+1] = packedOffsets[i]+temp;
    }
    packedFile.write((char*)packedOffsets.data(), sizeof(long)*readsCount);
    nameLengths.resize(0); nameLengths.shrink_to_fit();  // 清空省内存
    readLengths.resize(0); readLengths.shrink_to_fit();  // 清空省内存
  }
  packedFile.close();
  #pragma omp parallel
  {  // 读数据并压缩 90%以上耗时在这里 需要多核处理器
    std::ifstream fastaFile(option.fastaFile);  // 输入
    std::ofstream packedFile(option.packedFile, std::ios::in);  // 输出
    std::string line, lines;  // 读入一行 多行合成
    std::vector<unsigned int> buffer;  // 生成的数据
    int progress = 0;
    #pragma omp for schedule(dynamic)  // 动态划分任务
    for (int i=0; i<readsCount; i++) {  // 遍历序列
      fastaFile.seekg(fastaOffsets[i], std::ios::beg);  // 移到fasta文件起始
      packedFile.seekp(packedOffsets[i], std::ios::beg);  // 移到packed文件起始
      getline(fastaFile, line);  // 读序列名
      if (line.back() == '\r') line.pop_back();  // 去除\r
      packedFile.write((char*)line.c_str(), line.size());  // 写序列名
      lines.clear();  // 用前先清空
      while (fastaFile.peek() != EOF && fastaFile.peek() != '>') {  // 读序列
        getline(fastaFile, line);
        if (line.back() == '\r') line.pop_back();  // 去除\r
        lines += line;
      }
      if (option.type == 0) {
        makeData<2>(lines, buffer);  // 基因序列
      } else {
        makeData<5>(lines, buffer);  // 蛋白序列
      }
      packedFile.write((char*)buffer.data(), sizeof(int)*buffer.size());
      if (omp_get_thread_num()==0 && progress++%(1024*32)==0)
        std::cout << ". " << std::flush;  // 打印进度
    }
    fastaFile.close();
    packedFile.close();
    #pragma omp master
    {std::cout << "\npack " << readsCount << " sequences\n";}
  }
}

//--------主函数--------//
int main(int argc, char **argv) {
  Timer timer; timer.start();  // 开始计时
  Option option;  // 输入选项
  parse(argc, argv, option);  // 解析输入选项
  std::vector<Read> reads;  // 序列长度与偏移
  makeIndex(option, reads);  // 生成文件索引
  makeDB(option, reads);  // 生成数据库
  std::cout << "total:\t\t"; timer.getDuration();  // 结束计时
  timer.getTimeNow();  // 时间戳
}
