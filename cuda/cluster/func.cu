// func.cu
// Endian:little 默认小端
#include <iostream>  // cout
#include <fstream>  // fstream
#include <vector>  // vector
#include <mpi.h>  // MPI_Init
#include <thrust/device_ptr.h>  // thrust
#include <thrust/fill.h>  // fill
#include <thrust/copy.h>  // copy
#include "timer.h"  // 计时器
#include "cmdline.h"  // 解析器
#include "func.h"  // 数据结构与函数

#define HTD cudaMemcpyHostToDevice  // 内存拷显存
#define DTH cudaMemcpyDeviceToHost  // 显存拷内存
#define DTD cudaMemcpyDeviceToDevice  // 显存拷显存

// init 初始化 ok
void init(int argc, char **argv, Option &option) {
  {  // MPI初始化
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &option.size);  // MPI进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &option.rank);  // MPI编号
  }
  {  // GPU初始化
    int32_t GPUnum;
    cudaGetDeviceCount(&GPUnum);  // GPU数
    cudaSetDevice(option.rank%GPUnum);  // 选择GPU
  }
  {  // 解析参数
    cmdline::parser parser;
    parser.add<std::string>("packed", 'p', "packed file", true, "");
    parser.add<std::string>("result", 'r', "result file", true, "");
    parser.add<float>("similarity", 's', "similarity 0.1-1.0", false, 0.9,
      cmdline::range<float>(0.1, 1.0));
    parser.add<int32_t>("mode", 'm', "mode 0:fast 1:precise", false, 0,
      cmdline::range<int32_t>(0, 1));
    parser.parse_check(argc, argv);
    option.packedFile = parser.get<std::string>("packed");  // packed文件
    option.resultFile = parser.get<std::string>("result");  // result文件
    option.similarity = parser.get<float>("similarity");  // 相似度
    option.mode = parser.get<int32_t>("mode");  // 精确模式
  }
  if (option.rank == 0) {  // 打印信息
    std::cout << "packed:\t\t" << option.packedFile << "\n";
    std::cout << "result:\t\t" << option.resultFile << "\n";
    std::cout << "similarlity:\t" << option.similarity << "\n";
    std::cout << "mode:\t\t" << option.mode << " (0:fast 1:precise)\n";
  }
  cudaDeviceSynchronize(); MPI_Barrier(MPI_COMM_WORLD);  // 同步
}

// readData 读数据
void readData(const Option &option, Data &data) {
  int32_t readsCount = 0;  // 序列数
  std::vector<size_t> offsets;  // packed文件中序列偏移
  std::ifstream packedFile(option.packedFile);  // 打开packed文件
  {  // 读索引
    packedFile.read((char*)&data.readsCount, sizeof(int32_t));  // 序列数
    packedFile.read((char*)&data.longest, sizeof(int32_t));  // 最长序列
    packedFile.read((char*)&data.shortest, sizeof(int32_t));  // 最短序列
    packedFile.read((char*)&data.type, sizeof(int32_t));  // 序列种类
    readsCount = data.readsCount;
    data.nameLengths.resize(readsCount);  // 序列名长度
    packedFile.read((char*)data.nameLengths.data(), sizeof(int32_t)*readsCount);
    data.readLengths.resize(readsCount);  // 序列长度
    packedFile.read((char*)data.readLengths.data(), sizeof(int32_t)*readsCount);
    offsets.resize(readsCount);  // 序列偏移
    packedFile.read((char*)offsets.data(), sizeof(size_t)*readsCount);
  }
  {  // 读数据
    int32_t entropy = data.type==0?2:5;  // 熵
    size_t bufferLength = 0;  // 数据总长度
    for (int32_t i=option.rank; i<readsCount; i+=option.size) {
      bufferLength += 2+(data.readLengths[i]+31)/32*entropy;
    }
    cudaMallocHost(&data.bufferH, sizeof(uint32_t)*bufferLength);  // host
    cudaMalloc(&data.bufferD, sizeof(uint32_t)*bufferLength);  // device
    cudaMallocHost(&data.readsH, sizeof(uint32_t*)*readsCount);  // host
    cudaMalloc(&data.readsD, sizeof(uint32_t*)*readsCount);  // device
    uint32_t **readsDTemp;  // readsD的host端缓存
    cudaMallocHost(&readsDTemp, sizeof(uint32_t*)*readsCount);  // 缓存
    data.names.resize(readsCount);
    bufferLength = 0;
    for (int32_t i=option.rank; i<readsCount; i+=option.size) {
      packedFile.seekg(offsets[i], std::ios::beg);  // 跳到序列开头
      data.names[i].resize(data.nameLengths[i]);
      packedFile.read((char*)data.names[i].data(), data.nameLengths[i]);
      int32_t length = 2+(data.readLengths[i]+31)/32*entropy;
      packedFile.read((char*)(data.bufferH+bufferLength),
        sizeof(uint32_t)*length);  // 序列数据
      data.readsH[i] = data.bufferH+bufferLength;  // host端索引
      readsDTemp[i] = data.bufferD+bufferLength;  // device端索引
      bufferLength += length;
    }
    cudaMemcpy(data.bufferD, data.bufferH, sizeof(uint32_t)*bufferLength, HTD);
    cudaMemcpy(data.readsD, readsDTemp, sizeof(uint32_t*)*readsCount, HTD);
    cudaFreeHost(readsDTemp );
  }
  packedFile.close();  // 读完了
  if (option.rank == 0){  // 打印信息
    std::cout << "reads:\t\t" << data.readsCount << "\n";
    std::cout << "longest:\t" << data.longest << "\n";
    std::cout << "shortest:\t" << data.shortest << "\n";
    std::cout << "type:\t\t" << data.type << " (0:gene 1:protein)\n";
  }
  cudaDeviceSynchronize(); MPI_Barrier(MPI_COMM_WORLD);  // 同步
}

// MakeTable 生成lookup table 专用于kermel_filter0
template<int32_t entropy, int32_t kLength, int32_t tabLen, int32_t wordStep>
__device__ inline void MakeTable(uint32_t *read, uint32_t *table) {
  const int32_t length = (1<<kLength)*tabLen/4;  // table长度
  const int32_t kmer = (kLength+entropy-1)/entropy;  // 碱基/氨基酸个数
  const uint32_t mask = (1<<kLength)-1;  // 掩码
  for (int32_t i=threadIdx.x; i<length; i+=blockDim.x) table[i] = 0;  // 清零
  __syncthreads();
  uint32_t packs[entropy] = {0};  // 打包数据
  uint32_t word = 0;  // 短词
  int32_t netLength = read[1];  // 净长度
  for (int32_t i=threadIdx.x*wordStep; i<netLength; i+=blockDim.x*wordStep) {
    int32_t order = 2+i/32*entropy;  // 数据位置
    for (int32_t e=0; e<entropy; e++) {  // 取数据
      packs[e] = __funnelshift_r(read[order+e], read[order+entropy+e], i%32);
    }
    for (int32_t j=0; j<kmer-1; j++) {  // 准备word
      word <<= entropy;
      for (int32_t e=0; e<entropy; e++) word += (packs[e]>>j&1)<<e;
    }
    for (int32_t j=kmer-1; j<wordStep+kmer-1; j++) {  // 遍历word
      word <<= entropy;
      for (int32_t e=0; e<entropy; e++) word += (packs[e]>>j&1)<<e;
      uint32_t kbit = word>>kmer*entropy-kLength&mask;
      if (i+j<netLength)  // 写入table
        atomicAdd(&table[kbit/(4/tabLen)], 1<<kbit%(4/tabLen)*8*tabLen);
    }
  }
  __syncthreads();
}
// block 越大越好 thread 不超128
// kernel_filter0 短词过滤 entropy 2/5 kLength 8-13 tabLen 1/2 wordStep 6/16
template<int32_t entropy, int32_t kLength, int32_t tabLen, int32_t wordStep>
__global__ void kernel_filter0(uint32_t **reads, uint32_t *represent,
float similarity, int32_t *jobs, int32_t jobCount) {
  if (represent[1] <= 32) return;  // 净长太小 不算
  __shared__ uint32_t table0[(1<<kLength)*tabLen/4];  // 代表序列
  __shared__ uint32_t table1[(1<<kLength)*tabLen/4];  // 剩余序列
  MakeTable<entropy, kLength, tabLen, wordStep>(represent, table0);
  for (int32_t idx=blockIdx.x; idx<jobCount; idx+=gridDim.x) {  // 遍历剩余序列
    if (reads[jobs[idx]][1] <= 32) continue;  // 净长太小 跳过
    if ((float)reads[jobs[idx]][0]/(float)represent[0]<0.8f) return;  // 差距大
    MakeTable<entropy, kLength, tabLen, wordStep>(reads[jobs[idx]], table1);
    int32_t sum = 0;
    {  // 统计Seed hit
      for (int32_t i=threadIdx.x; i<(1<<kLength)*tabLen/4; i+=blockDim.x)
        for (int32_t e=0; e<32; e+=8*tabLen)
          sum += min(table0[i]>>e&(1<<8*tabLen)-1,table1[i]>>e&(1<<8*tabLen)-1);
      __syncthreads();  // 即将汇总结果
      sum = __reduce_add_sync(0xFFFFFFFF, sum);
      if (threadIdx.x%32 == 0) table1[threadIdx.x/32] = sum;
      __syncthreads();  // 即将跨wrap通讯
      for (int32_t i=1; i<blockDim.x/32; i++) sum += table1[i];
    }
    if (threadIdx.x == 0){  // 判断通过过滤
      const int32_t kmer = (kLength+entropy-1)/entropy;  // 碱基/氨基酸个数
      uint32_t length = reads[jobs[idx]][0];  // 小心jobs被先写后读
      int32_t kmers = length-kmer+1;  // kbit数
      int32_t polluted = length*(1.0f-similarity)*kLength/entropy;  // 污染数
      if (sum<kmers-polluted) jobs[idx] = -1;
    }
    __syncthreads();
  }
}
#define FT0(entropy, kLength, tabLen, wordStep)\
kernel_filter0<entropy, kLength, tabLen, wordStep><<<block, thread>>>\
(data.readsD, represent, option.similarity, jobs, jobCount);
// tabLen = kLength<12?2:1 wordStep = entropy==2?16:6
#define filter0(entropy, kLength)\
if (entropy == 2) {\
  switch (kLength) {\
    case ( 8): FT0(2,  8, 2, 16); break; case ( 9): FT0(2,  9, 2, 16); break;\
    case (10): FT0(2, 10, 2, 16); break; case (11): FT0(2, 11, 2, 16); break;\
    case (12): FT0(2, 12, 1, 16); break; case (13): FT0(2, 13, 1, 16); break;\
  }\
} else {\
  switch (kLength) {\
    case ( 8): FT0(5,  8, 2, 6); break; case ( 9): FT0(5,  9, 2, 6); break;\
    case (10): FT0(5, 10, 2, 6); break; case (11): FT0(5, 11, 2, 6); break;\
    case (12): FT0(5, 12, 1, 6); break; case (13): FT0(5, 13, 1, 6); break;\
  }\
}

// block 越大越好 thread 越多越好
// 瓶颈在计算而不是访存 无需并行读数据
// kernel_filter1 seedHit过滤 entropy 2/5 kLength 14-18 wordStep 6/16
template<int32_t entropy, int32_t kLength, int32_t wordStep>
__global__ void kernel_filter1(uint32_t **reads, uint32_t *represent,
float similarity, int32_t *jobs, int32_t jobCount){
  if (represent[1] <= 32) return;  // 净长太短 不算
  __shared__ uint32_t table[(1<<kLength)/32];  // lookup table
  const int32_t kmer = (kLength+entropy-1)/entropy;  // 碱基/氨基酸个数
  const uint32_t mask = (1<<kLength)-1;  // 掩码
  uint32_t packs[entropy] = {0};  // 打包数据
  {  // 生成lookup table
    for (int32_t i=threadIdx.x; i<(1<<kLength)/32; i+=blockDim.x) table[i] = 0;
    __syncthreads();
    uint32_t netLength = represent[1];  // 序列净长度
    for (int32_t i=threadIdx.x*wordStep; i<netLength; i+=blockDim.x*wordStep) {
      uint32_t word = 0;  // 生成的word
      int32_t order = 2+i/32*entropy;  // 数据位置
      for (int32_t e=0; e<entropy; e++) packs[e] = __funnelshift_r
        (represent[order+e], represent[order+entropy+e], i%32);  // 取数据
      for (int32_t j=0; j<kmer-1; j++) {  // 准备word
        word <<= entropy;
        for (int32_t e=0; e<entropy; e++) word += (packs[e]>>j&1)<<e;
      }
      for (int32_t j=kmer-1; j<wordStep+kmer-1; j++) {  // 遍历word
        word <<= entropy;
        for (int32_t e=0; e<entropy; e++) word += (packs[e]>>j&1)<<e;
        uint32_t kbit = word>>kmer*entropy-kLength&mask;
        if (i+j<netLength) { // 写入table
          uint32_t intNum = kbit/32;  // 第几个int
          uint32_t intOff = 1<<kbit%32;  // int中位置
          while (true) {  // 原子操作
            uint32_t oldVal = table[intNum];
            uint32_t newVal = (oldVal&(~intOff))+intOff;
            if (oldVal == atomicCAS(&table[intNum], oldVal, newVal)) break;
          }
        }
      }
    }
    __syncthreads();
  }
  {  // 过滤
    int32_t index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
    int32_t range = gridDim.x*blockDim.x;  // 线程范围
    for (int32_t idx=index; idx<jobCount; idx+=range) {  // 遍历剩余序列
      uint32_t *read = reads[jobs[idx]];
      if (read[1] <= 32) continue;  // 净长太短 跳过
      uint32_t length = read[0];
      if ((float)length/(float)represent[0]<0.8f) return;  // 长度差距大
      uint32_t netLength = read[1];
      uint32_t word = 0;  // 生成的word
      int sum = 0;  // 命中kbit数
      for (int32_t e=0; e<entropy; e++) packs[e] = read[2+e];  // 取数据
      for (int32_t j=0; j<kmer-1; j++) {  // 准备word
        word <<= entropy;
        for (int32_t e=0; e<entropy; e++) word += (packs[e]>>j&1)<<e;
      }
      for (int32_t j=kmer-1; j<32; j++) {  // 第一组碱基/氨基酸(32个)
        word <<= entropy;
        for (int32_t e=0; e<entropy; e++) word += (packs[e]>>j&1)<<e;
        uint32_t kbit = word>>kmer*entropy-kLength&mask;
        uint32_t intNum = kbit/32;
        uint32_t intOff = 1<<kbit%32;
        sum += ((table[intNum]&intOff)>0);
      }
      for (int i=32; i<netLength; i+=32) {  // 遍历剩余碱基/氨基酸
        for (int32_t e=0; e<entropy; e++) packs[e] = read[2+i/32*entropy+e];
        for (int j=0; j<min(32, netLength-i); j++) {  // 遍历32个int
          word <<= entropy;
          for (int32_t e=0; e<entropy; e++) word += (packs[e]>>j&1)<<e;
          uint32_t kbit = word>>kmer*entropy-kLength&mask;
          uint32_t intNum = kbit/32;
          uint32_t intOff = 1<<kbit%32;
          sum += ((table[intNum]&intOff)>0);
        }
      }
      int32_t kmers = length-kmer+1;  // kbit数
      int32_t polluted = length*(1.0f-similarity)*kLength/entropy;  // 污染数
      if (sum < kmers-polluted) jobs[idx] = -1;
    }
  }
}
// wordStep = entropy==2?16:6
#define filter1(entropy, kLength)\
if (entropy == 2) {\
  switch (kLength) {\
    case (14): kernel_filter1<2, 14, 16><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (15): kernel_filter1<2, 15, 16><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (16): kernel_filter1<2, 16, 16><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (17): kernel_filter1<2, 17, 16><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (18): kernel_filter1<2, 18, 16><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
  }\
} else {\
  switch (kLength) {\
    case (14): kernel_filter1<5, 14, 6><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (15): kernel_filter1<5, 15, 6><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (16): kernel_filter1<5, 16, 6><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (17): kernel_filter1<5, 17, 6><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
    case (18): kernel_filter1<5, 18, 6><<<block, thread>>>\
      (data.readsD, represent, option.similarity, jobs, jobCount); break;\
  }\
}

// k32 32*32的计算核心
template<int32_t entropy>
__device__ inline void k32(uint32_t *Rows, uint32_t *Cols,
uint32_t *carrys, uint32_t *line, int32_t colCount) {
  uint32_t row = *line;  // 上一行结果
  for (int32_t k=0; k<colCount; k++) {  // 32*32的核心
    uint32_t match = 0xFFFFFFFF;  // 匹配上的碱基/氨基酸
    for (int32_t e=0; e<entropy; e++) {  // 匹配
      uint32_t temp = Rows[e]^0x00000000;
      if (Cols[e]>>k&1) temp = Rows[e]^0xFFFFFFFF;
      match &= ~temp;
    }
    uint32_t carry = *carrys>>k&1;  // 进位
    uint32_t match0 = row & match;
    uint32_t match1 = row & (~match);
    uint32_t carryRow = row + carry;
    carry = carryRow < row;  // 是否发生进位
    carryRow += match0;
    carry |= carryRow < match0;  // 是否发生进位
    row = carryRow | match1;
    *carrys &= ~(1<<k); *carrys += carry<<k;  // 写回进位
  }
  *line = row;
}
// kernel_dynamicGen 动态规划
template<int32_t entropy>
__global__ void kernel_dynamic(uint32_t **reads, uint32_t *represent,
const float similarity, int32_t *jobs, const int32_t jobCount){
  int32_t index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  if (index >= jobCount) return;  // 超出范围
  uint32_t *read = reads[jobs[index]];  // 起始位置
  uint32_t length1 = represent[0];  // 代表序列长度
  uint32_t length2 = read[0];  // 剩余序列长度
  uint32_t netLength1 = represent[1];  // 代表序列净长度
  uint32_t netLength2 = read[1];  // 剩余序列净长度
  uint32_t line[2048];  // 每行结果
  memset(line, 0xFF, (netLength1+31)/32*sizeof(uint32_t));  // 0:匹配 1:不匹配
  uint32_t Rows[entropy] = {0};  // 从行取的32个碱基/氨基酸
  uint32_t Cols[entropy] = {0};  // 从列取的32个碱基/氨基酸
  int32_t lshift = (length2-ceil(length2*similarity)+31)/32;  // 左偏移
  int32_t rshift = (length1-ceil(length2*similarity)+31)/32;  // 右偏移
  // 计算
  for (int32_t i=0; i<netLength2/32*32; i+=32) {  // 遍历列
    int32_t colCount = 32;  // 列向剩余
    uint32_t carrys = 0;  // 进位
    for (int32_t e=0; e<entropy; e++) Cols[e] = read[2+i/32*entropy+e];
    int32_t jstart = max(i/32-lshift, 0);
    int32_t jend = min(i/32+rshift, (netLength1+31)/32-1);
    for (int32_t j=jstart; j<=jend; j++) {  // 遍历行
      for (int32_t e=0; e<entropy; e++) Rows[e] = represent[2+j*entropy+e];
      k32<entropy>(Rows, Cols, &carrys, &line[j], colCount);
    }
  }
  for (int32_t i=netLength2/32*32; i<netLength2; i+=32) {  // 补齐
    int32_t colCount = netLength2-i;  // 列向剩余
    uint32_t carrys = 0;  // 进位
    for (int32_t e=0; e<entropy; e++) Cols[e] = read[2+i/32*entropy+e];
    int32_t jstart = max(i/32-lshift, 0);
    int32_t jend = min(i/32+rshift, (netLength1+31)/32-1);
    for (int32_t j=jstart; j<=jend; j++) {  // 遍历行
      for (int32_t e=0; e<entropy; e++) Rows[e] = represent[2+j*entropy+e];
      k32<entropy>(Rows, Cols, &carrys, &line[j], colCount);
    }
  }
  {  // 统计结果
    int32_t sum = 0;
    for (int32_t i=0; i<netLength1/32*32; i+=32) {
      sum += 32 - __popc(line[i/32]);
    }
    if (netLength1%32 != 0) {
      uint32_t mask = (1<<netLength1%32)-1;
      sum += netLength1%32 - __popc(line[netLength1/32]&mask);
    }
    int32_t cutoff = ceil((float)length2*similarity);
    if (sum < cutoff) jobs[index] = -1;
  }
}

// kernel_update 根据过滤和比对结果更新聚类结果
__global__ void kernel_update(int32_t *cluster, int32_t *remains,
int32_t remainCount, int32_t *jobs, int32_t jobCount, int32_t top) {
  for (int32_t idx=threadIdx.x; idx<jobCount; idx+=blockDim.x) {  // 更新cluster
    if (jobs[idx] != -1) cluster[jobs[idx]] = top;  // 聚类成功
  }
  if (threadIdx.x == 0) cluster[top] = top;
  __syncthreads();  // 等待cluster更新完成
  for (int32_t idx=threadIdx.x; idx<remainCount; idx+=blockDim.x) {  // remains
    if (cluster[remains[idx]] != -1) remains[idx] = -1;  // 已经聚类
  }
}

struct passJob {__device__ bool operator() (const int x) {return (x > -1);}};

// clustering 聚类
void clustering(const Option &option, Data &data) {
  int32_t readsCount = data.readsCount;  // 序列数
  int32_t *cluster, *remains, remainCount, *jobs, jobCount;  // 结果 剩余 任务
  thrust::device_ptr<int32_t> t_cluster, t_remains, t_jobs, t_end;  // thrust
  {  // 节点内数据初始化
    cudaMalloc(&cluster, sizeof(int32_t)*readsCount);  // 初始化cluster
    t_cluster = thrust::device_pointer_cast(cluster);
    thrust::fill(t_cluster, t_cluster+readsCount, -1);
    std::vector<int32_t> remainsH;  // 初始化remains
    for (int32_t i=option.rank; i<readsCount; i+=option.size) {
      remainsH.push_back(i);
    }
    remainCount = remainsH.size();  // 剩余序列个数
    cudaMalloc(&remains, sizeof(int32_t)*remainCount);
    cudaMemcpy(remains, remainsH.data(), sizeof(int32_t)*remainCount, HTD);
    t_remains = thrust::device_pointer_cast(remains);
    cudaMalloc(&jobs, sizeof(int32_t)*remainCount);  // 初始化jobs
    cudaMemcpy(jobs, remainsH.data(), sizeof(int32_t)*remainCount, HTD);
    t_jobs = thrust::device_pointer_cast(jobs);
    jobCount = remainCount;
  }
  int32_t entropy = data.type==0?2:5;  // 熵
  int32_t kLength = 0;  // kbit长度
  int32_t block = 0, thread = 0;  // 线程参数
  int32_t deviceCount = 0;  // 节点上设备个数
  cudaGetDeviceCount(&deviceCount);  // 获取设备个数
  cudaDeviceProp deviceProp;  // 设备属性
  cudaGetDeviceProperties(&deviceProp, option.rank%deviceCount);  // 设备属性
  int32_t mode = option.mode;  // 是否精确模式
  uint32_t *represent;  // 代表序列 最长65536
  cudaMallocManaged(&represent, sizeof(uint32_t)*10242);  // 序列最长情况
  if (option.similarity < 0.8) mode = 1;  // 相似度太小不启用过滤
  if (mode == 0) {  // 选择最佳过滤算法
    int32_t length = 2+(data.readsH[option.rank][0]+31)/32*entropy;
    std::memcpy(represent, data.readsH[option.rank], sizeof(uint32_t)*length);
    cudaMemPrefetchAsync(represent, sizeof(uint32_t)*length,
      option.rank%deviceCount);  // 预取
    int32_t rejectMax = 0, kLengthMax = 0;  // 拒绝掉的最多序列数和kLength
    int32_t blockMax = 0, threadMax = 0;  // reject最大时的block和thread
    for (kLength=8; kLength<=18; kLength++) {  // 长度在8-11的kLength 短词过滤
      if (8<=kLength && kLength <=11)  // short的短词过滤 8-11
        block = deviceProp.sharedMemPerMultiprocessor/((1<<kLength)*4+1024);
      if (12<=kLength && kLength <=13)  // char的短词过滤 12-13
        block = deviceProp.sharedMemPerMultiprocessor/((1<<kLength)*2+1024);
      if (14<=kLength && kLength <=18)  // SeedHit过滤 14-18,
        block = deviceProp.sharedMemPerMultiprocessor/((1<<kLength)/8+1024);
      block = std::min(block, 16);  // 每个sm的block数 最大16
      thread = deviceProp.maxThreadsPerMultiProcessor/block;
      thread = thread/32*32;  // 每个block的thread数
      block = block*deviceProp.multiProcessorCount;  // 总block数
      if (kLength <= 13) {  // 短词过滤
        filter0(entropy, kLength);
      } else {  // SeedHit过滤
        filter1(entropy, kLength);
      }
      t_end = thrust::copy_if(t_jobs, t_jobs+jobCount, t_jobs, passJob());
      jobCount = thrust::distance(t_jobs, t_end);
      int32_t rejectMaxTemp = rejectMax*1.1;
      if (kLength == 14) rejectMaxTemp = rejectMax*0.9;
      if (remainCount-jobCount >= rejectMaxTemp) {  // 记录拒绝序列数最多的参数
        rejectMax = remainCount-jobCount;  // 拒绝数
        kLengthMax = kLength;  // kbit长度
        blockMax = block;  // block数
        threadMax = thread;  // threads数
      }
      cudaMemcpy(jobs, remains, sizeof(int32_t)*remainCount, DTD);  // 恢复jobs
      jobCount = remainCount;  // 恢复jobCount
    }
    kLength = kLengthMax; block = blockMax; thread = threadMax;
    // 广播过滤算法的参数
    MPI_Bcast(&kLength, 1, MPI_INT, 0, MPI_COMM_WORLD);  // kbit长度
    MPI_Bcast(&block, 1, MPI_INT, 0, MPI_COMM_WORLD);  // block数
    MPI_Bcast(&thread, 1, MPI_INT, 0, MPI_COMM_WORLD);  // thread数
    MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);  // 聚类模式
    if (mode==0 && option.rank==0) std::cout << "kLength:\t" << kLength << "\n";
  }
  int32_t top = 0;  // 当前代表序列
  std::vector<int32_t> tops(option.size);  // 各节点代表序列
  while (true) {  // 聚类
    {  // 节点间同步
      MPI_Gather(&top, 1, MPI_INT, tops.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
      for (int32_t i=0; i<option.size; i++) top = std::min(top, tops[i]);
      MPI_Bcast(&top, 1, MPI_INT, 0, MPI_COMM_WORLD);  // 广播当前进度
      if (option.rank == 0) std::cout << "\r" << top << std::flush;  // 进度
      if (top == readsCount) break;  // 聚类完成
      size_t length = 2+(data.readLengths[top]+31)/32*entropy;
      if (top%option.size == option.rank) {  // 代表序列所在节点
        std::memcpy(represent, data.readsH[top], sizeof(uint32_t)*length);
      }
      MPI_Bcast(represent, length, MPI_INT, top%option.size, MPI_COMM_WORLD);
      cudaMemPrefetchAsync(represent, sizeof(uint32_t)*length,
        option.rank%deviceCount);  // 预取
    }
    if (mode==0) {  // 过滤
      if (kLength <= 13) {  // 短词过滤
        filter0(entropy, kLength);
      } else {  // SeedHit过滤
        filter1(entropy, kLength);
      }
      t_end = thrust::copy_if(t_jobs, t_jobs+jobCount, t_jobs, passJob());
      jobCount = thrust::distance(t_jobs, t_end);
    }
    {  // 比对
      if (entropy == 2) {  // 基因
        kernel_dynamic<2><<<(jobCount+127)/128, 128>>>
          (data.readsD, represent, option.similarity, jobs, jobCount);
      } else {
        kernel_dynamic<5><<<(jobCount+127)/128, 128>>>
          (data.readsD, represent, option.similarity, jobs, jobCount);
      }
    }
    {  // 聚类结果
      kernel_update<<<1, 1024>>>
        (cluster, remains, remainCount, jobs, jobCount, top);
      t_end = thrust::copy_if(t_remains, t_remains+remainCount,
        t_remains, passJob());
      remainCount = thrust::distance(t_remains, t_end);
      if (remainCount > 0) {
        cudaMemcpy(&top, remains, sizeof(int32_t), DTH);
      } else {
        top = readsCount;
      }
      cudaMemcpy(jobs, remains, sizeof(int32_t)*remainCount, DTD);
      jobCount = remainCount;
    }
  }
  if (option.rank == 0) std::cout << "\r" << readsCount << "\n";  // 聚类完成
  {  // 收尾
    data.result.resize(readsCount);
    cudaMemcpy(data.result.data(), cluster, sizeof(int32_t)*readsCount, DTH);
    cudaFree(data.readsD);
    cudaFree(data.bufferD);
    cudaFree(cluster);
    cudaFree(remains);
    cudaFree(jobs);
    cudaFree(represent);
  }
}

char table2[4] = {'A', 'C', 'G', 'T'};
char table5[28] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '*', '-'};

void saveResult(Option &option, Data &data) {
  int32_t readsCount = data.readsCount;  // 序列数
  {  // 汇总结果
    std::vector<int32_t> results(option.size*readsCount);  // 汇总的结果
    MPI_Gather(data.result.data(), readsCount, MPI_INT, results.data(),
      readsCount, MPI_INT, 0, MPI_COMM_WORLD);
    int32_t sum = 0;
    for (int32_t i=0; i<readsCount; i++) {
      data.result[i] = results[i%option.size*readsCount+i];
      if (data.result[i] == i) sum += 1;
    }
    MPI_Bcast(data.result.data(), readsCount, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "cluster:\t" << sum << "\n";
  }
  {  // 保存结果
    std::vector<int32_t> offsets(readsCount+1);  // 每个序列的偏移
    for (int32_t i=0; i<readsCount; i++) {
      offsets[data.result[i]+1] += data.nameLengths[i]+1;
      offsets[data.result[i]+1] += data.result[i]==i?data.readLengths[i]+1:2;
    }
    for (int32_t i=1; i<readsCount+1; i++) offsets[i] += offsets[i-1];
    if (option.rank == 0) {  // 保证文件存在
      std::ofstream resultFile(option.resultFile);
      resultFile << "\n";
      resultFile.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::ofstream resultFile(option.resultFile, std::ios::in);  // 保存结果
    std::string name;  // 要写入的name
    std::string read;  // 要写入的read
    int32_t entropy = data.type==0?2:5;  // 熵
    for (int32_t i=0; i<data.readsCount; i++) {
      if (i%option.size == option.rank) {  // 序列在当前节点
        if (data.result[i] == i) {  // 是代表序列
          resultFile.seekp(offsets[data.result[i]], std::ios::beg);
          name = data.names[i]+"\n";
          resultFile.write(name.data(), name.size());
          read.clear();  // 解码序列数据
          for (int32_t l=0; l<data.readsH[i][1]; l++) {
            uint32_t word = 0;
            for (int32_t e=0; e<entropy; e++) {
              word += (data.readsH[i][2+l/32*entropy+e]>>(l%32)&1)<<e;
            }
            if (entropy == 2) read.push_back(table2[word]);
            if (entropy == 5) read.push_back(table5[word]);
          }
          for (int32_t l=data.readsH[i][1]; l<data.readsH[i][0]; l++) {
            read.push_back('N');
          }
          read.push_back('\n');
          resultFile.write(read.data(), read.size());
        } else {  // 不是代表序列
            resultFile.seekp(offsets[data.result[i]], std::ios::beg);
            name = "  "+data.names[i]+"\n";
            resultFile.write(name.data(), name.size());
        }
      }
      offsets[data.result[i]] += data.nameLengths[i]+1;
      offsets[data.result[i]] += data.result[i]==i?data.readLengths[i]+1:2;
    }
    resultFile.close();
  }
  cudaFreeHost(data.readsH);
  cudaFreeHost(data.bufferH);
}
