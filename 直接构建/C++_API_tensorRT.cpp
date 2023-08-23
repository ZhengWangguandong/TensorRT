#include <fstream> 
#include <iostream> 

#include <NvInfer.h> 
#include </samples/common/logger.h> 
 
using namespace nvinfer1; 
using namespace sample; 
 
const char* IN_NAME = "input"; 
const char* OUT_NAME = "output"; 
static const int IN_H = 224; 
static const int IN_W = 224; 
static const int BATCH_SIZE = 1; 
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
 
int main(int argc, char** argv) 
{ 
        // Create builder 
        Logger m_logger; 

        // nvinfer1:: createInferBuilder 对应 Python 中的 tensorrt.Builder
        // 需要传入 ILogger 类的实例，但是 ILogger 是一个抽象类，需要用户继承该类并实现内部的虚函数
        // 此处我们直接使用了 TensorRT 包解压后的 samples 文件夹
        IBuilder* builder = createInferBuilder(m_logger); 
        IBuilderConfig* config = builder->createBuilderConfig(); 
 
        // Create model to populate the network 
        INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH); 
        ITensor* input_tensor = network->addInput(IN_NAME, DataType::kFLOAT, Dims4{ BATCH_SIZE, 3, IN_H, IN_W }); 
        IPoolingLayer* pool = network->addPoolingNd(*input_tensor, PoolingType::kMAX, DimsHW{ 2, 2 }); 
        pool->setStrideNd(DimsHW{ 2, 2 }); 
        pool->getOutput(0)->setName(OUT_NAME); 
        network->markOutput(*pool->getOutput(0)); 
 
        // Build engine 
        IOptimizationProfile* profile = builder->createOptimizationProfile(); 
        profile->setDimensions(IN_NAME, OptProfileSelector::kMIN, Dims4(BATCH_SIZE, 3, IN_H, IN_W)); 
        profile->setDimensions(IN_NAME, OptProfileSelector::kOPT, Dims4(BATCH_SIZE, 3, IN_H, IN_W)); 
        profile->setDimensions(IN_NAME, OptProfileSelector::kMAX, Dims4(BATCH_SIZE, 3, IN_H, IN_W)); 
        config->setMaxWorkspaceSize(1 << 20); 
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config); 
 
        // Serialize the model to engine file 
        IHostMemory* modelStream{ nullptr }; 
        assert(engine != nullptr); 
        modelStream = engine->serialize(); 
 
        std::ofstream p("model.engine", std::ios::binary); 
        if (!p) { 
                std::cerr << "could not open output file to save model" << std::endl; 
                return -1; 
        } 
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size()); 
        std::cout << "generating file done!" << std::endl; 
 
        // Release resources 
        modelStream->destroy(); 
        network->destroy(); 
        engine->destroy(); 
        builder->destroy(); 
        config->destroy(); 
        return 0; 
} 