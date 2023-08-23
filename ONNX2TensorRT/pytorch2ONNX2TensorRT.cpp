#include <fstream> 
#include <iostream> 
 
#include <NvInfer.h> 
#include <NvOnnxParser.h> 
#include <common/logger.h> 
 
using namespace nvinfer1; 
using namespace nvonnxparser; 
using namespace sample; 
 
int main(int argc, char** argv) 
{ 
        // Create builder 
        Logger m_logger; 
        IBuilder* builder = createInferBuilder(m_logger); 

        // Build config
        IBuilderConfig* config = builder->createBuilderConfig(); 

        // Create model to populate the network 
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch); 
 
        // Parse ONNX file 
        IParser* parser = nvonnxparser::createParser(*network, m_logger); 
        bool parser_status = parser->parseFromFile("/home/zwgd/code/TensorRT/实战案例1/Resnet34_3inputs_448x448_20200609.onnx", static_cast<int>(ILogger::Severity::kWARNING)); 
 

        // 设置 config 最大的工作空间 设置为6000MB
        config->setMaxWorkspaceSize(6000 * 1024 * 1024);
        // config->setMaxWorkspaceSize(1 << 20); 

        // TensorRT默认的数据精度为FP32，我们还可以设置FP16或者INT8，前提是该硬件平台支持这种数据精度
        // if (builder->platformHasFastFp16()) {
        //         config->setFlag(nvinfer1::BuilderFlag::kFP16);
        // }
        
        // 创建一个优化配置文件
        Dims dim = network->getInput(0)->getDimensions(); 
        if (dim.d[0] == -1)  // -1 means it is a dynamic model 
        { 
                const char* name = network->getInput(0)->getName(); 
                IOptimizationProfile* profile = builder->createOptimizationProfile(); 
                profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3])); 
                profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(1, dim.d[1], dim.d[2], dim.d[3])); 
                profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(1, dim.d[1], dim.d[2], dim.d[3])); 
                config->addOptimizationProfile(profile); 
        } 
 
        // 使用 builder  network  config 创建 TensorRT 引擎
        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config); 
        assert(engine != nullptr); 

        // Serialize the model to modelStream
        IHostMemory* modelStream = engine->serialize(); 
 
        // 确保文件 model.engine 存在
        std::ofstream p("/home/zwgd/code/TensorRT/实战案例1/Resnet34_3inputs_448x448_20200609.engine", std::ios::binary); 
        if (!p) { 
                std::cerr << "could not open output file to save model" << std::endl; 
                return -1; 
        } 
        // 文件p 写入 modelStream
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size()); 
        std::cout << "generate file success!" << std::endl; 
 
        // Release resources 
        modelStream->destroy(); 
        std::cout << "Destroyed modelStream" << std::endl;
        network->destroy(); 
        std::cout << "Destroyed network" << std::endl;
        engine->destroy(); 
        std::cout << "Destroyed engine" << std::endl;
        config->destroy(); 
        std::cout << "Destroyed config" << std::endl;
        builder->destroy(); 
        std::cout << "Destroyed builder" << std::endl;


        return 0; 
} 