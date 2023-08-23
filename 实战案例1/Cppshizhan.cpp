#include <fstream> 
#include <iostream> 
 
#include <NvInfer.h> 
#include <../samples/common/logger.h> 
 
// 预处理器技巧: 在预处理阶段，将多行合并成一行。
// 在此处，将长宏定义分成多行，但在实际使用中，编译器会将这些行合并在一起，就像它们是单独的一行一样

// 在CUDA操作  返回非零状态时  输出错误消息，并终止程序执行
// 使用 CHECK 宏来检查CUDA操作的状态并采取适当的措施，以确保程序在出现错误时能够正确处理
#define CHECK(status) \ 
    do\ 
    {\ 
        auto ret = (status);\ 
        if (ret != 0)\ 
        {\ 
            std::cerr << "Cuda failure: " << ret << std::endl;\ 
            abort();\ 
        }\ 
    } while (0) 
 
//在CUDA操作返回非零状态时输出错误消息并终止程序执行

using namespace nvinfer1; 
using namespace sample; 
 
const char* IN_NAME = "input"; 
const char* OUT_NAME = "output"; 
static const int IN_H = 224; 
static const int IN_W = 224; 
static const int BATCH_SIZE = 1; 
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
 
 
/*      context：TensorRT 的执行上下文，用于运行推断
        input：指向输入数据的指针
        output：指向输出数据的指针
        batchSize：要推断的批次大小
*/
void doInference(IExecutionContext& context, float* input, float* output, int batchSize) 
{ 
        // 从 context 中获取 CUDA 引擎
        const ICudaEngine& engine = context.getEngine(); 
 
        // 确认引擎有两个绑定点，一个用于输入，一个用于输出
        assert(engine.getNbBindings() == 2); 

        // 定义一个指针数组，用于存储输入和输出的设备缓冲区地址
        // 模型的推理是在GPU上进行的，所以会存在搬运输入、输出数据的操作，因此有必要在GPU上创建内存区域用于存放输入、输出数据
        void* buffers[2]; 
 
        // 获取输入和输出张量的绑定索引。这些索引将用于 确定 哪个缓冲区用于输入 和 哪个用于输出        
        const int inputIndex = engine.getBindingIndex(IN_NAME); 
        const int outputIndex = engine.getBindingIndex(OUT_NAME); 
 
        // 为 输入和输出 在 GPU 上 分配内存
        static const int input_size = batchSize * 3 * IN_H * IN_W;
        static const int output_size = batchSize * 3 * IN_H * IN_W /4;
        CHECK(cudaMalloc(&buffers[inputIndex], input_size * sizeof(float))); 
        CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float))); 
 
        // 创建一个 CUDA 流，用于异步操作
        cudaStream_t stream; 
        CHECK(cudaStreamCreate(&stream)); 
 
        // 异步地将输入数据从 主机 复制到 GPU 设备
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream)); 
        // 推理： 在给定的 CUDA 流上异步执行模型推断。这将在 GPU 上处理输入数据，并将结果放入输出缓冲区
        context.enqueue(batchSize, buffers, stream, nullptr); 
        // 异步地将输出数据从 GPU 设备复制到主机
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream)); 
        // 等待 CUDA 流上的所有操作完成。这确保了推断已完成，且数据已经从设备复制回主机
        cudaStreamSynchronize(stream); 
 
        // 清理资源：销毁 CUDA 流并释放 GPU 上为输入和输出分配的内存
        cudaStreamDestroy(stream); 
        CHECK(cudaFree(buffers[inputIndex])); 
        CHECK(cudaFree(buffers[outputIndex])); 
} 
 
// 5. 对模型的输出结果进行解析，进行必要的后处理后得到最终的结果
void FinalProcess(float* output) 
{

}

int main(int argc, char** argv) 
{ 
        // 声明一个字符指针 trtModelStream， 用于存储模型的序列化数据
        char *trtModelStream{ nullptr }; 
        // 声明一个大小变量 size，初始化为 0，用于存储模型文件的大小
        size_t size{ 0 }; 
 
        // 检查模型文件是否存在
        if (!std::ifstream("model.engine")) { 
                std::cout << "Error: File not found." << std::endl; 
                return -1; 
        }

        std::ifstream file("model.engine", std::ios::binary); 
        if (file.good()) { 
                // 将文件指针移动到文件末尾，以便获取文件的大小
                file.seekg(0, file.end); 
                // 获取文件的大小
                size = file.tellg(); 
                // 将文件指针重新设置为文件开头，以便从头开始读取文件内容
                file.seekg(0, file.beg); 
                // 使用之前获取的文件大小，在堆上分配一块内存。将指针 trtModelStream 指向这块内存 
                trtModelStream = new char[size]; 
                assert(trtModelStream); 
                file.read(trtModelStream, size); 
                file.close(); 
        } 
 
        // Create runtime
        Logger m_logger; 
        IRuntime* runtime = createInferRuntime(m_logger); 
        assert(runtime != nullptr);

        // 将 trtModelStream 解序列为 ICudaEngine 对象， ICudaEngine 对象中存放着经过TensorRT优化后的模型
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr); 
        assert(engine != nullptr); 

        // 创建执行上下文   创建一个 IExecutionContext 对象来管理推理的过程
        IExecutionContext* context = engine->createExecutionContext(); 
        assert(context != nullptr); 
 


                 /********下方是构建输入和检测输出          可自行DIY设计程序******/
        // 1. 对输入图像数据做与模型训练时一样的预处理操作 (较麻烦)
        // 2. 把模型的输入数据从CPU拷贝到GPU中
        // 3. 调用模型推理接口 执行推理操作
        // 4. 把模型的输出数据从GPU拷贝到CPU中
        // 5. 对模型的输出结果进行解析，进行必要的后处理后得到最终的结果  (最麻烦)

        // 构建输入数据 input, 可DIY成: 1.对输入图像数据做与模型训练时一样的预处理操作
        float input[BATCH_SIZE * 3 * IN_H * IN_W]; 
        for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++) 
                input[i] = 1; 
 
        // 给模型输出数据分配相应的CPU内存
        float output[BATCH_SIZE * 3 * IN_H * IN_W /4]; 

        // 重点步骤: 执行推理操作  相当于 2. 3. 4. 步骤
        doInference(*context, input, output, BATCH_SIZE); 
 
        // 可以对模型的输出结果 output进行后处理，以得到最终结果
        FinalProcess(output);

        // Destroy the engine 
        context->destroy(); 
        engine->destroy(); 
        runtime->destroy(); 
        return 0; 
} 