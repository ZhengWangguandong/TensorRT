import tensorrt as trt 
 
verbose = True 
IN_NAME = 'input' 
OUT_NAME = 'output' 
IN_H = 224 
IN_W = 224 
BATCH_SIZE = 1 
 
# 设置网络定义创建标志为 EXPLICIT_BATCH。这意味着网络将使用显式的批处理大小，而不是动态的
EXPLICIT_BATCH = 1 << (int)( 
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
 
# 创建一个 TensorRT 日志记录器  如果 verbose 为 True，则使用详细的日志级别
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger() 

with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config( 
) as config, builder.create_network(EXPLICIT_BATCH) as network: 
    # define network 
    input_tensor = network.add_input( 
        name=IN_NAME, dtype=trt.float32, shape=(BATCH_SIZE, 3, IN_H, IN_W)) 
    pool = network.add_pooling( 
        input=input_tensor, type=trt.PoolingType.MAX, window_size=(2, 2)) 
    pool.stride = (2, 2) 
    pool.get_output(0).name = OUT_NAME             # 设置池化层的输出张量名称
    network.mark_output(pool.get_output(0))        # 标记网络的输出张量 
 
    # 序列化模型为引擎文件
    # 1. 为构建器创建一个优化配置文件
    profile = builder.create_optimization_profile()  
    # 2. 为优化配置文件设置输入张量的形状
    profile.set_shape_input('input', *[[BATCH_SIZE, 3, IN_H, IN_W]]*3)  
    # 3. 设置构建器的最大批处理大小
    builder.max_batch_size = 1 
    # 4. 设置构建过程中可用的最大工作空间大小
    config.max_workspace_size = 1 << 30 
    # 5(核心步骤). 使用给定的网络和配置构建一个 TensorRT 引擎 
    engine = builder.build_engine(network, config) 
    with open('model_python_trt.engine', mode='wb') as f: 
        f.write(bytearray(engine.serialize())) 
        print("generating file done!") 