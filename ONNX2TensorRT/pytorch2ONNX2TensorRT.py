import torch 
import onnx 
import tensorrt as trt 
 
 
device = torch.device('cuda:0') 
onnx_model_path = 'model.onnx' 
 
class NaiveModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.pool = torch.nn.MaxPool2d(2, 2) 
 
    def forward(self, x): 
        return self.pool(x) 
 
# generate ONNX model 
torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model_path, input_names=['input'], output_names=['output'], opset_version=11) 
onnx_model = onnx.load(onnx_model_path) 
 
 
# 由 logger 创建 builder
# 由 builder 创建 config 和 network
logger = trt.Logger(trt.Logger.ERROR) 
builder = trt.Builder(logger) 
config = builder.create_builder_config() 
EXPLICIT_BATCH = 1 << (int)( 
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
network = builder.create_network(EXPLICIT_BATCH) 
 
# parse onnx 
parser = trt.OnnxParser(network, logger) 

# 尝试解析 ONNX 模型。如果失败，提取并打印错误消息
if not parser.parse(onnx_model.SerializeToString()): 
    error_msgs = '' 
    for error in range(parser.num_errors): 
        error_msgs += f'{parser.get_error(error)}\n' 
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
 
 
# 设置 config 最大的工作空间
config.max_workspace_size = 1<<20               

# 创建一个优化配置文件
profile = builder.create_optimization_profile()   
profile.set_shape('input', [1,3,224,224], [1,3,224,224], [1,3,224,224])   # 为优化配置文件设置输入的形状
config.add_optimization_profile(profile)   # 将优化配置文件添加到构建配置中

# 使用 builder  network  config 创建 TensorRT 引擎
# 1. 解析 network graph，注册计算层
# 2. 删除冗余的常量节点与无用层
# 3. 进行 model fusion，构成新的计算层
# 4. 计算层最优执行 kernel 搜索
# 5. 打包最终的 kernel 方案构成 engine
with torch.cuda.device(device): 
    engine = builder.build_engine(network, config) 
 
with open('model.engine', mode='wb') as f: 
    f.write(bytearray(engine.serialize())) 
    print("generating file done!") 
 