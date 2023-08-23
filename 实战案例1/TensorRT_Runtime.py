from typing import Union, Optional, Sequence, Dict 
import torch 
import tensorrt as trt
import numpy as np
import cv2 
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class TRTWrapper(torch.nn.Module): 
    def __init__(self, engine: Union[str, trt.ICudaEngine], 
                 output_names: Optional[Sequence[str]] = None) -> None: 
        super().__init__() 

        # 1. 如果提供了引擎的路径（字符串），则加载它
        self.engine = engine 
        if isinstance(self.engine, str): 
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: 
                with open(self.engine, mode='rb') as f: 
                    engine_bytes = f.read() 
                self.engine = runtime.deserialize_cuda_engine(engine_bytes) 
                
        names = [_ for _ in self.engine] 
        input_names = list(filter(self.engine.binding_is_input, names)) 
        self._input_names = input_names 
        self._output_names = output_names 
        # 如果未提供 输出名，则从引擎中推断它们
        if self._output_names is None: 
            output_names = list(set(names) - set(input_names)) 
            self._output_names = output_names 
        
        

class TRTModule(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine], input_names, output_names):
        super(TRTModule, self).__init__()
        
        # 1. 如果提供了引擎的路径（字符串），则加载它
        self.engine = engine 
        if isinstance(self.engine, str): 
            with trt.Logger() as logger, trt.Runtime(logger) as runtime: 
                with open(self.engine, mode='rb') as f: 
                    engine_bytes = f.read() 
                self.engine = runtime.deserialize_cuda_engine(engine_bytes) 


        self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

        

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # 检查输入名是否有效
        assert self.input_names is not None 
        assert self.output_names is not None
        
        # 1. 创建一个绑定列表，其中每个输入和输出都有一个位置
        bindings = [None] * (len(self.input_names) + len(self.output_names)) 
        
        # 2. 为每个输入设置绑定
        profile_id = 0 
        for input_name, input_tensor in inputs.items(): 
            # 检查 输入的形状 是否匹配 engine 的预期
            profile = self.engine.get_profile_shape(profile_id, input_name) 
            assert input_tensor.dim() == len(profile[0]), 'Input dim is different from engine profile.' 
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, profile[2]): 
                assert s_min <= s_input <= s_max, 'Input shape should be between ' \
                + f'{profile[0]} and {profile[2]}' \
                + f' but get {tuple(input_tensor.shape)}.' 
            idx = self.engine.get_binding_index(input_name) 
 
            # All input tensors must be gpu variables 
            assert 'cuda' in input_tensor.device.type 
            input_tensor = input_tensor.contiguous() 
            if input_tensor.dtype == torch.long: 
                input_tensor = input_tensor.int() 
            self.context.set_binding_shape(idx, tuple(input_tensor.shape)) 
            bindings[idx] = input_tensor.contiguous().data_ptr() 
 
        # 为每个输出创建一个张量，并设置其绑定
        outputs = {} 
        for output_name in self.output_names: 
            idx = self.engine.get_binding_index(output_name) 
            shape = tuple(self.context.get_binding_shape(idx)) 
 
            output = torch.empty(size=shape, dtype=torch.float32, device=torch.device('cuda')) 
            outputs[output_name] = output 
            bindings[idx] = output.data_ptr() 
        
        # 异步执行 TensorRT 推理:
        self.context.execute_async_v2(bindings, 
                                      torch.cuda.current_stream().cuda_stream) 
        return outputs 
    
            
# 构建输入  输入的预处理要和模型训练时一致
img_input = cv2.imread("./Person.jpeg")
img_input = cv2.resize(img_input, (448,448))
img_input = img_input.transpose(2,0,1)
img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
img_input = torch.from_numpy(img_input)

print(f"img_input.shape:{img_input.shape}")
device = torch.device('cuda:0')
img_input = img_input.to(device)


# 获取输入输出的名字
NameModel = TRTWrapper('resnet34_3dpose.engine', None) 
print(NameModel._input_names)
print(NameModel._output_names)


# 运行模型
input_names = NameModel._input_names
output_names = NameModel._output_names
trt_model = TRTModule('resnet34_3dpose.engine', input_names, output_names)

img_input = {name: img_input for name in input_names}
result_trt = trt_model(img_input)
print(f"result_trt:{result_trt}")


# 接下来对模型进行后处理，即解析 result_trt 的 四个输出