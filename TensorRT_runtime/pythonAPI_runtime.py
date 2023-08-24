from typing import Union, Optional, Sequence, Dict 
import torch 
import tensorrt as trt 
 
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
                
        # 2. 创建执行上下文
        self.context = self.engine.create_execution_context() 
        
        # 3. 获取输入和输出的名字
        names = [_ for _ in self.engine] 
        input_names = list(filter(self.engine.binding_is_input, names)) 
        self._input_names = input_names 
        self._output_names = output_names 
        # 如果未提供 输出名，则从引擎中推断它们
        if self._output_names is None: 
            output_names = list(set(names) - set(input_names)) 
            self._output_names = output_names 
 
    def forward(self, inputs: Dict[str, torch.Tensor]): 
        # 检查输入名是否有效
        assert self._input_names is not None 
        assert self._output_names is not None
        
        # 1. 创建一个绑定列表，其中每个输入和输出都有一个位置
        bindings = [None] * (len(self._input_names) + len(self._output_names)) 
        
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
        for output_name in self._output_names: 
            idx = self.engine.get_binding_index(output_name) 
            shape = tuple(self.context.get_binding_shape(idx)) 
 
            output = torch.empty(size=shape, dtype=torch.float32, device=torch.device('cuda')) 
            outputs[output_name] = output 
            bindings[idx] = output.data_ptr() 
        
        # 异步执行 TensorRT 推理:
        self.context.execute_async_v2(bindings, 
                                      torch.cuda.current_stream().cuda_stream) 
        return outputs 
 
model = TRTWrapper('model.engine', ['output']) 
# 输入了一个dict， key 为 input， value 为 torch.Tensor
output = model(dict(input = torch.randn(1, 3, 224, 224).cuda())) 
print(output) 