.onnx 转换为 .engine(.trt)  

#### 注意 .trt完全等同于.engine  都表示 TensorRT 的推理引擎,如何命名取决于个人偏好
 
#### 方法1：使用TensorRT自带的trtexc把onnx转化为.engine(.trt)
(trtexc位于安装好的TensorRT-8.5.1.7的bin文件夹下)

1) cd ~/TensorRT-8.5.1.7/bin
2) conda activate tensorRT
3) ./trtexec --onnx=<onnxModelFilePath>  --saveEngine=<trtEngineFliePath>

举例：./trtexec --onnx=/home/zwgd/code/TensorRT/ONNX2TensorRT/model.onnx  --saveEngine=/home/zwgd/code/TensorRT/ONNX2TensorRT/model.engine

参数推荐：
--maxBatch=200：指定trt最大的batch_size=200
--workspace=1000：指定转化过程中的工作空间是1000M
(网络层实现过程中通常会需要一些临时的工作空间，这个属性会限制最大能申请的工作空间的容量，
如果容量不够的话，会导致该网络层不能成功实现而导致错误)
--fp16：指定采用了fp16精度，也还可以是int8 
(TensorRT 会默认开启 TF32 数据格式，它是截断版本的 FP32，只有 19 bit，保持了 fp16 的精度和 fp32 的指数范围)
(一般来说，只开 fp16 可以把速度提一倍并且几乎不损失精度；但是开 --int8 会大大损失精度，速度会比 fp16 快，但不一定能快一倍)


#### 方法2:使用 tensorRT python API 把onnx转化为.engine(.trt)


#### 方法3:使用 tensorRT Cpp API 把onnx转化为.engine(.trt)

