# 该项目介绍
该项目Fork自[CodeLlama](https://github.com/meta-llama/codellama)，主要做了NPU兼容性小改。
使其可以运行NPU设备上


# 项目使用

## 环境准备

1. 准备NPU 服务器并确保已安装相关驱动

2. 安装运行时docker
建议使用容器运行，这里以docker为例，需要在环境上预先安装docker。

3.克隆项目

```
git clone https://github.com/ai-study-room/codellama

``` 

4.构建运行镜像。

5.下载模型文件，参见：[模型下载](https://github.com/meta-llama/codellama?tab=readme-ov-file#download)

```
cd codellama
docker build -t codellama:v1 -f Dockfile .

```

说明：由于该处主要用于运行测试，容器镜像没有自动启动脚本，请根据实际情况修改

## 运行

1. 启动容器

```
docker run -d \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
-v /usr/local/dcmi:/usr/local/dcmi:ro \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
-v /usr/local/Ascend:/usr/local/Ascend:ro \
-v <project_path>/CodeLlama-34-Instruct:/codellama/CodeLlama-34b-Instruct \
-e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common \
-it codellama:v1 /bin/bash
```

2.进入容器测试

查看容器ID
```
docker ps -a | grep codellame 

```

进入容器

```
docker exec -it <container-id> bash
```


运行测试
```
torchrun --nproc_per_node 4 example_completion.py \
    --ckpt_dir CodeLlama-34b-Instruction/ \
    --tokenizer_path CodeLlama-34b-Instruction/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```


其它测试脚本同上游社区一致

```
torchrun --nproc_per_node 1 example_infilling.py \
    --ckpt_dir CodeLlama-7b/ \
    --tokenizer_path CodeLlama-7b/tokenizer.model \
    --max_seq_len 192 --max_batch_size 4
```


```
torchrun --nproc_per_node 1 example_instructions.py \
    --ckpt_dir CodeLlama-7b-Instruct/ \
    --tokenizer_path CodeLlama-7b-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 4
```
