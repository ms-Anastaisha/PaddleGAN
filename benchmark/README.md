# PaddGAN模型性能复现
## 目录

```
├── README.md               # 说明文档
├── benchmark.yaml          # 配置文件，设置测试模型及模型参数
├── run_all.sh              # 执行入口，测试并获取所有生成对抗模型的训练性能
└── run_benchmark.sh        # 执行实体，测试单个分割模型的训练性能  
```

## 环境介绍
### 物理机环境
- 单机（单卡、8卡）
  - 系统：CentOS release 7.5 (Final)
  - GPU：Tesla V100-SXM2-32GB * 8
  - CPU：Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz * 80
  - CUDA、cudnn Version: cuda10.2-cudnn7

#### 备注
BasicVSR模型因竞品torch模型只能测4卡，故这里也测4卡。

### Docker 镜像

- **镜像版本**: `registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7`
- **paddle 版本**: `2.1.2`
- **CUDA 版本**: `10.2`
- **cuDnn 版本**: `7`

## 测试步骤

```bash
bash benchmark/run_all.sh  
```

## 输出

执行完成后，在PaddleGAN目录会产出模型训练性能数据的文件，比如`esrgan_mp_bs32_fp32_8`等文件。
