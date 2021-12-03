## 前提

1. 已安装好CUDA和CUDNN，且版本适配。

2. 在安装前先检查一下，电脑的cuda版本和pytorch内的cuda版本是否一样，不一样的话就把低版本的进行升级。

```shell
# 查看电脑的cuda版本、
>> nvcc -V
```

```shell
# pytorch内的cuda版本
import torch
torch.version.cuda
```

## 安装

按照官网的命令输入即可

```shell
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

**注意安装可能会报错**

```shell
Cuda extensions are being compiled with a version of Cuda that does " +
not match the version used to compile Pytorch binaries. " +
Pytorch binaries were compiled with Cuda
————————————————
版权声明：本文为CSDN博主「咆哮的阿杰」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_34914551/article/details/103203862
```

但我没遇上！

在保证cuda版本一致的前提下进入apex文件夹，使用命令：python setup.py install 即可安装成功

验证安装成功

```python
from apex import amp
```

​	没有报错就是成功了。



## 试验：

```shell
PS D:\YOLOX\apex-master> python .\setup.py install
```


torch.__version__  = 1.9.0+cu111

D:\YOLOX\apex-master\setup.py:67: UserWarning: Option --pyprof not specified. Not installing PyProf dependencies!
  warnings.warn("Option --pyprof not specified. Not installing PyProf dependencies!")
running install
running bdist_egg
running egg_info
writing apex.egg-info\PKG-INFO
writing dependency_links to apex.egg-info\dependency_links.txt
writing top-level names to apex.egg-info\top_level.txt
adding license file 'LICENSE' (matched pattern 'LICEN[CS]E*')
reading manifest file 'apex.egg-info\SOURCES.txt'
writing manifest file 'apex.egg-info\SOURCES.txt'
installing library code to build\bdist.win-amd64\egg
running install_lib
running build_py
creating build
creating build\lib
creating build\lib\apex
copying apex\__init__.py -> build\lib\apex
creating build\lib\apex\amp
copying apex\amp\amp.py -> build\lib\apex\amp
copying apex\amp\compat.py -> build\lib\apex\amp
copying apex\amp\frontend.py -> build\lib\apex\amp
copying apex\amp\handle.py -> build\lib\apex\amp
copying apex\amp\opt.py -> build\lib\apex\amp
copying apex\amp\rnn_compat.py -> build\lib\apex\amp
copying apex\amp\scaler.py -> build\lib\apex\amp
copying apex\amp\utils.py -> build\lib\apex\amp
copying apex\amp\wrap.py -> build\lib\apex\amp
copying apex\amp\_amp_state.py -> build\lib\apex\amp
copying apex\amp\_initialize.py -> build\lib\apex\amp
copying apex\amp\_process_optimizer.py -> build\lib\apex\amp
copying apex\amp\__init__.py -> build\lib\apex\amp
copying apex\amp\__version__.py -> build\lib\apex\amp
creating build\lib\apex\contrib
copying apex\contrib\__init__.py -> build\lib\apex\contrib
creating build\lib\apex\fp16_utils
copying apex\fp16_utils\fp16util.py -> build\lib\apex\fp16_utils
copying apex\fp16_utils\fp16_optimizer.py -> build\lib\apex\fp16_utils
copying apex\fp16_utils\loss_scaler.py -> build\lib\apex\fp16_utils
copying apex\fp16_utils\__init__.py -> build\lib\apex\fp16_utils
creating build\lib\apex\mlp
copying apex\mlp\mlp.py -> build\lib\apex\mlp
copying apex\mlp\__init__.py -> build\lib\apex\mlp
creating build\lib\apex\multi_tensor_apply
copying apex\multi_tensor_apply\multi_tensor_apply.py -> build\lib\apex\multi_tensor_apply
copying apex\multi_tensor_apply\__init__.py -> build\lib\apex\multi_tensor_apply
creating build\lib\apex\normalization
copying apex\normalization\fused_layer_norm.py -> build\lib\apex\normalization
copying apex\normalization\__init__.py -> build\lib\apex\normalization
creating build\lib\apex\optimizers
copying apex\optimizers\fused_adagrad.py -> build\lib\apex\optimizers
copying apex\optimizers\fused_adam.py -> build\lib\apex\optimizers
copying apex\optimizers\fused_lamb.py -> build\lib\apex\optimizers
copying apex\optimizers\fused_novograd.py -> build\lib\apex\optimizers
copying apex\optimizers\fused_sgd.py -> build\lib\apex\optimizers
copying apex\optimizers\__init__.py -> build\lib\apex\optimizers
creating build\lib\apex\parallel
copying apex\parallel\distributed.py -> build\lib\apex\parallel
copying apex\parallel\LARC.py -> build\lib\apex\parallel
copying apex\parallel\multiproc.py -> build\lib\apex\parallel
copying apex\parallel\optimized_sync_batchnorm.py -> build\lib\apex\parallel
copying apex\parallel\optimized_sync_batchnorm_kernel.py -> build\lib\apex\parallel
copying apex\parallel\sync_batchnorm.py -> build\lib\apex\parallel
copying apex\parallel\sync_batchnorm_kernel.py -> build\lib\apex\parallel
copying apex\parallel\__init__.py -> build\lib\apex\parallel
creating build\lib\apex\pyprof
copying apex\pyprof\__init__.py -> build\lib\apex\pyprof
creating build\lib\apex\reparameterization
copying apex\reparameterization\reparameterization.py -> build\lib\apex\reparameterization
copying apex\reparameterization\weight_norm.py -> build\lib\apex\reparameterization
copying apex\reparameterization\__init__.py -> build\lib\apex\reparameterization
creating build\lib\apex\RNN
copying apex\RNN\cells.py -> build\lib\apex\RNN
copying apex\RNN\models.py -> build\lib\apex\RNN
copying apex\RNN\RNNBackend.py -> build\lib\apex\RNN
copying apex\RNN\__init__.py -> build\lib\apex\RNN
creating build\lib\apex\amp\lists
copying apex\amp\lists\functional_overrides.py -> build\lib\apex\amp\lists
copying apex\amp\lists\tensor_overrides.py -> build\lib\apex\amp\lists
copying apex\amp\lists\torch_overrides.py -> build\lib\apex\amp\lists
copying apex\amp\lists\__init__.py -> build\lib\apex\amp\lists
creating build\lib\apex\contrib\bottleneck
copying apex\contrib\bottleneck\bottleneck.py -> build\lib\apex\contrib\bottleneck
copying apex\contrib\bottleneck\test.py -> build\lib\apex\contrib\bottleneck
copying apex\contrib\bottleneck\__init__.py -> build\lib\apex\contrib\bottleneck
creating build\lib\apex\contrib\fmha
copying apex\contrib\fmha\fmha.py -> build\lib\apex\contrib\fmha
copying apex\contrib\fmha\__init__.py -> build\lib\apex\contrib\fmha
creating build\lib\apex\contrib\groupbn
copying apex\contrib\groupbn\batch_norm.py -> build\lib\apex\contrib\groupbn
copying apex\contrib\groupbn\__init__.py -> build\lib\apex\contrib\groupbn
creating build\lib\apex\contrib\layer_norm
copying apex\contrib\layer_norm\layer_norm.py -> build\lib\apex\contrib\layer_norm
copying apex\contrib\layer_norm\__init__.py -> build\lib\apex\contrib\layer_norm
creating build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\encdec_multihead_attn.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\encdec_multihead_attn_func.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\fast_encdec_multihead_attn_func.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\fast_encdec_multihead_attn_norm_add_func.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\fast_self_multihead_attn_func.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\fast_self_multihead_attn_norm_add_func.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\mask_softmax_dropout_func.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\self_multihead_attn.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\self_multihead_attn_func.py -> build\lib\apex\contrib\multihead_attn
copying apex\contrib\multihead_attn\__init__.py -> build\lib\apex\contrib\multihead_attn
creating build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\distributed_fused_adam.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\distributed_fused_adam_v2.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\distributed_fused_adam_v3.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\distributed_fused_lamb.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\fp16_optimizer.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\fused_adam.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\fused_lamb.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\fused_sgd.py -> build\lib\apex\contrib\optimizers
copying apex\contrib\optimizers\__init__.py -> build\lib\apex\contrib\optimizers
creating build\lib\apex\contrib\sparsity
copying apex\contrib\sparsity\asp.py -> build\lib\apex\contrib\sparsity
copying apex\contrib\sparsity\sparse_masklib.py -> build\lib\apex\contrib\sparsity
copying apex\contrib\sparsity\__init__.py -> build\lib\apex\contrib\sparsity
creating build\lib\apex\contrib\transducer
copying apex\contrib\transducer\transducer.py -> build\lib\apex\contrib\transducer
copying apex\contrib\transducer\__init__.py -> build\lib\apex\contrib\transducer
creating build\lib\apex\contrib\xentropy
copying apex\contrib\xentropy\softmax_xentropy.py -> build\lib\apex\contrib\xentropy
copying apex\contrib\xentropy\__init__.py -> build\lib\apex\contrib\xentropy
creating build\lib\apex\pyprof\nvtx
copying apex\pyprof\nvtx\nvmarker.py -> build\lib\apex\pyprof\nvtx
copying apex\pyprof\nvtx\__init__.py -> build\lib\apex\pyprof\nvtx
creating build\lib\apex\pyprof\parse
copying apex\pyprof\parse\db.py -> build\lib\apex\pyprof\parse
copying apex\pyprof\parse\kernel.py -> build\lib\apex\pyprof\parse
copying apex\pyprof\parse\nvvp.py -> build\lib\apex\pyprof\parse
copying apex\pyprof\parse\parse.py -> build\lib\apex\pyprof\parse
copying apex\pyprof\parse\__init__.py -> build\lib\apex\pyprof\parse
copying apex\pyprof\parse\__main__.py -> build\lib\apex\pyprof\parse
creating build\lib\apex\pyprof\prof
copying apex\pyprof\prof\activation.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\base.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\blas.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\conv.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\convert.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\data.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\dropout.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\embedding.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\index_slice_join_mutate.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\linear.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\loss.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\misc.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\normalization.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\optim.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\output.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\pointwise.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\pooling.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\prof.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\randomSample.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\recurrentCell.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\reduction.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\softmax.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\usage.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\utility.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\__init__.py -> build\lib\apex\pyprof\prof
copying apex\pyprof\prof\__main__.py -> build\lib\apex\pyprof\prof
creating build\bdist.win-amd64
creating build\bdist.win-amd64\egg
creating build\bdist.win-amd64\egg\apex
creating build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\amp.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\compat.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\frontend.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\handle.py -> build\bdist.win-amd64\egg\apex\amp
creating build\bdist.win-amd64\egg\apex\amp\lists
copying build\lib\apex\amp\lists\functional_overrides.py -> build\bdist.win-amd64\egg\apex\amp\lists
copying build\lib\apex\amp\lists\tensor_overrides.py -> build\bdist.win-amd64\egg\apex\amp\lists
copying build\lib\apex\amp\lists\torch_overrides.py -> build\bdist.win-amd64\egg\apex\amp\lists
copying build\lib\apex\amp\lists\__init__.py -> build\bdist.win-amd64\egg\apex\amp\lists
copying build\lib\apex\amp\opt.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\rnn_compat.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\scaler.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\utils.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\wrap.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\_amp_state.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\_initialize.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\_process_optimizer.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\__init__.py -> build\bdist.win-amd64\egg\apex\amp
copying build\lib\apex\amp\__version__.py -> build\bdist.win-amd64\egg\apex\amp
creating build\bdist.win-amd64\egg\apex\contrib
creating build\bdist.win-amd64\egg\apex\contrib\bottleneck
copying build\lib\apex\contrib\bottleneck\bottleneck.py -> build\bdist.win-amd64\egg\apex\contrib\bottleneck
copying build\lib\apex\contrib\bottleneck\test.py -> build\bdist.win-amd64\egg\apex\contrib\bottleneck
copying build\lib\apex\contrib\bottleneck\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\bottleneck
creating build\bdist.win-amd64\egg\apex\contrib\fmha
copying build\lib\apex\contrib\fmha\fmha.py -> build\bdist.win-amd64\egg\apex\contrib\fmha
copying build\lib\apex\contrib\fmha\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\fmha
creating build\bdist.win-amd64\egg\apex\contrib\groupbn
copying build\lib\apex\contrib\groupbn\batch_norm.py -> build\bdist.win-amd64\egg\apex\contrib\groupbn
copying build\lib\apex\contrib\groupbn\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\groupbn
creating build\bdist.win-amd64\egg\apex\contrib\layer_norm
copying build\lib\apex\contrib\layer_norm\layer_norm.py -> build\bdist.win-amd64\egg\apex\contrib\layer_norm
copying build\lib\apex\contrib\layer_norm\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\layer_norm
creating build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\encdec_multihead_attn.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\encdec_multihead_attn_func.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\fast_encdec_multihead_attn_func.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\fast_encdec_multihead_attn_norm_add_func.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\fast_self_multihead_attn_func.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\fast_self_multihead_attn_norm_add_func.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\mask_softmax_dropout_func.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\self_multihead_attn.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\self_multihead_attn_func.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
copying build\lib\apex\contrib\multihead_attn\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\multihead_attn
creating build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\distributed_fused_adam.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\distributed_fused_adam_v2.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\distributed_fused_adam_v3.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\distributed_fused_lamb.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\fp16_optimizer.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\fused_adam.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\fused_lamb.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\fused_sgd.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
copying build\lib\apex\contrib\optimizers\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\optimizers
creating build\bdist.win-amd64\egg\apex\contrib\sparsity
copying build\lib\apex\contrib\sparsity\asp.py -> build\bdist.win-amd64\egg\apex\contrib\sparsity
copying build\lib\apex\contrib\sparsity\sparse_masklib.py -> build\bdist.win-amd64\egg\apex\contrib\sparsity
copying build\lib\apex\contrib\sparsity\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\sparsity
creating build\bdist.win-amd64\egg\apex\contrib\transducer
copying build\lib\apex\contrib\transducer\transducer.py -> build\bdist.win-amd64\egg\apex\contrib\transducer
copying build\lib\apex\contrib\transducer\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\transducer
creating build\bdist.win-amd64\egg\apex\contrib\xentropy
copying build\lib\apex\contrib\xentropy\softmax_xentropy.py -> build\bdist.win-amd64\egg\apex\contrib\xentropy
copying build\lib\apex\contrib\xentropy\__init__.py -> build\bdist.win-amd64\egg\apex\contrib\xentropy
copying build\lib\apex\contrib\__init__.py -> build\bdist.win-amd64\egg\apex\contrib
creating build\bdist.win-amd64\egg\apex\fp16_utils
copying build\lib\apex\fp16_utils\fp16util.py -> build\bdist.win-amd64\egg\apex\fp16_utils
copying build\lib\apex\fp16_utils\fp16_optimizer.py -> build\bdist.win-amd64\egg\apex\fp16_utils
copying build\lib\apex\fp16_utils\loss_scaler.py -> build\bdist.win-amd64\egg\apex\fp16_utils
copying build\lib\apex\fp16_utils\__init__.py -> build\bdist.win-amd64\egg\apex\fp16_utils
creating build\bdist.win-amd64\egg\apex\mlp
copying build\lib\apex\mlp\mlp.py -> build\bdist.win-amd64\egg\apex\mlp
copying build\lib\apex\mlp\__init__.py -> build\bdist.win-amd64\egg\apex\mlp
creating build\bdist.win-amd64\egg\apex\multi_tensor_apply
copying build\lib\apex\multi_tensor_apply\multi_tensor_apply.py -> build\bdist.win-amd64\egg\apex\multi_tensor_apply
copying build\lib\apex\multi_tensor_apply\__init__.py -> build\bdist.win-amd64\egg\apex\multi_tensor_apply
creating build\bdist.win-amd64\egg\apex\normalization
copying build\lib\apex\normalization\fused_layer_norm.py -> build\bdist.win-amd64\egg\apex\normalization
copying build\lib\apex\normalization\__init__.py -> build\bdist.win-amd64\egg\apex\normalization
creating build\bdist.win-amd64\egg\apex\optimizers
copying build\lib\apex\optimizers\fused_adagrad.py -> build\bdist.win-amd64\egg\apex\optimizers
copying build\lib\apex\optimizers\fused_adam.py -> build\bdist.win-amd64\egg\apex\optimizers
copying build\lib\apex\optimizers\fused_lamb.py -> build\bdist.win-amd64\egg\apex\optimizers
copying build\lib\apex\optimizers\fused_novograd.py -> build\bdist.win-amd64\egg\apex\optimizers
copying build\lib\apex\optimizers\fused_sgd.py -> build\bdist.win-amd64\egg\apex\optimizers
copying build\lib\apex\optimizers\__init__.py -> build\bdist.win-amd64\egg\apex\optimizers
creating build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\distributed.py -> build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\LARC.py -> build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\multiproc.py -> build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\optimized_sync_batchnorm.py -> build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\optimized_sync_batchnorm_kernel.py -> build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\sync_batchnorm.py -> build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\sync_batchnorm_kernel.py -> build\bdist.win-amd64\egg\apex\parallel
copying build\lib\apex\parallel\__init__.py -> build\bdist.win-amd64\egg\apex\parallel
creating build\bdist.win-amd64\egg\apex\pyprof
creating build\bdist.win-amd64\egg\apex\pyprof\nvtx
copying build\lib\apex\pyprof\nvtx\nvmarker.py -> build\bdist.win-amd64\egg\apex\pyprof\nvtx
copying build\lib\apex\pyprof\nvtx\__init__.py -> build\bdist.win-amd64\egg\apex\pyprof\nvtx
creating build\bdist.win-amd64\egg\apex\pyprof\parse
copying build\lib\apex\pyprof\parse\db.py -> build\bdist.win-amd64\egg\apex\pyprof\parse
copying build\lib\apex\pyprof\parse\kernel.py -> build\bdist.win-amd64\egg\apex\pyprof\parse
copying build\lib\apex\pyprof\parse\nvvp.py -> build\bdist.win-amd64\egg\apex\pyprof\parse
copying build\lib\apex\pyprof\parse\parse.py -> build\bdist.win-amd64\egg\apex\pyprof\parse
copying build\lib\apex\pyprof\parse\__init__.py -> build\bdist.win-amd64\egg\apex\pyprof\parse
copying build\lib\apex\pyprof\parse\__main__.py -> build\bdist.win-amd64\egg\apex\pyprof\parse
creating build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\activation.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\base.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\blas.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\conv.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\convert.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\data.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\dropout.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\embedding.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\index_slice_join_mutate.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\linear.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\loss.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\misc.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\normalization.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\optim.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\output.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\pointwise.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\pooling.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\prof.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\randomSample.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\recurrentCell.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\reduction.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\softmax.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\usage.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\utility.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\__init__.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\prof\__main__.py -> build\bdist.win-amd64\egg\apex\pyprof\prof
copying build\lib\apex\pyprof\__init__.py -> build\bdist.win-amd64\egg\apex\pyprof
creating build\bdist.win-amd64\egg\apex\reparameterization
copying build\lib\apex\reparameterization\reparameterization.py -> build\bdist.win-amd64\egg\apex\reparameterization
copying build\lib\apex\reparameterization\weight_norm.py -> build\bdist.win-amd64\egg\apex\reparameterization
copying build\lib\apex\reparameterization\__init__.py -> build\bdist.win-amd64\egg\apex\reparameterization
creating build\bdist.win-amd64\egg\apex\RNN
copying build\lib\apex\RNN\cells.py -> build\bdist.win-amd64\egg\apex\RNN
copying build\lib\apex\RNN\models.py -> build\bdist.win-amd64\egg\apex\RNN
copying build\lib\apex\RNN\RNNBackend.py -> build\bdist.win-amd64\egg\apex\RNN
copying build\lib\apex\RNN\__init__.py -> build\bdist.win-amd64\egg\apex\RNN
copying build\lib\apex\__init__.py -> build\bdist.win-amd64\egg\apex
byte-compiling build\bdist.win-amd64\egg\apex\amp\amp.py to amp.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\compat.py to compat.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\frontend.py to frontend.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\handle.py to handle.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\lists\functional_overrides.py to functional_overrides.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\lists\tensor_overrides.py to tensor_overrides.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\lists\torch_overrides.py to torch_overrides.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\lists\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\opt.py to opt.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\rnn_compat.py to rnn_compat.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\scaler.py to scaler.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\utils.py to utils.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\wrap.py to wrap.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\_amp_state.py to _amp_state.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\_initialize.py to _initialize.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\_process_optimizer.py to _process_optimizer.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\amp\__version__.py to __version__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\bottleneck\bottleneck.py to bottleneck.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\bottleneck\test.py to test.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\bottleneck\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\fmha\fmha.py to fmha.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\fmha\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\groupbn\batch_norm.py to batch_norm.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\groupbn\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\layer_norm\layer_norm.py to layer_norm.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\layer_norm\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\encdec_multihead_attn.py to encdec_multihead_attn.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\encdec_multihead_attn_func.py to encdec_multihead_attn_func.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\fast_encdec_multihead_attn_func.py to fast_encdec_multihead_attn_func.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\fast_encdec_multihead_attn_norm_add_func.py to fast_encdec_multihead_attn_norm_add_func.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\fast_self_multihead_attn_func.py to fast_self_multihead_attn_func.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\fast_self_multihead_attn_norm_add_func.py to fast_self_multihead_attn_norm_add_func.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\mask_softmax_dropout_func.py to mask_softmax_dropout_func.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\self_multihead_attn.py to self_multihead_attn.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\self_multihead_attn_func.py to self_multihead_attn_func.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\multihead_attn\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\distributed_fused_adam.py to distributed_fused_adam.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\distributed_fused_adam_v2.py to distributed_fused_adam_v2.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\distributed_fused_adam_v3.py to distributed_fused_adam_v3.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\distributed_fused_lamb.py to distributed_fused_lamb.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\fp16_optimizer.py to fp16_optimizer.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\fused_adam.py to fused_adam.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\fused_lamb.py to fused_lamb.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\fused_sgd.py to fused_sgd.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\optimizers\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\sparsity\asp.py to asp.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\sparsity\sparse_masklib.py to sparse_masklib.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\sparsity\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\transducer\transducer.py to transducer.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\transducer\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\xentropy\softmax_xentropy.py to softmax_xentropy.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\xentropy\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\contrib\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\fp16_utils\fp16util.py to fp16util.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\fp16_utils\fp16_optimizer.py to fp16_optimizer.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\fp16_utils\loss_scaler.py to loss_scaler.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\fp16_utils\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\mlp\mlp.py to mlp.cpython-39.pyc
build\bdist.win-amd64\egg\apex\mlp\mlp.py:40: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if activation is 'none':
build\bdist.win-amd64\egg\apex\mlp\mlp.py:42: SyntaxWarning: "is" with a literal. Did you mean "=="?
  elif activation is 'relu':
build\bdist.win-amd64\egg\apex\mlp\mlp.py:44: SyntaxWarning: "is" with a literal. Did you mean "=="?
  elif activation is 'sigmoid':
byte-compiling build\bdist.win-amd64\egg\apex\mlp\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\multi_tensor_apply\multi_tensor_apply.py to multi_tensor_apply.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\multi_tensor_apply\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\normalization\fused_layer_norm.py to fused_layer_norm.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\normalization\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\optimizers\fused_adagrad.py to fused_adagrad.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\optimizers\fused_adam.py to fused_adam.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\optimizers\fused_lamb.py to fused_lamb.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\optimizers\fused_novograd.py to fused_novograd.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\optimizers\fused_sgd.py to fused_sgd.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\optimizers\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\distributed.py to distributed.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\LARC.py to LARC.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\multiproc.py to multiproc.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\optimized_sync_batchnorm.py to optimized_sync_batchnorm.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\optimized_sync_batchnorm_kernel.py to optimized_sync_batchnorm_kernel.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\sync_batchnorm.py to sync_batchnorm.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\sync_batchnorm_kernel.py to sync_batchnorm_kernel.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\parallel\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\nvtx\nvmarker.py to nvmarker.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\nvtx\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\parse\db.py to db.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\parse\kernel.py to kernel.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\parse\nvvp.py to nvvp.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\parse\parse.py to parse.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\parse\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\parse\__main__.py to __main__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\activation.py to activation.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\base.py to base.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\blas.py to blas.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\conv.py to conv.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\convert.py to convert.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\data.py to data.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\dropout.py to dropout.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\embedding.py to embedding.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\index_slice_join_mutate.py to index_slice_join_mutate.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\linear.py to linear.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\loss.py to loss.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\misc.py to misc.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\normalization.py to normalization.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\optim.py to optim.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\output.py to output.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\pointwise.py to pointwise.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\pooling.py to pooling.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\prof.py to prof.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\randomSample.py to randomSample.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\recurrentCell.py to recurrentCell.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\reduction.py to reduction.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\softmax.py to softmax.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\usage.py to usage.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\utility.py to utility.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\prof\__main__.py to __main__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\pyprof\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\reparameterization\reparameterization.py to reparameterization.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\reparameterization\weight_norm.py to weight_norm.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\reparameterization\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\RNN\cells.py to cells.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\RNN\models.py to models.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\RNN\RNNBackend.py to RNNBackend.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\RNN\__init__.py to __init__.cpython-39.pyc
byte-compiling build\bdist.win-amd64\egg\apex\__init__.py to __init__.cpython-39.pyc
creating build\bdist.win-amd64\egg\EGG-INFO
copying apex.egg-info\PKG-INFO -> build\bdist.win-amd64\egg\EGG-INFO
copying apex.egg-info\SOURCES.txt -> build\bdist.win-amd64\egg\EGG-INFO
copying apex.egg-info\dependency_links.txt -> build\bdist.win-amd64\egg\EGG-INFO
copying apex.egg-info\top_level.txt -> build\bdist.win-amd64\egg\EGG-INFO
zip_safe flag not set; analyzing archive contents...
creating dist
creating 'dist\apex-0.1-py3.9.egg' and adding 'build\bdist.win-amd64\egg' to it
removing 'build\bdist.win-amd64\egg' (and everything under it)
Processing apex-0.1-py3.9.egg
Copying apex-0.1-py3.9.egg to d:\python39\lib\site-packages
Removing apex 0.1 from easy-install.pth file
Adding apex 0.1 to easy-install.pth file

Installed d:\python39\lib\site-packages\apex-0.1-py3.9.egg
Processing dependencies for apex==0.1
Finished processing dependencies for apex==0.1