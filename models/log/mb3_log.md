1）类似于v1和v2版本，采用pw-dw-pw模式，类似于resnet的残差模块shortcut模式

2）采用relu6/hardswish/hardsigmoid激活函数

3）采用_squeeze_excitation_layer(类似于SENet，attention机制，avgpool--fc(relu6)--fc(hard-sigmoid)，得到权重参数，乘到input上)

4)通过mobilenet_block循环进行，每个block包含**pointwise**，hardswish/relu6激活，**depthwise**，bn，hardswish/relu6激活，Se模块(上面3，可选)，pointwise，shortcut(类似resnet，可选模块)