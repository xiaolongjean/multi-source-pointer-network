# 多源指针网络生成短标题
# Multi-Source Pointer Network

------



### 一、模型说明：

:pray: 基于论文"**Multi-Source Pointer Network for Product Title Summarization**"开发，模型的具体介绍以及短标题的生成参考： [https://mp.weixin.qq.com/s/5rAM44D50JHE-q1IrLEatw](https://mp.weixin.qq.com/s/5rAM44D50JHE-q1IrLEatw "https://mp.weixin.qq.com/s/5rAM44D50JHE-q1IrLEatw")，在论文的基础上进行了如下几方面的优化：

- :one: 融合词语和字符特征，解决测试阶段多个OOV词汇共享同一个Embedding的问题。

- :two: 针对不同特征，编码器采用多组Transformer，然后对特征进行了融合。

- :three: 推理阶段采用局部拷贝机制，解决全局拷贝无法解决#UNK#的情形。

- :four: Beam Search阶段对初始分布添加mask，防止一开始生成 '#EOS#' 这一token。

- :recycle: 如果采用GPU，所有batch的数据会一次性加载到GPU中，可以修改data.py分批加载。

  

### 二、网络结构：

**模型编码器部分的整体结构如下图所示，解码器部分后面会进行补充。**

![encoder](figs/encoder.jpg)



**模型在推理阶段采用了局部拷贝机制，将每个训练样本的source1和source2的tokens动态地映射成局部id，计算token概率分布，并按照局部token id进行token的概率合并。**

Step-1: 对于每个训练样本，利用两个输入源的tokens来动态构地造局部的token词典。

Step-2: 基于该token词典，分别将两个输入源的tokens序列映射成局部id序列。

Step-3: 分别得到两个输入源的所有tokens关于解码器的概率分布source1 probs和source2 probs。

Step-4: 将两个输入源的概率分布进行merge，并取概率最大的token id。

Step-5: 利用Step-1构造的局部动态词典将概率最大的token id映射成对应的token，得到当前时刻的预测token。

![inference](/figs/inference.jpg)





### 三、运行方式：

1. #### 数据格式：

   - 训练/验证数据：训练数据和验证数据分别有3列，即：source1，source2和target。
   - 在数据预处理阶段过滤了字包含英文、数字、标点等token。
   - 测试数据：只需要两列，即source1，source2。
   - 需要注意的是：两个source和target的数据需要进行分词，词语之间用空格分隔。

2. #### 模型训练：

   - **训练阶段**利用到的是 `main.py` 脚本中的 `train()` 函数，训练阶段可以根据情况把 `main.py` 脚本中的 `test()` 函数注释掉。

   - 准备好训练数据后按照下面的示例进行训练，或者修改 `config.py` 中相关训练数据的路径，然后直接执行 `python3 main.py`。**如果需要使用GPU，参考`config.py` 中的相关参数。**

     ```python
     python3 main.py --train_data_path ../data/train.dat \
                     --valid_data_path ../data/valid.dat \
                     --model_dir ../model/ \
                     --max_epoch 50
     ```

3. #### 模型预测：

   - 预测阶段利用到的是 `main.py` 脚本中的 `test()` 函数，预测阶段需要把 `train()` 函数注释掉。
   - 预测阶段，可以参考 `main.py` 脚本中给定的测试数据格式、模型加载方式等进行预测。
   - 需要注意的是：模型预测阶段也是以batch的方式进行的，因此在预测时需要参考 `main.py` 脚本中的 `test()` 函数，将数据准备成batch的形式。

   

### 四、注意事项：

- 项目的data目录下给定的是demo数据，是为了测试模型训练和预测的流程，由于demo数据的量很少，所以训练时打印的**mean_valid_loss**可能会随着训练次数变大，或者出现BLEU值为0的情况，在训练数据足够的情况下，不会出现此种情况。




# 🚩TODO：

- 在训练阶段会保存每个epoch的模型参数，后期可以只保留top n个准确率比较高的模型。
- target和输入数据共享了embedding，训练阶段容易发生震荡，以后可以考虑尝试对输入数据和target采用不同的embedding。

