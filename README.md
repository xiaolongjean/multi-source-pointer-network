# 多源指针网络生成短标题
# Multi-Source Pointer Network

------



### 一、模型说明：

:pray: 基于论文"**Multi-Source Pointer Network for Product Title Summarization**"开发，模型的具体介绍以及短标题的生成参考： [https://mp.weixin.qq.com/s/5rAM44D50JHE-q1IrLEatw](https://mp.weixin.qq.com/s/5rAM44D50JHE-q1IrLEatw "https://mp.weixin.qq.com/s/5rAM44D50JHE-q1IrLEatw")，在论文的基础上进行了如下几方面的优化：

- :one: 融合多特征，解决测试阶段多个OOV词汇共享同一个Embedding的问题。

- :two: 针对不同特征，编码器采用多组Transformer，然后对特征进行了融合。

- :three: 推理阶段采用局部拷贝机制，解决全局拷贝无法解决#UNK#的情形。

- :four: Beam Search阶段对初始分布添加mask，防止一开始生成 '#EOS#' 这一token。

- :recycle: 如果采用GPU，所有batch的数据会一次性加载到GPU中，可以修改data.py分批加载。

  ​



### 二、运行方式：

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
                     --test_data_path ../data/test.dat \
                     --model_dir ../model/ \
                     --max_epoch 50
     ```

3. #### 模型预测：

   - 预测阶段利用到的是 `main.py` 脚本中的 `test()` 函数，训练阶段需要把 `train()` 函数注释掉。。
   - 预测阶段，可以参考 `main.py` 脚本中给定的测试数据格式、模型加载方式等进行预测。
   - 需要注意的是：模型预测阶段也是以batch的方式进行的，因此在预测时需要参考 `main.py` 脚本中的 `test()` 函数，将数据准备成batch的形式。

   ​

### 三、注意事项：

- 项目的data目录下给定的是demo数据，是为了测试模型训练和预测的流程，由于demo数据的量很少，所以训练时打印的**mean_valid_loss**可能会随着训练次数变大，或者出现BLEU值为0的情况，在训练数据足够的情况下，不会出现此种情况。




# 🚩TODO：

- 在训练阶段会保存每个epoch的模型参数，后期可以只保留top n个准确率比较高的模型。
- target和输入数据共享了embedding，训练阶段容易发生震荡，以后可以考虑尝试对输入数据和target采用不同的embedding。

