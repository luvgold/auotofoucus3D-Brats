import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential
import tensorflow as tf

class BasicBlock(layers.Layer):
    def __init__(self, inplanes1, outplanes1, outplanes2, kernel=3, activation=None,**kwargs):
        if kwargs.get("padding") and kwargs["padding"].upper() != "SAME":
            raise NotImplementedError("Only implemented for padding 'SAME'")
        kwargs["padding"] = "SAME"
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv3D(outplanes1,kernel_size=kernel,dilation_rate=2,padding="SAME")
        # 残差块的第一个卷积层
        self.bn1 = layers.BatchNormalization()
        # 将卷积层输出的数据批量归一化
        self.relu = layers.ReLU()
        # import归一化后进行线性计算
        self.conv2 = layers.Conv3D(outplanes2,kernel_size=kernel,dilation_rate=2,padding="SAME")
        # 残差块的第二个卷积层
        self.bn2 = layers.BatchNormalization()
        # 将卷积层输出的数据批量归一化
        print("test_point")
        if inplanes1 == outplanes2:
            self.downsample = lambda x:x
        else:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv3D(outplanes2, kernel_size=1))
            self.downsample.add(layers.BatchNormalization())

    def call(self, inputs, training= None):
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out , training=training)
        identity = self.downsample(inputs)
        output = layers.add([out,identity])
        output = tf.nn.relu(output)

        return output
