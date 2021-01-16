

from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet169

class UpscaleBlock(Model):
    def __init__(self, filters):      
        super(UpscaleBlock, self).__init__()
        self.up = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.concat = Concatenate() # for the skip connection    
        self.conv1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        self.conv2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        self.relu1 = LeakyReLU(alpha=0.2)
        self.relu2 = LeakyReLU(alpha=0.2)
    
    def call(self, x):        
        return self.relu2(self.conv2(self.relu1(self.conv1(self.concat([self.up(x[0]), x[1]])))))


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()                
        self.base_model = DenseNet169(include_top=False, input_shape=(None, None, 3), weights='imagenet')   
        
        outputs = [self.base_model.outputs[-1]]
        for name in ['pool1', 'pool2_pool', 'pool3_pool', 'conv1/relu']: 
            outputs.append( self.base_model.get_layer(name).output )        
        self.encoder = Model(inputs=self.base_model.inputs, outputs=outputs) #create a model w preset inputs and outputs from densenet-169
        
    def call(self, x):
        return self.encoder(x)
    
class Decoder(Model):
    def __init__(self, decoder_filter):
        super(Decoder, self).__init__()        
        self.conv2 =  Conv2D(filters=decoder_filter, kernel_size=1, padding='same')        
        self.up1 = UpscaleBlock(filters=decoder_filter//2)
        self.up2 = UpscaleBlock(filters=decoder_filter//4)
        self.up3 = UpscaleBlock(filters=decoder_filter//8)
        self.up4 = UpscaleBlock(filters=decoder_filter//16)       
        self.conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same') #makes depth mapth w/ dimensionality = 1

    def call(self, features):        
        x, pool1, pool2, pool3, conv1 = features[0], features[1], features[2], features[3], features[4]
        up0 = self.conv2(x)        
        up1 = self.up1([up0, pool3])        
        up2 = self.up2([up1, pool2])        
        up3 = self.up3([up2, pool1])        
        up4 = self.up4([up3, conv1])        
        return self.conv3(up4)
    
class Depth(Model):
    def __init__(self):
        super(Depth, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(decoder_filter = int(self.encoder.layers[-1].output[0].shape[-1] // 2 ))
        print('\nModel created.')

    def call(self, x):
        return self.decoder( self.encoder(x) )
