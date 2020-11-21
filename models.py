from keras.layers import Input
from keras.layers import Input, Conv2D, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape, Conv2D, LocallyConnected2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,  SpatialDropout2D, Cropping2D, ZeroPadding2D, GlobalMaxPooling2D
from keras.layers import Lambda, Input, Dense, Dropout, Flatten, Activation
from keras.layers.merge import add, concatenate, average, multiply

modelArch = {}
addModel = lambda f : modelArch.setdefault(f.__name__, f)

def makeModel(architecture, verbose = True):
    model = modelArch[architecture]()
    if verbose:
        print(model.summary(line_length = 120))
    return model

@addModel
def dummyModel(img_shape = (128, 128, 3)):
    x = Input(shape = img_shape)
    cc = Conv2D(1, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
    o = Conv2D(3, kernel_size = (3,3), activation = 'relu', padding = 'same')(cc)
    return Model(inputs = x, outputs = o)

@addModel
def noiseNet001():
	in1 = Input((128,128,3))

	#encoder
	stack1E = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same',data_format='channels_last')(in1)
	stack1E = BatchNormalization()(stack1E)
	
	stack2E = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack1E)
	stack2E = BatchNormalization()(stack2E)
	stack2E = MaxPooling2D(pool_size=(2, 2))(stack2E)

	stack3E = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same',data_format='channels_last')(stack2E)
	stack3E = BatchNormalization()(stack3E)

	stack4E = Conv2D(128, (3, 3), strides=1, activation='relu', padding='same',data_format='channels_last')(stack3E)
	stack4E = BatchNormalization()(stack4E)
	stack4E = MaxPooling2D(pool_size=(2, 2))(stack4E)

	stack5E = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4E)
	stack5E = BatchNormalization()(stack5E)
	stack5E = MaxPooling2D(pool_size=(2, 2))(stack5E)

	#decoder
	stack5D = Dropout(0.5)(stack5E)
	stack5D = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack5D)
	# stack5D = add([stack5D,stack5E])

	stack4D = UpSampling2D((2, 2))(stack5D)
	stack4D = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack4D)
	stack4D = add([stack4D,stack4E])
	stack4D = BatchNormalization()(stack4D)

	stack3D = UpSampling2D((2, 2))(stack4D)
	stack3D = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack3D)
	stack3D = add([stack3D,stack3E])
	stack3D = BatchNormalization()(stack3D)
	
	stack2D = UpSampling2D((2, 2))(stack3D)
	stack2D = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack2D)
	stack2D = BatchNormalization()(stack2D)

	stack1D = UpSampling2D((2, 2))(stack2D)
	stack1D = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_last')(stack1D)
	stack1D = BatchNormalization()(stack1D)

	stack0D = UpSampling2D((2, 2))(stack1D)
	stack0D = Conv2D(3, (3, 3), activation='sigmoid', padding='same',data_format='channels_last')(stack0D)

	return Model(inputs=in1, outputs=stack0D)