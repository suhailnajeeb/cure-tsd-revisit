from keras.layers import Input
from keras.layers import Input, Conv2D, concatenate
from keras.models import Model

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

