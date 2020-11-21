from keras.utils.vis_utils import plot_model
from models import makeModel

model = makeModel('noiseNet001')
plot_model(model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = False)