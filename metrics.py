import tensorflow as tf
def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def psnr_tf(im1, im2):
  return tf.image.psnr(im1, im2, max_val = 255)