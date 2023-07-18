# machine-learning-template
# for maximising GPU memory
gpu =  '1' 

os.environ["CUDA_VISIBLE_DEVICES"]= gpu

gpus= tf.config.list_physical_devices('GPU') 

print(gpus) 

tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])

strategy = tf.distribute.OneDeviceStrategy("/gpu:%s"%gpu)
