import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import yaml

def make_optimizer(optimizer_dict):

    optim_fun = optimizer_dict.pop('Name')
    fun = getattr(tf.keras.optimizers, optim_fun)

    return fun(**optimizer_dict)

def make_losses(losses_dict):
    loss_fun = losses_dict.pop('Name')
    fun = getattr(tf.keras.losses, loss_fun)

    return fun(**losses_dict)

def make_layer(layer_dict):
    layer_fun = layer_dict.pop('Layer')
    fun = getattr(tf.keras.layers, layer_fun)

    return fun(**layer_dict)

def make_model(yaml_file):
    yaml_file = yaml.safe_load(open(yaml_file, 'r'))

    #learning_rate = yaml_file['Learning_Rate']

    model_layers = []
    total_layer_str = ''
    for i, layer_dict in enumerate(yaml_file['Layer_Architecture']):
        
        name = layer_dict['Layer']
        model_layers.append(make_layer(layer_dict))

        layer_str = ['{}: {}'.format(layer_name, value) for layer_name, value in layer_dict.items()]
        layer_str = 'Layer {:d} - {}\n\t{}'.format(i, name, '\n\t'.join(layer_str))

        total_layer_str += '\n{}'.format(layer_str)
    print(total_layer_str)
    optimizer = make_optimizer(yaml_file['Optimizer'])
    losses = make_losses(yaml_file['Losses'])
    model = models.Sequential(model_layers)
    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics=['accuracy'])

    return model

