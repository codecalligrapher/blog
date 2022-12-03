---
title: "TensorFlow Custom Loop"
date: 2022-11-30T00:38:42-04:00
ShowCodeCopyButtons: true
keywords: ["machine-learning", "tensorflow"]
comments: true
showReadingTime: false
draft: false 
---

A pattern I use for pretty-progress bars, custom logging and metric-handling in `tensorflow`.

```python
'''
This overviews the TensorFlow custom training loop in its (what I think is) most general sense. Four steps:
    1. Define Model
    2. Define Metrics, Optimizer and Losses
    3. Define train and test (validation) functions
    4. Write training loop
'''
import wandb
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import enlighten
import logging
import time
from dataclasses import dataclass



# STEP 0 - Set up datasets

'''
---------------------------------------------------------
STEP 1 - Define Model 

Define inputs, outputs and wrap using keras
'''
inputs = ...
outputs = ...
model = keras.Model(inputs=inputs, outputs=outputs)

'''
---------------------------------------------------------
STEP 2 - Define Metrics, Optimizer and Losses

Use keras.metrics.Metric and keras.optimizers
Can subclass if necessary

'''
train_metric = keras.metrics...
val_metric = keras.metrics...

optimizer = keras.optimizers...

loss_fn = keras.losses...


'''
---------------------------------------------------------
STEP 3 - Define training and test functions 

both take inputs and labels
both return a loss value

training invoke tape and applies loss gradient to weights
test just finds loss value

'''

@tf.function
def train_step(input, labels):
    # invoke GradientTape()
    with tf.GradientTape() as tape:
        # find predicted
        pred = model(input, training=True) 
        # calculate loss
        loss_value = loss_fn(labels, pred)
        loss_value += sum(model.losses)

    # find gradient loss and weights
    grads = tape.gradient(loss_value, model.trainable_weights)
    # apply gradients to update weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # update metric
    train_metric.update_state(labels, pred)

    return loss_value

@tf.function
def test_step(input, labels):
    # find predicted
    pred = model(input, training=False)
    # update metric
    val_metric.update_state(labels, pred)


'''
---------------------------------------------------------
STEP 4 - Training/Validation Loop

Remember to reset metric states

'''

#-------------------------------CONFIGURATION---------------------------------#
@dataclass
class ModelConf:
    epochs: int
    batch_size: int
    learning_rate: float
    dropout_rate: float

wandb_config = {
    "entity": "aadi350",
    "project": "urban_heat_index",
    "model_name": "vit",
    ...
}

with open('path/to/conf.yaml', 'r') as f:
    conf_file = yaml.load(f, Loader=yaml.SafeLoader)
    m = ModelConf(
        conf_file['epochs'],
        conf_file['batch_size'],
        conf_file['learning_rate'],
        conf_file['dropout_rate'],
        ...
    )

    wandb_config.update(
       conf_file 
    )
    wandb.init(config=wandb_config)

    m.epochs = conf_file['epochs']
    m.batch_size = conf_file['batch_size']
    m.learning_rate = conf_file['learning_rate']
    ...
#-----------------------------END CONFIGURATION-------------------------------#



log_batch = 10

# Set up logging
mlog = logging.getLogger("metric_logger")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
mlog.addHandler(handler)
mlog.setLevel(logging.DEBUG)

# set up progress bar for epochs
manager = enlighten.get_manager()
epoch_counter = manager.counter(
    total=m.epochs, desc="Epoch", unit="epochs", color="green"
)
status_bar = manager.status_bar(
    "Best metrics", color="white_on_blue", justify=enlighten.Justify.CENTER
)

for epoch in range(m.epochs):
    # progress bar for train step
    step_counter = manager.counter(
        total=m.epochs//m.batch_size
        desc="Train Step",
        unit="steps",
        leave=False,
        color="bright_green",
    )
    # ---------------------TRAIN------------------------#
    for step, (in_batch_train, label_batch_train) in enumerate(train_dataset):
        loss_value = train_step(in_batch_train, label_batch_train)
        step_counter.update()


    train_acc = train_metric.result()
    train_loss = loss_value
    mlog.info("Training acc over epoch: %.4f" % (float(train_acc),))
    train_metric.reset_states()
    step_counter.close()

    # ---------------------VAL---------------------------#
    # progress bar for validation step
    step_counter = manager.counter(
        total=m.epochs//m.batch_size,
        desc="Validation Step",
        unit="steps",
        leave=False,
        color="bright_yellow",
    )
    # validation loop 
    for step, (in_batch_val, label_batch_val) in enumerate(val_dataset):
        test_step(in_batch_val, label_batch_val)
    
    val_acc = val_metric.result()
    val_metric.reset_states()
    step_counter.close()

    # -------------------STATUS-----------------------------#
    if best_val < val_acc:
        best_val = val_acc
        status_bar.update(f"Best validation metric: {best_val}, epoch: {epoch}")
        tf.saved_model.save('path/to/model')
    epoch_counter.update()

    # -------------------WANDB------------------------------#
    wandb.log(
        {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        }
    )

```
