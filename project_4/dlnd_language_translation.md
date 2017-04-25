
# Language Translation
In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028
    
    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .
    
    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .


## Implement Preprocessing Function
### Text to Word Ids
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of each sentence from `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    # TODO: Implement Function
    source_id_text = [[source_vocab_to_int[word] 
                       for word in (sentence).split()] 
                      for sentence in source_text.split("\n")]
    target_id_text = [[target_vocab_to_int[word] 
                       for word in (sentence+" <EOS>").split()] 
                      for sentence in target_text.split("\n")]
    return source_id_text, target_id_text

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    Tests Passed


### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) in [LooseVersion('1.0.0'), LooseVersion('1.0.1')], 'This project requires TensorFlow version 1.0  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.1
    Default GPU Device: /gpu:0


## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoding_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.

Return the placeholders in the following the tuple (Input, Targets, Learing Rate, Keep Probability)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate, keep probability)
    """
    # TODO: Implement Function
    inputs=tf.placeholder(tf.int32,[None,None],name="input")
    targets=tf.placeholder(tf.int32,[None,None],name="targets")
    learning_rate=tf.placeholder(tf.float32,name="learning_rate")
    keep_prob=tf.placeholder(tf.float32,name="keep_prob")
    return inputs, targets, learning_rate, keep_prob

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Process Decoding Input
Implement `process_decoding_input` using TensorFlow to remove the last word id from each batch in `target_data` and concat the GO ID to the beginning of each batch.


```python
def process_decoding_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    # TODO: Implement Function
    ending = tf.strided_slice(target_data, [0, 0], 
                              [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], 
                                   target_vocab_to_int['<GO>']), 
                           ending], 1)
    return dec_input

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_decoding_input(process_decoding_input)
```

    Tests Passed


### Encoding
Implement `encoding_layer()` to create a Encoder RNN layer using [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn).


```python
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """
    # TODO: Implement Function
    encoder_cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    encoder_cell = tf.contrib.rnn.DropoutWrapper(
        encoder_cell, output_keep_prob=keep_prob)
    _, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, rnn_inputs, dtype=tf.float32)
    return encoder_state



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed


### Decoding - Training
Create training logits using [`tf.contrib.seq2seq.simple_decoder_fn_train()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_train) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder).  Apply the `output_fn` to the [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder) outputs.


```python
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param sequence_length: Sequence Length
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Train Logits
    """
    # TODO: Implement Function
    train_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(
        encoder_state)
    dec_cell = tf.contrib.rnn.DropoutWrapper(
        dec_cell, output_keep_prob=keep_prob)
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell, train_decoder_fn, dec_embed_input, sequence_length,
        scope=decoding_scope)
    # Apply output function
    train_logits =  output_fn(train_pred)
    return train_logits 


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed


### Decoding - Inference
Create inference logits using [`tf.contrib.seq2seq.simple_decoder_fn_inference()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/simple_decoder_fn_inference) and [`tf.contrib.seq2seq.dynamic_rnn_decoder()`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_rnn_decoder). 


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param maximum_length: The maximum allowed time steps to decode
    :param vocab_size: Size of vocabulary
    :param decoding_scope: TensorFlow Variable Scope for decoding
    :param output_fn: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: Inference Logits
    """
    # TODO: Implement Function
    inference_decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_inference(
        output_fn, encoder_state, dec_embeddings, start_of_sequence_id, end_of_sequence_id, 
        maximum_length - 1, vocab_size)
    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, inference_decoder_fn, scope=decoding_scope)
    
    return inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

- Create RNN cell for decoding using `rnn_size` and `num_layers`.
- Create the output fuction using [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) to transform it's input, logits, to class logits.
- Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob)` function to get the training logits.
- Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size, decoding_scope, output_fn, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, target_vocab_to_int, keep_prob):
    """
    Create decoding layer
    :param dec_embed_input: Decoder embedded input
    :param dec_embeddings: Decoder embeddings
    :param encoder_state: The encoded state
    :param vocab_size: Size of vocabulary
    :param sequence_length: Sequence Length
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param keep_prob: Dropout keep probability
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)] * num_layers)
    
    with tf.variable_scope("decoding") as decoding_scope:
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size, None, scope=decoding_scope)
        train_logits = decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, 
                                            output_fn, keep_prob)
    with tf.variable_scope("decoding", reuse=True) as decoding_scope:
        inference_logits = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'], 
                                                target_vocab_to_int['<EOS>'], sequence_length, vocab_size, decoding_scope, 
                                                output_fn, keep_prob)
    return train_logits, inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:

- Apply embedding to the input data for the encoder.
- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob)`.
- Process target data using your `process_decoding_input(target_data, target_vocab_to_int, batch_size)` function.
- Apply embedding to the target data for the decoder.
- Decode the encoded input using your `decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)`.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size, rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param sequence_length: Sequence Length
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training Logits, Inference Logits)
    """
    # TODO: Implement Function
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, enc_embedding_size)
    encoder_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob)
    dec_input = process_decoding_input(target_data, target_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    train_logits, inference_logits = decoding_layer(dec_embed_input, dec_embeddings, encoder_state, target_vocab_size, 
                                                    sequence_length, rnn_size, num_layers, target_vocab_to_int, keep_prob)
    return train_logits, inference_logits


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability


```python
# Number of Epochs
epochs = 64
# Batch Size
batch_size = 6072
# RNN Size
rnn_size = 128 
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 32
decoding_embedding_size = 32
# Learning Rate
learning_rate = 0.003
# Dropout Keep Probability
keep_probability = 0.7
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_source_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob = model_inputs()
    sequence_length = tf.placeholder_with_default(max_source_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    
    train_logits, inference_logits = seq2seq_model(
        tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(source_vocab_to_int), len(target_vocab_to_int),
        encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, target_vocab_to_int)

    tf.identity(inference_logits, 'logits')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            train_logits,
            targets,
            tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
```

### Train
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import time

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1]), (0,0)],
            'constant')

    return np.mean(np.equal(target, np.argmax(logits, 2)))

train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]

valid_source = helper.pad_sentence_batch(source_int_text[:batch_size])
valid_target = helper.pad_sentence_batch(target_int_text[:batch_size])

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch) in enumerate(
                helper.batch_data(train_source, train_target, batch_size)):
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 sequence_length: target_batch.shape[1],
                 keep_prob: keep_probability})
            
            batch_train_logits = sess.run(
                inference_logits,
                {input_data: source_batch, keep_prob: 1.0})
            batch_valid_logits = sess.run(
                inference_logits,
                {input_data: valid_source, keep_prob: 1.0})
                
            train_acc = get_accuracy(target_batch, batch_train_logits)
            valid_acc = get_accuracy(np.array(valid_target), batch_valid_logits)
            end_time = time.time()
            print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.3f}, Validation Accuracy: {:>6.3f}, Loss: {:>6.3f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/22 - Train Accuracy:  0.306, Validation Accuracy:  0.306, Loss:  5.915
    Epoch   0 Batch    1/22 - Train Accuracy:  0.307, Validation Accuracy:  0.306, Loss:  5.642
    Epoch   0 Batch    2/22 - Train Accuracy:  0.309, Validation Accuracy:  0.306, Loss:  5.247
    Epoch   0 Batch    3/22 - Train Accuracy:  0.310, Validation Accuracy:  0.306, Loss:  4.734
    Epoch   0 Batch    4/22 - Train Accuracy:  0.305, Validation Accuracy:  0.306, Loss:  4.336
    Epoch   0 Batch    5/22 - Train Accuracy:  0.309, Validation Accuracy:  0.306, Loss:  4.048
    Epoch   0 Batch    6/22 - Train Accuracy:  0.336, Validation Accuracy:  0.335, Loss:  3.899
    Epoch   0 Batch    7/22 - Train Accuracy:  0.312, Validation Accuracy:  0.338, Loss:  3.834
    Epoch   0 Batch    8/22 - Train Accuracy:  0.340, Validation Accuracy:  0.340, Loss:  3.546
    Epoch   0 Batch    9/22 - Train Accuracy:  0.320, Validation Accuracy:  0.348, Loss:  3.572
    Epoch   0 Batch   10/22 - Train Accuracy:  0.315, Validation Accuracy:  0.346, Loss:  3.475
    Epoch   0 Batch   11/22 - Train Accuracy:  0.342, Validation Accuracy:  0.340, Loss:  3.276
    Epoch   0 Batch   12/22 - Train Accuracy:  0.340, Validation Accuracy:  0.340, Loss:  3.226
    Epoch   0 Batch   13/22 - Train Accuracy:  0.340, Validation Accuracy:  0.340, Loss:  3.191
    Epoch   0 Batch   14/22 - Train Accuracy:  0.398, Validation Accuracy:  0.340, Loss:  2.933
    Epoch   0 Batch   15/22 - Train Accuracy:  0.309, Validation Accuracy:  0.340, Loss:  3.272
    Epoch   0 Batch   16/22 - Train Accuracy:  0.342, Validation Accuracy:  0.340, Loss:  3.106
    Epoch   0 Batch   17/22 - Train Accuracy:  0.344, Validation Accuracy:  0.340, Loss:  3.053
    Epoch   0 Batch   18/22 - Train Accuracy:  0.398, Validation Accuracy:  0.341, Loss:  2.802
    Epoch   0 Batch   19/22 - Train Accuracy:  0.379, Validation Accuracy:  0.348, Loss:  2.859
    Epoch   0 Batch   20/22 - Train Accuracy:  0.358, Validation Accuracy:  0.358, Loss:  2.940
    Epoch   1 Batch    0/22 - Train Accuracy:  0.374, Validation Accuracy:  0.373, Loss:  2.892
    Epoch   1 Batch    1/22 - Train Accuracy:  0.378, Validation Accuracy:  0.377, Loss:  2.869
    Epoch   1 Batch    2/22 - Train Accuracy:  0.374, Validation Accuracy:  0.372, Loss:  2.851
    Epoch   1 Batch    3/22 - Train Accuracy:  0.377, Validation Accuracy:  0.374, Loss:  2.836
    Epoch   1 Batch    4/22 - Train Accuracy:  0.373, Validation Accuracy:  0.373, Loss:  2.836
    Epoch   1 Batch    5/22 - Train Accuracy:  0.381, Validation Accuracy:  0.379, Loss:  2.792
    Epoch   1 Batch    6/22 - Train Accuracy:  0.387, Validation Accuracy:  0.386, Loss:  2.786
    Epoch   1 Batch    7/22 - Train Accuracy:  0.364, Validation Accuracy:  0.390, Loss:  2.879
    Epoch   1 Batch    8/22 - Train Accuracy:  0.405, Validation Accuracy:  0.406, Loss:  2.757
    Epoch   1 Batch    9/22 - Train Accuracy:  0.382, Validation Accuracy:  0.408, Loss:  2.860
    Epoch   1 Batch   10/22 - Train Accuracy:  0.383, Validation Accuracy:  0.411, Loss:  2.825
    Epoch   1 Batch   11/22 - Train Accuracy:  0.415, Validation Accuracy:  0.414, Loss:  2.702
    Epoch   1 Batch   12/22 - Train Accuracy:  0.422, Validation Accuracy:  0.422, Loss:  2.678
    Epoch   1 Batch   13/22 - Train Accuracy:  0.422, Validation Accuracy:  0.423, Loss:  2.655
    Epoch   1 Batch   14/22 - Train Accuracy:  0.479, Validation Accuracy:  0.429, Loss:  2.430
    Epoch   1 Batch   15/22 - Train Accuracy:  0.397, Validation Accuracy:  0.426, Loss:  2.738
    Epoch   1 Batch   16/22 - Train Accuracy:  0.440, Validation Accuracy:  0.439, Loss:  2.610
    Epoch   1 Batch   17/22 - Train Accuracy:  0.431, Validation Accuracy:  0.427, Loss:  2.564
    Epoch   1 Batch   18/22 - Train Accuracy:  0.487, Validation Accuracy:  0.438, Loss:  2.369
    Epoch   1 Batch   19/22 - Train Accuracy:  0.459, Validation Accuracy:  0.432, Loss:  2.439
    Epoch   1 Batch   20/22 - Train Accuracy:  0.434, Validation Accuracy:  0.434, Loss:  2.547
    Epoch   2 Batch    0/22 - Train Accuracy:  0.445, Validation Accuracy:  0.444, Loss:  2.519
    Epoch   2 Batch    1/22 - Train Accuracy:  0.441, Validation Accuracy:  0.440, Loss:  2.503
    Epoch   2 Batch    2/22 - Train Accuracy:  0.446, Validation Accuracy:  0.444, Loss:  2.485
    Epoch   2 Batch    3/22 - Train Accuracy:  0.447, Validation Accuracy:  0.445, Loss:  2.466
    Epoch   2 Batch    4/22 - Train Accuracy:  0.443, Validation Accuracy:  0.445, Loss:  2.471
    Epoch   2 Batch    5/22 - Train Accuracy:  0.447, Validation Accuracy:  0.446, Loss:  2.458
    Epoch   2 Batch    6/22 - Train Accuracy:  0.445, Validation Accuracy:  0.445, Loss:  2.436
    Epoch   2 Batch    7/22 - Train Accuracy:  0.426, Validation Accuracy:  0.449, Loss:  2.538
    Epoch   2 Batch    8/22 - Train Accuracy:  0.442, Validation Accuracy:  0.443, Loss:  2.416
    Epoch   2 Batch    9/22 - Train Accuracy:  0.425, Validation Accuracy:  0.449, Loss:  2.502
    Epoch   2 Batch   10/22 - Train Accuracy:  0.427, Validation Accuracy:  0.454, Loss:  2.503
    Epoch   2 Batch   11/22 - Train Accuracy:  0.453, Validation Accuracy:  0.452, Loss:  2.373
    Epoch   2 Batch   12/22 - Train Accuracy:  0.455, Validation Accuracy:  0.453, Loss:  2.368
    Epoch   2 Batch   13/22 - Train Accuracy:  0.459, Validation Accuracy:  0.460, Loss:  2.360
    Epoch   2 Batch   14/22 - Train Accuracy:  0.508, Validation Accuracy:  0.462, Loss:  2.140
    Epoch   2 Batch   15/22 - Train Accuracy:  0.441, Validation Accuracy:  0.467, Loss:  2.440
    Epoch   2 Batch   16/22 - Train Accuracy:  0.462, Validation Accuracy:  0.459, Loss:  2.317
    Epoch   2 Batch   17/22 - Train Accuracy:  0.477, Validation Accuracy:  0.472, Loss:  2.293
    Epoch   2 Batch   18/22 - Train Accuracy:  0.516, Validation Accuracy:  0.470, Loss:  2.107
    Epoch   2 Batch   19/22 - Train Accuracy:  0.487, Validation Accuracy:  0.461, Loss:  2.177
    Epoch   2 Batch   20/22 - Train Accuracy:  0.469, Validation Accuracy:  0.470, Loss:  2.283
    Epoch   3 Batch    0/22 - Train Accuracy:  0.467, Validation Accuracy:  0.465, Loss:  2.265
    Epoch   3 Batch    1/22 - Train Accuracy:  0.471, Validation Accuracy:  0.471, Loss:  2.252
    Epoch   3 Batch    2/22 - Train Accuracy:  0.471, Validation Accuracy:  0.470, Loss:  2.233
    Epoch   3 Batch    3/22 - Train Accuracy:  0.459, Validation Accuracy:  0.456, Loss:  2.218
    Epoch   3 Batch    4/22 - Train Accuracy:  0.469, Validation Accuracy:  0.473, Loss:  2.240
    Epoch   3 Batch    5/22 - Train Accuracy:  0.457, Validation Accuracy:  0.456, Loss:  2.236
    Epoch   3 Batch    6/22 - Train Accuracy:  0.478, Validation Accuracy:  0.477, Loss:  2.231
    Epoch   3 Batch    7/22 - Train Accuracy:  0.466, Validation Accuracy:  0.489, Loss:  2.293
    Epoch   3 Batch    8/22 - Train Accuracy:  0.463, Validation Accuracy:  0.464, Loss:  2.192
    Epoch   3 Batch    9/22 - Train Accuracy:  0.462, Validation Accuracy:  0.484, Loss:  2.286
    Epoch   3 Batch   10/22 - Train Accuracy:  0.456, Validation Accuracy:  0.480, Loss:  2.280
    Epoch   3 Batch   11/22 - Train Accuracy:  0.473, Validation Accuracy:  0.472, Loss:  2.152
    Epoch   3 Batch   12/22 - Train Accuracy:  0.490, Validation Accuracy:  0.492, Loss:  2.152
    Epoch   3 Batch   13/22 - Train Accuracy:  0.485, Validation Accuracy:  0.484, Loss:  2.146
    Epoch   3 Batch   14/22 - Train Accuracy:  0.523, Validation Accuracy:  0.478, Loss:  1.935
    Epoch   3 Batch   15/22 - Train Accuracy:  0.468, Validation Accuracy:  0.494, Loss:  2.218
    Epoch   3 Batch   16/22 - Train Accuracy:  0.486, Validation Accuracy:  0.484, Loss:  2.105
    Epoch   3 Batch   17/22 - Train Accuracy:  0.497, Validation Accuracy:  0.492, Loss:  2.075
    Epoch   3 Batch   18/22 - Train Accuracy:  0.532, Validation Accuracy:  0.491, Loss:  1.902
    Epoch   3 Batch   19/22 - Train Accuracy:  0.503, Validation Accuracy:  0.478, Loss:  1.972
    Epoch   3 Batch   20/22 - Train Accuracy:  0.479, Validation Accuracy:  0.483, Loss:  2.074
    Epoch   4 Batch    0/22 - Train Accuracy:  0.471, Validation Accuracy:  0.469, Loss:  2.075
    Epoch   4 Batch    1/22 - Train Accuracy:  0.487, Validation Accuracy:  0.488, Loss:  2.061
    Epoch   4 Batch    2/22 - Train Accuracy:  0.493, Validation Accuracy:  0.491, Loss:  2.022
    Epoch   4 Batch    3/22 - Train Accuracy:  0.480, Validation Accuracy:  0.476, Loss:  2.005
    Epoch   4 Batch    4/22 - Train Accuracy:  0.483, Validation Accuracy:  0.488, Loss:  2.037
    Epoch   4 Batch    5/22 - Train Accuracy:  0.490, Validation Accuracy:  0.490, Loss:  2.005
    Epoch   4 Batch    6/22 - Train Accuracy:  0.485, Validation Accuracy:  0.485, Loss:  1.987
    Epoch   4 Batch    7/22 - Train Accuracy:  0.458, Validation Accuracy:  0.482, Loss:  2.077
    Epoch   4 Batch    8/22 - Train Accuracy:  0.495, Validation Accuracy:  0.499, Loss:  1.991
    Epoch   4 Batch    9/22 - Train Accuracy:  0.469, Validation Accuracy:  0.492, Loss:  2.031
    Epoch   4 Batch   10/22 - Train Accuracy:  0.447, Validation Accuracy:  0.473, Loss:  2.039
    Epoch   4 Batch   11/22 - Train Accuracy:  0.494, Validation Accuracy:  0.495, Loss:  1.951
    Epoch   4 Batch   12/22 - Train Accuracy:  0.486, Validation Accuracy:  0.487, Loss:  1.937
    Epoch   4 Batch   13/22 - Train Accuracy:  0.474, Validation Accuracy:  0.476, Loss:  1.919
    Epoch   4 Batch   14/22 - Train Accuracy:  0.537, Validation Accuracy:  0.492, Loss:  1.745
    Epoch   4 Batch   15/22 - Train Accuracy:  0.449, Validation Accuracy:  0.479, Loss:  2.003
    Epoch   4 Batch   16/22 - Train Accuracy:  0.477, Validation Accuracy:  0.479, Loss:  1.894
    Epoch   4 Batch   17/22 - Train Accuracy:  0.494, Validation Accuracy:  0.490, Loss:  1.865
    Epoch   4 Batch   18/22 - Train Accuracy:  0.527, Validation Accuracy:  0.486, Loss:  1.721
    Epoch   4 Batch   19/22 - Train Accuracy:  0.515, Validation Accuracy:  0.491, Loss:  1.781
    Epoch   4 Batch   20/22 - Train Accuracy:  0.486, Validation Accuracy:  0.491, Loss:  1.864
    Epoch   5 Batch    0/22 - Train Accuracy:  0.489, Validation Accuracy:  0.491, Loss:  1.846
    Epoch   5 Batch    1/22 - Train Accuracy:  0.500, Validation Accuracy:  0.500, Loss:  1.838
    Epoch   5 Batch    2/22 - Train Accuracy:  0.490, Validation Accuracy:  0.489, Loss:  1.828
    Epoch   5 Batch    3/22 - Train Accuracy:  0.511, Validation Accuracy:  0.509, Loss:  1.817
    Epoch   5 Batch    4/22 - Train Accuracy:  0.479, Validation Accuracy:  0.486, Loss:  1.831
    Epoch   5 Batch    5/22 - Train Accuracy:  0.504, Validation Accuracy:  0.504, Loss:  1.819
    Epoch   5 Batch    6/22 - Train Accuracy:  0.491, Validation Accuracy:  0.493, Loss:  1.804
    Epoch   5 Batch    7/22 - Train Accuracy:  0.471, Validation Accuracy:  0.493, Loss:  1.866
    Epoch   5 Batch    8/22 - Train Accuracy:  0.495, Validation Accuracy:  0.500, Loss:  1.784
    Epoch   5 Batch    9/22 - Train Accuracy:  0.459, Validation Accuracy:  0.481, Loss:  1.842
    Epoch   5 Batch   10/22 - Train Accuracy:  0.481, Validation Accuracy:  0.505, Loss:  1.862
    Epoch   5 Batch   11/22 - Train Accuracy:  0.468, Validation Accuracy:  0.469, Loss:  1.795
    Epoch   5 Batch   12/22 - Train Accuracy:  0.499, Validation Accuracy:  0.499, Loss:  1.793
    Epoch   5 Batch   13/22 - Train Accuracy:  0.499, Validation Accuracy:  0.499, Loss:  1.749
    Epoch   5 Batch   14/22 - Train Accuracy:  0.524, Validation Accuracy:  0.479, Loss:  1.578
    Epoch   5 Batch   15/22 - Train Accuracy:  0.475, Validation Accuracy:  0.501, Loss:  1.837
    Epoch   5 Batch   16/22 - Train Accuracy:  0.497, Validation Accuracy:  0.498, Loss:  1.727
    Epoch   5 Batch   17/22 - Train Accuracy:  0.481, Validation Accuracy:  0.478, Loss:  1.686
    Epoch   5 Batch   18/22 - Train Accuracy:  0.541, Validation Accuracy:  0.499, Loss:  1.569
    Epoch   5 Batch   19/22 - Train Accuracy:  0.515, Validation Accuracy:  0.493, Loss:  1.610
    Epoch   5 Batch   20/22 - Train Accuracy:  0.483, Validation Accuracy:  0.487, Loss:  1.682
    Epoch   6 Batch    0/22 - Train Accuracy:  0.497, Validation Accuracy:  0.496, Loss:  1.675
    Epoch   6 Batch    1/22 - Train Accuracy:  0.495, Validation Accuracy:  0.496, Loss:  1.658
    Epoch   6 Batch    2/22 - Train Accuracy:  0.488, Validation Accuracy:  0.488, Loss:  1.642
    Epoch   6 Batch    3/22 - Train Accuracy:  0.501, Validation Accuracy:  0.499, Loss:  1.636
    Epoch   6 Batch    4/22 - Train Accuracy:  0.494, Validation Accuracy:  0.499, Loss:  1.638
    Epoch   6 Batch    5/22 - Train Accuracy:  0.486, Validation Accuracy:  0.486, Loss:  1.621
    Epoch   6 Batch    6/22 - Train Accuracy:  0.494, Validation Accuracy:  0.497, Loss:  1.619
    Epoch   6 Batch    7/22 - Train Accuracy:  0.471, Validation Accuracy:  0.493, Loss:  1.672
    Epoch   6 Batch    8/22 - Train Accuracy:  0.478, Validation Accuracy:  0.483, Loss:  1.592
    Epoch   6 Batch    9/22 - Train Accuracy:  0.476, Validation Accuracy:  0.498, Loss:  1.638
    Epoch   6 Batch   10/22 - Train Accuracy:  0.466, Validation Accuracy:  0.491, Loss:  1.647
    Epoch   6 Batch   11/22 - Train Accuracy:  0.492, Validation Accuracy:  0.492, Loss:  1.562
    Epoch   6 Batch   12/22 - Train Accuracy:  0.497, Validation Accuracy:  0.498, Loss:  1.548
    Epoch   6 Batch   13/22 - Train Accuracy:  0.487, Validation Accuracy:  0.488, Loss:  1.542
    Epoch   6 Batch   14/22 - Train Accuracy:  0.541, Validation Accuracy:  0.496, Loss:  1.394
    Epoch   6 Batch   15/22 - Train Accuracy:  0.473, Validation Accuracy:  0.498, Loss:  1.595
    Epoch   6 Batch   16/22 - Train Accuracy:  0.486, Validation Accuracy:  0.487, Loss:  1.509
    Epoch   6 Batch   17/22 - Train Accuracy:  0.501, Validation Accuracy:  0.498, Loss:  1.489
    Epoch   6 Batch   18/22 - Train Accuracy:  0.532, Validation Accuracy:  0.489, Loss:  1.369
    Epoch   6 Batch   19/22 - Train Accuracy:  0.514, Validation Accuracy:  0.490, Loss:  1.407
    Epoch   6 Batch   20/22 - Train Accuracy:  0.500, Validation Accuracy:  0.502, Loss:  1.473
    Epoch   7 Batch    0/22 - Train Accuracy:  0.487, Validation Accuracy:  0.487, Loss:  1.457
    Epoch   7 Batch    1/22 - Train Accuracy:  0.506, Validation Accuracy:  0.505, Loss:  1.452
    Epoch   7 Batch    2/22 - Train Accuracy:  0.497, Validation Accuracy:  0.495, Loss:  1.437
    Epoch   7 Batch    3/22 - Train Accuracy:  0.505, Validation Accuracy:  0.501, Loss:  1.418
    Epoch   7 Batch    4/22 - Train Accuracy:  0.500, Validation Accuracy:  0.505, Loss:  1.422
    Epoch   7 Batch    5/22 - Train Accuracy:  0.483, Validation Accuracy:  0.482, Loss:  1.412
    Epoch   7 Batch    6/22 - Train Accuracy:  0.504, Validation Accuracy:  0.506, Loss:  1.405
    Epoch   7 Batch    7/22 - Train Accuracy:  0.462, Validation Accuracy:  0.483, Loss:  1.457
    Epoch   7 Batch    8/22 - Train Accuracy:  0.496, Validation Accuracy:  0.501, Loss:  1.385
    Epoch   7 Batch    9/22 - Train Accuracy:  0.471, Validation Accuracy:  0.493, Loss:  1.419
    Epoch   7 Batch   10/22 - Train Accuracy:  0.464, Validation Accuracy:  0.489, Loss:  1.420
    Epoch   7 Batch   11/22 - Train Accuracy:  0.495, Validation Accuracy:  0.494, Loss:  1.349
    Epoch   7 Batch   12/22 - Train Accuracy:  0.473, Validation Accuracy:  0.474, Loss:  1.338
    Epoch   7 Batch   13/22 - Train Accuracy:  0.488, Validation Accuracy:  0.490, Loss:  1.333
    Epoch   7 Batch   14/22 - Train Accuracy:  0.514, Validation Accuracy:  0.467, Loss:  1.206
    Epoch   7 Batch   15/22 - Train Accuracy:  0.451, Validation Accuracy:  0.478, Loss:  1.379
    Epoch   7 Batch   16/22 - Train Accuracy:  0.480, Validation Accuracy:  0.481, Loss:  1.304
    Epoch   7 Batch   17/22 - Train Accuracy:  0.492, Validation Accuracy:  0.489, Loss:  1.281
    Epoch   7 Batch   18/22 - Train Accuracy:  0.542, Validation Accuracy:  0.499, Loss:  1.180
    Epoch   7 Batch   19/22 - Train Accuracy:  0.516, Validation Accuracy:  0.493, Loss:  1.215
    Epoch   7 Batch   20/22 - Train Accuracy:  0.498, Validation Accuracy:  0.500, Loss:  1.269
    Epoch   8 Batch    0/22 - Train Accuracy:  0.497, Validation Accuracy:  0.494, Loss:  1.254
    Epoch   8 Batch    1/22 - Train Accuracy:  0.508, Validation Accuracy:  0.506, Loss:  1.246
    Epoch   8 Batch    2/22 - Train Accuracy:  0.496, Validation Accuracy:  0.493, Loss:  1.245
    Epoch   8 Batch    3/22 - Train Accuracy:  0.509, Validation Accuracy:  0.507, Loss:  1.236
    Epoch   8 Batch    4/22 - Train Accuracy:  0.487, Validation Accuracy:  0.490, Loss:  1.246
    Epoch   8 Batch    5/22 - Train Accuracy:  0.516, Validation Accuracy:  0.513, Loss:  1.232
    Epoch   8 Batch    6/22 - Train Accuracy:  0.506, Validation Accuracy:  0.506, Loss:  1.212
    Epoch   8 Batch    7/22 - Train Accuracy:  0.478, Validation Accuracy:  0.499, Loss:  1.255
    Epoch   8 Batch    8/22 - Train Accuracy:  0.515, Validation Accuracy:  0.516, Loss:  1.204
    Epoch   8 Batch    9/22 - Train Accuracy:  0.485, Validation Accuracy:  0.504, Loss:  1.241
    Epoch   8 Batch   10/22 - Train Accuracy:  0.495, Validation Accuracy:  0.519, Loss:  1.239
    Epoch   8 Batch   11/22 - Train Accuracy:  0.523, Validation Accuracy:  0.522, Loss:  1.173
    Epoch   8 Batch   12/22 - Train Accuracy:  0.511, Validation Accuracy:  0.509, Loss:  1.166
    Epoch   8 Batch   13/22 - Train Accuracy:  0.523, Validation Accuracy:  0.525, Loss:  1.163
    Epoch   8 Batch   14/22 - Train Accuracy:  0.572, Validation Accuracy:  0.532, Loss:  1.049
    Epoch   8 Batch   15/22 - Train Accuracy:  0.497, Validation Accuracy:  0.521, Loss:  1.202
    Epoch   8 Batch   16/22 - Train Accuracy:  0.541, Validation Accuracy:  0.538, Loss:  1.141
    Epoch   8 Batch   17/22 - Train Accuracy:  0.536, Validation Accuracy:  0.532, Loss:  1.123
    Epoch   8 Batch   18/22 - Train Accuracy:  0.573, Validation Accuracy:  0.534, Loss:  1.034
    Epoch   8 Batch   19/22 - Train Accuracy:  0.559, Validation Accuracy:  0.536, Loss:  1.067
    Epoch   8 Batch   20/22 - Train Accuracy:  0.526, Validation Accuracy:  0.526, Loss:  1.115
    Epoch   9 Batch    0/22 - Train Accuracy:  0.528, Validation Accuracy:  0.522, Loss:  1.098
    Epoch   9 Batch    1/22 - Train Accuracy:  0.527, Validation Accuracy:  0.523, Loss:  1.091
    Epoch   9 Batch    2/22 - Train Accuracy:  0.518, Validation Accuracy:  0.516, Loss:  1.091
    Epoch   9 Batch    3/22 - Train Accuracy:  0.510, Validation Accuracy:  0.506, Loss:  1.079
    Epoch   9 Batch    4/22 - Train Accuracy:  0.509, Validation Accuracy:  0.508, Loss:  1.085
    Epoch   9 Batch    5/22 - Train Accuracy:  0.517, Validation Accuracy:  0.516, Loss:  1.078
    Epoch   9 Batch    6/22 - Train Accuracy:  0.516, Validation Accuracy:  0.514, Loss:  1.073
    Epoch   9 Batch    7/22 - Train Accuracy:  0.493, Validation Accuracy:  0.514, Loss:  1.115
    Epoch   9 Batch    8/22 - Train Accuracy:  0.514, Validation Accuracy:  0.514, Loss:  1.060
    Epoch   9 Batch    9/22 - Train Accuracy:  0.490, Validation Accuracy:  0.510, Loss:  1.090
    Epoch   9 Batch   10/22 - Train Accuracy:  0.491, Validation Accuracy:  0.514, Loss:  1.098
    Epoch   9 Batch   11/22 - Train Accuracy:  0.509, Validation Accuracy:  0.506, Loss:  1.043
    Epoch   9 Batch   12/22 - Train Accuracy:  0.511, Validation Accuracy:  0.509, Loss:  1.035
    Epoch   9 Batch   13/22 - Train Accuracy:  0.520, Validation Accuracy:  0.517, Loss:  1.034
    Epoch   9 Batch   14/22 - Train Accuracy:  0.551, Validation Accuracy:  0.508, Loss:  0.938
    Epoch   9 Batch   15/22 - Train Accuracy:  0.499, Validation Accuracy:  0.522, Loss:  1.076
    Epoch   9 Batch   16/22 - Train Accuracy:  0.534, Validation Accuracy:  0.533, Loss:  1.019
    Epoch   9 Batch   17/22 - Train Accuracy:  0.529, Validation Accuracy:  0.523, Loss:  1.006
    Epoch   9 Batch   18/22 - Train Accuracy:  0.571, Validation Accuracy:  0.531, Loss:  0.927
    Epoch   9 Batch   19/22 - Train Accuracy:  0.558, Validation Accuracy:  0.537, Loss:  0.960
    Epoch   9 Batch   20/22 - Train Accuracy:  0.535, Validation Accuracy:  0.537, Loss:  1.000
    Epoch  10 Batch    0/22 - Train Accuracy:  0.534, Validation Accuracy:  0.534, Loss:  0.989
    Epoch  12 Batch   17/22 - Train Accuracy:  0.574, Validation Accuracy:  0.569, Loss:  0.801
    Epoch  12 Batch   18/22 - Train Accuracy:  0.608, Validation Accuracy:  0.572, Loss:  0.738
    Epoch  12 Batch   19/22 - Train Accuracy:  0.594, Validation Accuracy:  0.573, Loss:  0.765
    Epoch  12 Batch   20/22 - Train Accuracy:  0.576, Validation Accuracy:  0.576, Loss:  0.801
    Epoch  13 Batch    0/22 - Train Accuracy:  0.578, Validation Accuracy:  0.577, Loss:  0.792
    Epoch  13 Batch    1/22 - Train Accuracy:  0.582, Validation Accuracy:  0.578, Loss:  0.788
    Epoch  13 Batch    2/22 - Train Accuracy:  0.583, Validation Accuracy:  0.578, Loss:  0.791
    Epoch  13 Batch    3/22 - Train Accuracy:  0.576, Validation Accuracy:  0.573, Loss:  0.780
    Epoch  13 Batch    4/22 - Train Accuracy:  0.567, Validation Accuracy:  0.571, Loss:  0.789
    Epoch  13 Batch    5/22 - Train Accuracy:  0.575, Validation Accuracy:  0.573, Loss:  0.786
    Epoch  13 Batch    6/22 - Train Accuracy:  0.575, Validation Accuracy:  0.575, Loss:  0.780
    Epoch  13 Batch    7/22 - Train Accuracy:  0.557, Validation Accuracy:  0.577, Loss:  0.817
    Epoch  13 Batch    8/22 - Train Accuracy:  0.571, Validation Accuracy:  0.575, Loss:  0.779
    Epoch  13 Batch    9/22 - Train Accuracy:  0.555, Validation Accuracy:  0.571, Loss:  0.802
    Epoch  13 Batch   10/22 - Train Accuracy:  0.551, Validation Accuracy:  0.570, Loss:  0.807
    Epoch  13 Batch   11/22 - Train Accuracy:  0.574, Validation Accuracy:  0.572, Loss:  0.769
    Epoch  13 Batch   12/22 - Train Accuracy:  0.576, Validation Accuracy:  0.573, Loss:  0.768
    Epoch  13 Batch   13/22 - Train Accuracy:  0.579, Validation Accuracy:  0.576, Loss:  0.768
    Epoch  13 Batch   14/22 - Train Accuracy:  0.616, Validation Accuracy:  0.578, Loss:  0.697
    Epoch  13 Batch   15/22 - Train Accuracy:  0.559, Validation Accuracy:  0.580, Loss:  0.799
    Epoch  13 Batch   16/22 - Train Accuracy:  0.584, Validation Accuracy:  0.581, Loss:  0.761
    Epoch  13 Batch   17/22 - Train Accuracy:  0.589, Validation Accuracy:  0.580, Loss:  0.752
    Epoch  13 Batch   18/22 - Train Accuracy:  0.617, Validation Accuracy:  0.581, Loss:  0.693
    Epoch  13 Batch   19/22 - Train Accuracy:  0.602, Validation Accuracy:  0.582, Loss:  0.719
    Epoch  13 Batch   20/22 - Train Accuracy:  0.583, Validation Accuracy:  0.581, Loss:  0.754
    Epoch  14 Batch    0/22 - Train Accuracy:  0.583, Validation Accuracy:  0.581, Loss:  0.748
    Epoch  14 Batch    1/22 - Train Accuracy:  0.583, Validation Accuracy:  0.581, Loss:  0.743
    Epoch  14 Batch    2/22 - Train Accuracy:  0.585, Validation Accuracy:  0.580, Loss:  0.745
    Epoch  14 Batch    3/22 - Train Accuracy:  0.581, Validation Accuracy:  0.578, Loss:  0.736
    Epoch  14 Batch    4/22 - Train Accuracy:  0.575, Validation Accuracy:  0.579, Loss:  0.745
    Epoch  14 Batch    5/22 - Train Accuracy:  0.584, Validation Accuracy:  0.580, Loss:  0.742
    Epoch  14 Batch    6/22 - Train Accuracy:  0.580, Validation Accuracy:  0.580, Loss:  0.736
    Epoch  14 Batch    7/22 - Train Accuracy:  0.560, Validation Accuracy:  0.580, Loss:  0.771
    Epoch  14 Batch    8/22 - Train Accuracy:  0.578, Validation Accuracy:  0.581, Loss:  0.736
    Epoch  14 Batch    9/22 - Train Accuracy:  0.566, Validation Accuracy:  0.581, Loss:  0.759
    Epoch  14 Batch   10/22 - Train Accuracy:  0.563, Validation Accuracy:  0.581, Loss:  0.765
    Epoch  14 Batch   11/22 - Train Accuracy:  0.584, Validation Accuracy:  0.580, Loss:  0.729
    Epoch  14 Batch   12/22 - Train Accuracy:  0.583, Validation Accuracy:  0.579, Loss:  0.726
    Epoch  14 Batch   13/22 - Train Accuracy:  0.580, Validation Accuracy:  0.577, Loss:  0.727
    Epoch  14 Batch   14/22 - Train Accuracy:  0.616, Validation Accuracy:  0.578, Loss:  0.660
    Epoch  14 Batch   15/22 - Train Accuracy:  0.561, Validation Accuracy:  0.580, Loss:  0.757
    Epoch  14 Batch   16/22 - Train Accuracy:  0.585, Validation Accuracy:  0.582, Loss:  0.721
    Epoch  14 Batch   17/22 - Train Accuracy:  0.589, Validation Accuracy:  0.581, Loss:  0.713
    Epoch  14 Batch   18/22 - Train Accuracy:  0.616, Validation Accuracy:  0.580, Loss:  0.657
    Epoch  14 Batch   19/22 - Train Accuracy:  0.603, Validation Accuracy:  0.582, Loss:  0.683
    Epoch  14 Batch   20/22 - Train Accuracy:  0.585, Validation Accuracy:  0.583, Loss:  0.713
    Epoch  15 Batch    0/22 - Train Accuracy:  0.583, Validation Accuracy:  0.580, Loss:  0.709
    Epoch  15 Batch    1/22 - Train Accuracy:  0.581, Validation Accuracy:  0.578, Loss:  0.705
    Epoch  15 Batch    2/22 - Train Accuracy:  0.587, Validation Accuracy:  0.580, Loss:  0.709
    Epoch  15 Batch    3/22 - Train Accuracy:  0.583, Validation Accuracy:  0.580, Loss:  0.699
    Epoch  15 Batch    4/22 - Train Accuracy:  0.577, Validation Accuracy:  0.581, Loss:  0.708
    Epoch  15 Batch    5/22 - Train Accuracy:  0.585, Validation Accuracy:  0.581, Loss:  0.706
    Epoch  15 Batch    6/22 - Train Accuracy:  0.586, Validation Accuracy:  0.587, Loss:  0.701
    Epoch  15 Batch    7/22 - Train Accuracy:  0.565, Validation Accuracy:  0.583, Loss:  0.732
    Epoch  15 Batch    8/22 - Train Accuracy:  0.579, Validation Accuracy:  0.582, Loss:  0.701
    Epoch  15 Batch    9/22 - Train Accuracy:  0.572, Validation Accuracy:  0.586, Loss:  0.721
    Epoch  15 Batch   10/22 - Train Accuracy:  0.565, Validation Accuracy:  0.583, Loss:  0.728
    Epoch  15 Batch   11/22 - Train Accuracy:  0.589, Validation Accuracy:  0.585, Loss:  0.693
    Epoch  15 Batch   12/22 - Train Accuracy:  0.585, Validation Accuracy:  0.580, Loss:  0.692
    Epoch  15 Batch   13/22 - Train Accuracy:  0.590, Validation Accuracy:  0.585, Loss:  0.694
    Epoch  15 Batch   14/22 - Train Accuracy:  0.623, Validation Accuracy:  0.585, Loss:  0.631
    Epoch  15 Batch   15/22 - Train Accuracy:  0.566, Validation Accuracy:  0.585, Loss:  0.719
    Epoch  15 Batch   16/22 - Train Accuracy:  0.591, Validation Accuracy:  0.587, Loss:  0.686
    Epoch  15 Batch   17/22 - Train Accuracy:  0.590, Validation Accuracy:  0.582, Loss:  0.681
    Epoch  15 Batch   18/22 - Train Accuracy:  0.623, Validation Accuracy:  0.585, Loss:  0.627
    Epoch  15 Batch   19/22 - Train Accuracy:  0.607, Validation Accuracy:  0.587, Loss:  0.650
    Epoch  15 Batch   20/22 - Train Accuracy:  0.587, Validation Accuracy:  0.585, Loss:  0.681
    Epoch  16 Batch    0/22 - Train Accuracy:  0.590, Validation Accuracy:  0.587, Loss:  0.677
    Epoch  16 Batch    1/22 - Train Accuracy:  0.588, Validation Accuracy:  0.585, Loss:  0.673
    Epoch  16 Batch    2/22 - Train Accuracy:  0.596, Validation Accuracy:  0.590, Loss:  0.677
    Epoch  16 Batch    3/22 - Train Accuracy:  0.585, Validation Accuracy:  0.583, Loss:  0.668
    Epoch  16 Batch    4/22 - Train Accuracy:  0.586, Validation Accuracy:  0.590, Loss:  0.677
    Epoch  16 Batch    5/22 - Train Accuracy:  0.596, Validation Accuracy:  0.591, Loss:  0.673
    Epoch  16 Batch    6/22 - Train Accuracy:  0.587, Validation Accuracy:  0.587, Loss:  0.668
    Epoch  16 Batch    7/22 - Train Accuracy:  0.572, Validation Accuracy:  0.591, Loss:  0.701
    Epoch  16 Batch    8/22 - Train Accuracy:  0.580, Validation Accuracy:  0.585, Loss:  0.672
    Epoch  16 Batch    9/22 - Train Accuracy:  0.578, Validation Accuracy:  0.591, Loss:  0.693
    Epoch  16 Batch   10/22 - Train Accuracy:  0.567, Validation Accuracy:  0.584, Loss:  0.698
    Epoch  16 Batch   11/22 - Train Accuracy:  0.595, Validation Accuracy:  0.590, Loss:  0.665
    Epoch  16 Batch   12/22 - Train Accuracy:  0.595, Validation Accuracy:  0.591, Loss:  0.663
    Epoch  16 Batch   13/22 - Train Accuracy:  0.588, Validation Accuracy:  0.584, Loss:  0.664
    Epoch  16 Batch   14/22 - Train Accuracy:  0.630, Validation Accuracy:  0.593, Loss:  0.605
    Epoch  16 Batch   15/22 - Train Accuracy:  0.576, Validation Accuracy:  0.593, Loss:  0.690
    Epoch  16 Batch   16/22 - Train Accuracy:  0.594, Validation Accuracy:  0.590, Loss:  0.657
    Epoch  16 Batch   17/22 - Train Accuracy:  0.601, Validation Accuracy:  0.592, Loss:  0.651
    Epoch  16 Batch   18/22 - Train Accuracy:  0.631, Validation Accuracy:  0.593, Loss:  0.601
    Epoch  16 Batch   19/22 - Train Accuracy:  0.617, Validation Accuracy:  0.595, Loss:  0.622
    Epoch  16 Batch   20/22 - Train Accuracy:  0.598, Validation Accuracy:  0.598, Loss:  0.653
    Epoch  17 Batch    0/22 - Train Accuracy:  0.600, Validation Accuracy:  0.597, Loss:  0.652
    Epoch  17 Batch    1/22 - Train Accuracy:  0.602, Validation Accuracy:  0.597, Loss:  0.645
    Epoch  17 Batch    2/22 - Train Accuracy:  0.597, Validation Accuracy:  0.591, Loss:  0.646
    Epoch  17 Batch    3/22 - Train Accuracy:  0.602, Validation Accuracy:  0.596, Loss:  0.639
    Epoch  17 Batch    4/22 - Train Accuracy:  0.593, Validation Accuracy:  0.597, Loss:  0.648
    Epoch  17 Batch    5/22 - Train Accuracy:  0.602, Validation Accuracy:  0.598, Loss:  0.647
    Epoch  17 Batch    6/22 - Train Accuracy:  0.599, Validation Accuracy:  0.597, Loss:  0.640
    Epoch  17 Batch    7/22 - Train Accuracy:  0.574, Validation Accuracy:  0.591, Loss:  0.672
    Epoch  17 Batch    8/22 - Train Accuracy:  0.598, Validation Accuracy:  0.602, Loss:  0.645
    Epoch  17 Batch    9/22 - Train Accuracy:  0.582, Validation Accuracy:  0.596, Loss:  0.661
    Epoch  17 Batch   10/22 - Train Accuracy:  0.586, Validation Accuracy:  0.603, Loss:  0.668
    Epoch  17 Batch   11/22 - Train Accuracy:  0.594, Validation Accuracy:  0.593, Loss:  0.641
    Epoch  17 Batch   12/22 - Train Accuracy:  0.609, Validation Accuracy:  0.605, Loss:  0.650
    Epoch  17 Batch   13/22 - Train Accuracy:  0.595, Validation Accuracy:  0.592, Loss:  0.656
    Epoch  17 Batch   14/22 - Train Accuracy:  0.643, Validation Accuracy:  0.604, Loss:  0.605
    Epoch  17 Batch   15/22 - Train Accuracy:  0.590, Validation Accuracy:  0.609, Loss:  0.665
    Epoch  17 Batch   16/22 - Train Accuracy:  0.600, Validation Accuracy:  0.595, Loss:  0.651
    Epoch  17 Batch   17/22 - Train Accuracy:  0.610, Validation Accuracy:  0.602, Loss:  0.646
    Epoch  17 Batch   18/22 - Train Accuracy:  0.639, Validation Accuracy:  0.602, Loss:  0.585
    Epoch  17 Batch   19/22 - Train Accuracy:  0.630, Validation Accuracy:  0.609, Loss:  0.618
    Epoch  17 Batch   20/22 - Train Accuracy:  0.591, Validation Accuracy:  0.592, Loss:  0.632
    Epoch  18 Batch    0/22 - Train Accuracy:  0.610, Validation Accuracy:  0.607, Loss:  0.640
    Epoch  18 Batch    1/22 - Train Accuracy:  0.608, Validation Accuracy:  0.603, Loss:  0.626
    Epoch  18 Batch    2/22 - Train Accuracy:  0.608, Validation Accuracy:  0.602, Loss:  0.637
    Epoch  18 Batch    3/22 - Train Accuracy:  0.602, Validation Accuracy:  0.598, Loss:  0.622
    Epoch  18 Batch    4/22 - Train Accuracy:  0.602, Validation Accuracy:  0.605, Loss:  0.632
    Epoch  18 Batch    5/22 - Train Accuracy:  0.612, Validation Accuracy:  0.607, Loss:  0.630
    Epoch  18 Batch    6/22 - Train Accuracy:  0.598, Validation Accuracy:  0.596, Loss:  0.622
    Epoch  18 Batch    7/22 - Train Accuracy:  0.597, Validation Accuracy:  0.610, Loss:  0.655
    Epoch  18 Batch    8/22 - Train Accuracy:  0.615, Validation Accuracy:  0.616, Loss:  0.622
    Epoch  18 Batch    9/22 - Train Accuracy:  0.606, Validation Accuracy:  0.617, Loss:  0.646
    Epoch  18 Batch   10/22 - Train Accuracy:  0.599, Validation Accuracy:  0.614, Loss:  0.645
    Epoch  18 Batch   11/22 - Train Accuracy:  0.620, Validation Accuracy:  0.617, Loss:  0.617
    Epoch  18 Batch   12/22 - Train Accuracy:  0.627, Validation Accuracy:  0.623, Loss:  0.614
    Epoch  18 Batch   13/22 - Train Accuracy:  0.625, Validation Accuracy:  0.624, Loss:  0.616
    Epoch  18 Batch   14/22 - Train Accuracy:  0.654, Validation Accuracy:  0.619, Loss:  0.560
    Epoch  18 Batch   15/22 - Train Accuracy:  0.606, Validation Accuracy:  0.622, Loss:  0.642
    Epoch  18 Batch   16/22 - Train Accuracy:  0.628, Validation Accuracy:  0.623, Loss:  0.610
    Epoch  18 Batch   17/22 - Train Accuracy:  0.632, Validation Accuracy:  0.624, Loss:  0.605
    Epoch  18 Batch   18/22 - Train Accuracy:  0.657, Validation Accuracy:  0.623, Loss:  0.557
    Epoch  18 Batch   19/22 - Train Accuracy:  0.642, Validation Accuracy:  0.622, Loss:  0.578
    Epoch  18 Batch   20/22 - Train Accuracy:  0.630, Validation Accuracy:  0.628, Loss:  0.609
    Epoch  19 Batch    0/22 - Train Accuracy:  0.629, Validation Accuracy:  0.629, Loss:  0.602
    Epoch  19 Batch    1/22 - Train Accuracy:  0.631, Validation Accuracy:  0.627, Loss:  0.598
    Epoch  19 Batch    2/22 - Train Accuracy:  0.633, Validation Accuracy:  0.629, Loss:  0.602
    Epoch  19 Batch    3/22 - Train Accuracy:  0.629, Validation Accuracy:  0.623, Loss:  0.594
    Epoch  19 Batch    4/22 - Train Accuracy:  0.619, Validation Accuracy:  0.624, Loss:  0.602
    Epoch  19 Batch    5/22 - Train Accuracy:  0.633, Validation Accuracy:  0.627, Loss:  0.600
    Epoch  19 Batch    6/22 - Train Accuracy:  0.631, Validation Accuracy:  0.629, Loss:  0.596
    Epoch  19 Batch    7/22 - Train Accuracy:  0.614, Validation Accuracy:  0.627, Loss:  0.622
    Epoch  19 Batch    8/22 - Train Accuracy:  0.627, Validation Accuracy:  0.630, Loss:  0.596
    Epoch  19 Batch    9/22 - Train Accuracy:  0.621, Validation Accuracy:  0.631, Loss:  0.617
    Epoch  19 Batch   10/22 - Train Accuracy:  0.609, Validation Accuracy:  0.627, Loss:  0.622
    Epoch  19 Batch   11/22 - Train Accuracy:  0.636, Validation Accuracy:  0.633, Loss:  0.595
    Epoch  19 Batch   12/22 - Train Accuracy:  0.635, Validation Accuracy:  0.631, Loss:  0.591
    Epoch  19 Batch   13/22 - Train Accuracy:  0.631, Validation Accuracy:  0.631, Loss:  0.591
    Epoch  19 Batch   14/22 - Train Accuracy:  0.669, Validation Accuracy:  0.634, Loss:  0.537
    Epoch  19 Batch   15/22 - Train Accuracy:  0.616, Validation Accuracy:  0.631, Loss:  0.615
    Epoch  19 Batch   16/22 - Train Accuracy:  0.642, Validation Accuracy:  0.636, Loss:  0.587
    Epoch  19 Batch   17/22 - Train Accuracy:  0.644, Validation Accuracy:  0.634, Loss:  0.582
    Epoch  19 Batch   18/22 - Train Accuracy:  0.667, Validation Accuracy:  0.633, Loss:  0.535
    Epoch  19 Batch   19/22 - Train Accuracy:  0.656, Validation Accuracy:  0.636, Loss:  0.554
    Epoch  19 Batch   20/22 - Train Accuracy:  0.637, Validation Accuracy:  0.634, Loss:  0.584
    Epoch  20 Batch    0/22 - Train Accuracy:  0.635, Validation Accuracy:  0.633, Loss:  0.580
    Epoch  20 Batch    1/22 - Train Accuracy:  0.639, Validation Accuracy:  0.635, Loss:  0.576
    Epoch  20 Batch    2/22 - Train Accuracy:  0.641, Validation Accuracy:  0.635, Loss:  0.580
    Epoch  20 Batch    3/22 - Train Accuracy:  0.642, Validation Accuracy:  0.636, Loss:  0.570
    Epoch  20 Batch    4/22 - Train Accuracy:  0.631, Validation Accuracy:  0.635, Loss:  0.578
    Epoch  20 Batch    5/22 - Train Accuracy:  0.640, Validation Accuracy:  0.635, Loss:  0.578
    Epoch  20 Batch    6/22 - Train Accuracy:  0.640, Validation Accuracy:  0.639, Loss:  0.574
    Epoch  20 Batch    7/22 - Train Accuracy:  0.624, Validation Accuracy:  0.638, Loss:  0.598
    Epoch  20 Batch    8/22 - Train Accuracy:  0.633, Validation Accuracy:  0.637, Loss:  0.575
    Epoch  20 Batch    9/22 - Train Accuracy:  0.629, Validation Accuracy:  0.637, Loss:  0.592
    Epoch  20 Batch   10/22 - Train Accuracy:  0.626, Validation Accuracy:  0.639, Loss:  0.597
    Epoch  20 Batch   11/22 - Train Accuracy:  0.643, Validation Accuracy:  0.639, Loss:  0.570
    Epoch  20 Batch   12/22 - Train Accuracy:  0.642, Validation Accuracy:  0.636, Loss:  0.571
    Epoch  20 Batch   13/22 - Train Accuracy:  0.638, Validation Accuracy:  0.638, Loss:  0.576
    Epoch  20 Batch   14/22 - Train Accuracy:  0.672, Validation Accuracy:  0.637, Loss:  0.526
    Epoch  20 Batch   15/22 - Train Accuracy:  0.625, Validation Accuracy:  0.641, Loss:  0.602
    Epoch  20 Batch   16/22 - Train Accuracy:  0.647, Validation Accuracy:  0.641, Loss:  0.575
    Epoch  20 Batch   17/22 - Train Accuracy:  0.654, Validation Accuracy:  0.644, Loss:  0.566
    Epoch  20 Batch   18/22 - Train Accuracy:  0.676, Validation Accuracy:  0.642, Loss:  0.517
    Epoch  20 Batch   19/22 - Train Accuracy:  0.659, Validation Accuracy:  0.639, Loss:  0.538
    Epoch  20 Batch   20/22 - Train Accuracy:  0.646, Validation Accuracy:  0.644, Loss:  0.568
    Epoch  21 Batch    0/22 - Train Accuracy:  0.647, Validation Accuracy:  0.645, Loss:  0.566
    Epoch  21 Batch    1/22 - Train Accuracy:  0.648, Validation Accuracy:  0.644, Loss:  0.557
    Epoch  21 Batch    2/22 - Train Accuracy:  0.649, Validation Accuracy:  0.642, Loss:  0.560
    Epoch  21 Batch    3/22 - Train Accuracy:  0.650, Validation Accuracy:  0.644, Loss:  0.555
    Epoch  21 Batch    4/22 - Train Accuracy:  0.641, Validation Accuracy:  0.645, Loss:  0.564
    Epoch  21 Batch    5/22 - Train Accuracy:  0.648, Validation Accuracy:  0.643, Loss:  0.561
    Epoch  21 Batch    6/22 - Train Accuracy:  0.648, Validation Accuracy:  0.645, Loss:  0.556
    Epoch  21 Batch    7/22 - Train Accuracy:  0.634, Validation Accuracy:  0.647, Loss:  0.578
    Epoch  21 Batch    8/22 - Train Accuracy:  0.641, Validation Accuracy:  0.644, Loss:  0.559
    Epoch  21 Batch    9/22 - Train Accuracy:  0.639, Validation Accuracy:  0.646, Loss:  0.579
    Epoch  21 Batch   10/22 - Train Accuracy:  0.633, Validation Accuracy:  0.648, Loss:  0.584
    Epoch  21 Batch   11/22 - Train Accuracy:  0.656, Validation Accuracy:  0.651, Loss:  0.553
    Epoch  21 Batch   12/22 - Train Accuracy:  0.656, Validation Accuracy:  0.651, Loss:  0.550
    Epoch  21 Batch   13/22 - Train Accuracy:  0.651, Validation Accuracy:  0.649, Loss:  0.554
    Epoch  21 Batch   14/22 - Train Accuracy:  0.688, Validation Accuracy:  0.654, Loss:  0.504
    Epoch  21 Batch   15/22 - Train Accuracy:  0.639, Validation Accuracy:  0.654, Loss:  0.573
    Epoch  21 Batch   16/22 - Train Accuracy:  0.654, Validation Accuracy:  0.649, Loss:  0.547
    Epoch  21 Batch   17/22 - Train Accuracy:  0.660, Validation Accuracy:  0.651, Loss:  0.546
    Epoch  21 Batch   18/22 - Train Accuracy:  0.687, Validation Accuracy:  0.655, Loss:  0.505
    Epoch  21 Batch   19/22 - Train Accuracy:  0.667, Validation Accuracy:  0.646, Loss:  0.524
    Epoch  21 Batch   20/22 - Train Accuracy:  0.654, Validation Accuracy:  0.652, Loss:  0.554
    Epoch  22 Batch    0/22 - Train Accuracy:  0.651, Validation Accuracy:  0.647, Loss:  0.556
    Epoch  22 Batch    1/22 - Train Accuracy:  0.654, Validation Accuracy:  0.650, Loss:  0.549
    Epoch  22 Batch    2/22 - Train Accuracy:  0.650, Validation Accuracy:  0.646, Loss:  0.550
    Epoch  22 Batch    3/22 - Train Accuracy:  0.659, Validation Accuracy:  0.653, Loss:  0.541
    Epoch  22 Batch    4/22 - Train Accuracy:  0.650, Validation Accuracy:  0.654, Loss:  0.545
    Epoch  22 Batch    5/22 - Train Accuracy:  0.659, Validation Accuracy:  0.654, Loss:  0.543
    Epoch  22 Batch    6/22 - Train Accuracy:  0.656, Validation Accuracy:  0.656, Loss:  0.543
    Epoch  22 Batch    7/22 - Train Accuracy:  0.643, Validation Accuracy:  0.656, Loss:  0.565
    Epoch  22 Batch    8/22 - Train Accuracy:  0.653, Validation Accuracy:  0.656, Loss:  0.539
    Epoch  22 Batch    9/22 - Train Accuracy:  0.649, Validation Accuracy:  0.658, Loss:  0.557
    Epoch  22 Batch   10/22 - Train Accuracy:  0.647, Validation Accuracy:  0.660, Loss:  0.561
    Epoch  22 Batch   11/22 - Train Accuracy:  0.665, Validation Accuracy:  0.661, Loss:  0.532
    Epoch  22 Batch   12/22 - Train Accuracy:  0.663, Validation Accuracy:  0.660, Loss:  0.533
    Epoch  22 Batch   13/22 - Train Accuracy:  0.662, Validation Accuracy:  0.662, Loss:  0.535
    Epoch  22 Batch   14/22 - Train Accuracy:  0.694, Validation Accuracy:  0.660, Loss:  0.485
    Epoch  22 Batch   15/22 - Train Accuracy:  0.645, Validation Accuracy:  0.660, Loss:  0.556
    Epoch  22 Batch   16/22 - Train Accuracy:  0.664, Validation Accuracy:  0.661, Loss:  0.531
    Epoch  22 Batch   17/22 - Train Accuracy:  0.670, Validation Accuracy:  0.663, Loss:  0.527
    Epoch  22 Batch   18/22 - Train Accuracy:  0.694, Validation Accuracy:  0.660, Loss:  0.486
    Epoch  22 Batch   19/22 - Train Accuracy:  0.681, Validation Accuracy:  0.663, Loss:  0.504
    Epoch  22 Batch   20/22 - Train Accuracy:  0.665, Validation Accuracy:  0.663, Loss:  0.529
    Epoch  23 Batch    0/22 - Train Accuracy:  0.660, Validation Accuracy:  0.658, Loss:  0.532
    Epoch  23 Batch    1/22 - Train Accuracy:  0.659, Validation Accuracy:  0.654, Loss:  0.544
    Epoch  23 Batch    2/22 - Train Accuracy:  0.656, Validation Accuracy:  0.652, Loss:  0.569
    Epoch  23 Batch    3/22 - Train Accuracy:  0.666, Validation Accuracy:  0.659, Loss:  0.553
    Epoch  23 Batch    4/22 - Train Accuracy:  0.653, Validation Accuracy:  0.656, Loss:  0.531
    Epoch  23 Batch    5/22 - Train Accuracy:  0.661, Validation Accuracy:  0.658, Loss:  0.549
    Epoch  23 Batch    6/22 - Train Accuracy:  0.659, Validation Accuracy:  0.658, Loss:  0.533
    Epoch  23 Batch    7/22 - Train Accuracy:  0.646, Validation Accuracy:  0.659, Loss:  0.562
    Epoch  23 Batch    8/22 - Train Accuracy:  0.657, Validation Accuracy:  0.663, Loss:  0.539
    Epoch  23 Batch    9/22 - Train Accuracy:  0.651, Validation Accuracy:  0.660, Loss:  0.548
    Epoch  23 Batch   10/22 - Train Accuracy:  0.646, Validation Accuracy:  0.660, Loss:  0.561
    Epoch  23 Batch   11/22 - Train Accuracy:  0.662, Validation Accuracy:  0.659, Loss:  0.527
    Epoch  23 Batch   12/22 - Train Accuracy:  0.663, Validation Accuracy:  0.661, Loss:  0.531
    Epoch  23 Batch   13/22 - Train Accuracy:  0.665, Validation Accuracy:  0.663, Loss:  0.528
    Epoch  23 Batch   14/22 - Train Accuracy:  0.696, Validation Accuracy:  0.664, Loss:  0.478
    Epoch  23 Batch   15/22 - Train Accuracy:  0.649, Validation Accuracy:  0.663, Loss:  0.550
    Epoch  23 Batch   16/22 - Train Accuracy:  0.666, Validation Accuracy:  0.663, Loss:  0.520
    Epoch  23 Batch   17/22 - Train Accuracy:  0.675, Validation Accuracy:  0.667, Loss:  0.518
    Epoch  23 Batch   18/22 - Train Accuracy:  0.697, Validation Accuracy:  0.664, Loss:  0.475
    Epoch  23 Batch   19/22 - Train Accuracy:  0.681, Validation Accuracy:  0.663, Loss:  0.491
    Epoch  23 Batch   20/22 - Train Accuracy:  0.670, Validation Accuracy:  0.667, Loss:  0.519
    Epoch  24 Batch    0/22 - Train Accuracy:  0.670, Validation Accuracy:  0.666, Loss:  0.513
    Epoch  24 Batch    1/22 - Train Accuracy:  0.673, Validation Accuracy:  0.669, Loss:  0.511
    Epoch  24 Batch    2/22 - Train Accuracy:  0.670, Validation Accuracy:  0.669, Loss:  0.513
    Epoch  24 Batch    3/22 - Train Accuracy:  0.674, Validation Accuracy:  0.667, Loss:  0.505
    Epoch  24 Batch    4/22 - Train Accuracy:  0.665, Validation Accuracy:  0.670, Loss:  0.514
    Epoch  24 Batch    5/22 - Train Accuracy:  0.670, Validation Accuracy:  0.666, Loss:  0.509
    Epoch  24 Batch    6/22 - Train Accuracy:  0.672, Validation Accuracy:  0.670, Loss:  0.507
    Epoch  24 Batch    7/22 - Train Accuracy:  0.653, Validation Accuracy:  0.665, Loss:  0.529
    Epoch  24 Batch    8/22 - Train Accuracy:  0.666, Validation Accuracy:  0.668, Loss:  0.510
    Epoch  24 Batch    9/22 - Train Accuracy:  0.648, Validation Accuracy:  0.655, Loss:  0.527
    Epoch  24 Batch   10/22 - Train Accuracy:  0.643, Validation Accuracy:  0.655, Loss:  0.536
    Epoch  24 Batch   11/22 - Train Accuracy:  0.664, Validation Accuracy:  0.659, Loss:  0.520
    Epoch  24 Batch   12/22 - Train Accuracy:  0.673, Validation Accuracy:  0.669, Loss:  0.512
    Epoch  24 Batch   13/22 - Train Accuracy:  0.667, Validation Accuracy:  0.666, Loss:  0.508
    Epoch  24 Batch   14/22 - Train Accuracy:  0.701, Validation Accuracy:  0.667, Loss:  0.463
    Epoch  24 Batch   15/22 - Train Accuracy:  0.657, Validation Accuracy:  0.670, Loss:  0.524
    Epoch  24 Batch   16/22 - Train Accuracy:  0.673, Validation Accuracy:  0.670, Loss:  0.501
    Epoch  24 Batch   17/22 - Train Accuracy:  0.681, Validation Accuracy:  0.672, Loss:  0.499
    Epoch  24 Batch   18/22 - Train Accuracy:  0.703, Validation Accuracy:  0.670, Loss:  0.456
    Epoch  24 Batch   19/22 - Train Accuracy:  0.691, Validation Accuracy:  0.674, Loss:  0.474
    Epoch  24 Batch   20/22 - Train Accuracy:  0.678, Validation Accuracy:  0.676, Loss:  0.498
    Epoch  25 Batch    0/22 - Train Accuracy:  0.674, Validation Accuracy:  0.670, Loss:  0.495
    Epoch  25 Batch    1/22 - Train Accuracy:  0.679, Validation Accuracy:  0.674, Loss:  0.491
    Epoch  25 Batch    2/22 - Train Accuracy:  0.678, Validation Accuracy:  0.676, Loss:  0.494
    Epoch  25 Batch    3/22 - Train Accuracy:  0.679, Validation Accuracy:  0.672, Loss:  0.486
    Epoch  25 Batch    4/22 - Train Accuracy:  0.673, Validation Accuracy:  0.675, Loss:  0.493
    Epoch  25 Batch    5/22 - Train Accuracy:  0.679, Validation Accuracy:  0.676, Loss:  0.492
    Epoch  25 Batch    6/22 - Train Accuracy:  0.679, Validation Accuracy:  0.675, Loss:  0.489
    Epoch  25 Batch    7/22 - Train Accuracy:  0.662, Validation Accuracy:  0.674, Loss:  0.512
    Epoch  25 Batch    8/22 - Train Accuracy:  0.675, Validation Accuracy:  0.677, Loss:  0.491
    Epoch  25 Batch    9/22 - Train Accuracy:  0.665, Validation Accuracy:  0.670, Loss:  0.508
    Epoch  25 Batch   10/22 - Train Accuracy:  0.662, Validation Accuracy:  0.672, Loss:  0.516
    Epoch  25 Batch   11/22 - Train Accuracy:  0.672, Validation Accuracy:  0.668, Loss:  0.502
    Epoch  25 Batch   12/22 - Train Accuracy:  0.677, Validation Accuracy:  0.674, Loss:  0.510
    Epoch  25 Batch   13/22 - Train Accuracy:  0.677, Validation Accuracy:  0.676, Loss:  0.506
    Epoch  25 Batch   14/22 - Train Accuracy:  0.704, Validation Accuracy:  0.673, Loss:  0.444
    Epoch  25 Batch   15/22 - Train Accuracy:  0.665, Validation Accuracy:  0.678, Loss:  0.516
    Epoch  25 Batch   16/22 - Train Accuracy:  0.676, Validation Accuracy:  0.673, Loss:  0.500
    Epoch  25 Batch   17/22 - Train Accuracy:  0.685, Validation Accuracy:  0.675, Loss:  0.487
    Epoch  25 Batch   18/22 - Train Accuracy:  0.704, Validation Accuracy:  0.672, Loss:  0.447
    Epoch  25 Batch   19/22 - Train Accuracy:  0.695, Validation Accuracy:  0.676, Loss:  0.469
    Epoch  25 Batch   20/22 - Train Accuracy:  0.680, Validation Accuracy:  0.678, Loss:  0.485
    Epoch  26 Batch    0/22 - Train Accuracy:  0.681, Validation Accuracy:  0.679, Loss:  0.485
    Epoch  26 Batch    1/22 - Train Accuracy:  0.685, Validation Accuracy:  0.679, Loss:  0.483
    Epoch  26 Batch    2/22 - Train Accuracy:  0.684, Validation Accuracy:  0.681, Loss:  0.481
    Epoch  26 Batch    3/22 - Train Accuracy:  0.687, Validation Accuracy:  0.682, Loss:  0.474
    Epoch  26 Batch    4/22 - Train Accuracy:  0.676, Validation Accuracy:  0.678, Loss:  0.482
    Epoch  26 Batch    5/22 - Train Accuracy:  0.687, Validation Accuracy:  0.683, Loss:  0.478
    Epoch  26 Batch    6/22 - Train Accuracy:  0.686, Validation Accuracy:  0.683, Loss:  0.475
    Epoch  26 Batch    7/22 - Train Accuracy:  0.668, Validation Accuracy:  0.681, Loss:  0.498
    Epoch  26 Batch    8/22 - Train Accuracy:  0.683, Validation Accuracy:  0.686, Loss:  0.476
    Epoch  26 Batch    9/22 - Train Accuracy:  0.677, Validation Accuracy:  0.684, Loss:  0.491
    Epoch  26 Batch   10/22 - Train Accuracy:  0.670, Validation Accuracy:  0.684, Loss:  0.496
    Epoch  26 Batch   11/22 - Train Accuracy:  0.688, Validation Accuracy:  0.685, Loss:  0.471
    Epoch  26 Batch   12/22 - Train Accuracy:  0.688, Validation Accuracy:  0.685, Loss:  0.470
    Epoch  26 Batch   13/22 - Train Accuracy:  0.687, Validation Accuracy:  0.686, Loss:  0.471
    Epoch  26 Batch   14/22 - Train Accuracy:  0.717, Validation Accuracy:  0.687, Loss:  0.430
    Epoch  26 Batch   15/22 - Train Accuracy:  0.674, Validation Accuracy:  0.688, Loss:  0.490
    Epoch  26 Batch   16/22 - Train Accuracy:  0.689, Validation Accuracy:  0.688, Loss:  0.467
    Epoch  26 Batch   17/22 - Train Accuracy:  0.696, Validation Accuracy:  0.688, Loss:  0.463
    Epoch  26 Batch   18/22 - Train Accuracy:  0.717, Validation Accuracy:  0.688, Loss:  0.427
    Epoch  26 Batch   19/22 - Train Accuracy:  0.706, Validation Accuracy:  0.690, Loss:  0.442
    Epoch  26 Batch   20/22 - Train Accuracy:  0.691, Validation Accuracy:  0.690, Loss:  0.463
    Epoch  27 Batch    0/22 - Train Accuracy:  0.694, Validation Accuracy:  0.690, Loss:  0.463
    Epoch  27 Batch    1/22 - Train Accuracy:  0.693, Validation Accuracy:  0.691, Loss:  0.459
    Epoch  27 Batch    2/22 - Train Accuracy:  0.694, Validation Accuracy:  0.692, Loss:  0.462
    Epoch  27 Batch    3/22 - Train Accuracy:  0.699, Validation Accuracy:  0.693, Loss:  0.454
    Epoch  27 Batch    4/22 - Train Accuracy:  0.688, Validation Accuracy:  0.694, Loss:  0.459
    Epoch  27 Batch    5/22 - Train Accuracy:  0.694, Validation Accuracy:  0.690, Loss:  0.459
    Epoch  27 Batch    6/22 - Train Accuracy:  0.686, Validation Accuracy:  0.685, Loss:  0.459
    Epoch  27 Batch    7/22 - Train Accuracy:  0.654, Validation Accuracy:  0.665, Loss:  0.484
    Epoch  27 Batch    8/22 - Train Accuracy:  0.648, Validation Accuracy:  0.651, Loss:  0.489
    Epoch  27 Batch    9/22 - Train Accuracy:  0.648, Validation Accuracy:  0.658, Loss:  0.558
    Epoch  27 Batch   10/22 - Train Accuracy:  0.618, Validation Accuracy:  0.631, Loss:  0.562
    Epoch  27 Batch   11/22 - Train Accuracy:  0.667, Validation Accuracy:  0.660, Loss:  0.547
    Epoch  27 Batch   12/22 - Train Accuracy:  0.662, Validation Accuracy:  0.656, Loss:  0.501
    Epoch  27 Batch   13/22 - Train Accuracy:  0.669, Validation Accuracy:  0.667, Loss:  0.505
    Epoch  27 Batch   14/22 - Train Accuracy:  0.702, Validation Accuracy:  0.668, Loss:  0.450
    Epoch  27 Batch   15/22 - Train Accuracy:  0.668, Validation Accuracy:  0.680, Loss:  0.504
    Epoch  27 Batch   16/22 - Train Accuracy:  0.674, Validation Accuracy:  0.670, Loss:  0.483
    Epoch  27 Batch   17/22 - Train Accuracy:  0.683, Validation Accuracy:  0.675, Loss:  0.476
    Epoch  27 Batch   18/22 - Train Accuracy:  0.712, Validation Accuracy:  0.679, Loss:  0.442
    Epoch  27 Batch   19/22 - Train Accuracy:  0.702, Validation Accuracy:  0.685, Loss:  0.452
    Epoch  27 Batch   20/22 - Train Accuracy:  0.681, Validation Accuracy:  0.680, Loss:  0.470
    Epoch  28 Batch    0/22 - Train Accuracy:  0.684, Validation Accuracy:  0.680, Loss:  0.469
    Epoch  28 Batch    1/22 - Train Accuracy:  0.691, Validation Accuracy:  0.688, Loss:  0.465
    Epoch  28 Batch    2/22 - Train Accuracy:  0.689, Validation Accuracy:  0.685, Loss:  0.464
    Epoch  28 Batch    3/22 - Train Accuracy:  0.691, Validation Accuracy:  0.687, Loss:  0.459
    Epoch  28 Batch    4/22 - Train Accuracy:  0.680, Validation Accuracy:  0.683, Loss:  0.461
    Epoch  28 Batch    5/22 - Train Accuracy:  0.690, Validation Accuracy:  0.688, Loss:  0.459
    Epoch  28 Batch    6/22 - Train Accuracy:  0.692, Validation Accuracy:  0.691, Loss:  0.456
    Epoch  28 Batch    7/22 - Train Accuracy:  0.680, Validation Accuracy:  0.689, Loss:  0.475
    Epoch  28 Batch    8/22 - Train Accuracy:  0.687, Validation Accuracy:  0.690, Loss:  0.455
    Epoch  28 Batch    9/22 - Train Accuracy:  0.685, Validation Accuracy:  0.691, Loss:  0.470
    Epoch  28 Batch   10/22 - Train Accuracy:  0.677, Validation Accuracy:  0.690, Loss:  0.472
    Epoch  28 Batch   11/22 - Train Accuracy:  0.699, Validation Accuracy:  0.692, Loss:  0.448
    Epoch  28 Batch   12/22 - Train Accuracy:  0.697, Validation Accuracy:  0.695, Loss:  0.448
    Epoch  28 Batch   13/22 - Train Accuracy:  0.695, Validation Accuracy:  0.694, Loss:  0.449
    Epoch  28 Batch   14/22 - Train Accuracy:  0.726, Validation Accuracy:  0.695, Loss:  0.407
    Epoch  28 Batch   15/22 - Train Accuracy:  0.684, Validation Accuracy:  0.696, Loss:  0.465
    Epoch  28 Batch   16/22 - Train Accuracy:  0.700, Validation Accuracy:  0.696, Loss:  0.443
    Epoch  28 Batch   17/22 - Train Accuracy:  0.702, Validation Accuracy:  0.695, Loss:  0.441
    Epoch  28 Batch   18/22 - Train Accuracy:  0.725, Validation Accuracy:  0.696, Loss:  0.406
    Epoch  28 Batch   19/22 - Train Accuracy:  0.711, Validation Accuracy:  0.696, Loss:  0.419
    Epoch  28 Batch   20/22 - Train Accuracy:  0.700, Validation Accuracy:  0.698, Loss:  0.441
    Epoch  29 Batch    0/22 - Train Accuracy:  0.702, Validation Accuracy:  0.698, Loss:  0.439
    Epoch  29 Batch    1/22 - Train Accuracy:  0.702, Validation Accuracy:  0.698, Loss:  0.436
    Epoch  29 Batch    2/22 - Train Accuracy:  0.703, Validation Accuracy:  0.699, Loss:  0.439
    Epoch  29 Batch    3/22 - Train Accuracy:  0.705, Validation Accuracy:  0.700, Loss:  0.430
    Epoch  29 Batch    4/22 - Train Accuracy:  0.697, Validation Accuracy:  0.700, Loss:  0.436
    Epoch  29 Batch    5/22 - Train Accuracy:  0.707, Validation Accuracy:  0.701, Loss:  0.435
    Epoch  29 Batch    6/22 - Train Accuracy:  0.703, Validation Accuracy:  0.700, Loss:  0.433
    Epoch  29 Batch    7/22 - Train Accuracy:  0.689, Validation Accuracy:  0.699, Loss:  0.452
    Epoch  29 Batch    8/22 - Train Accuracy:  0.698, Validation Accuracy:  0.701, Loss:  0.434
    Epoch  29 Batch    9/22 - Train Accuracy:  0.696, Validation Accuracy:  0.702, Loss:  0.447
    Epoch  29 Batch   10/22 - Train Accuracy:  0.690, Validation Accuracy:  0.703, Loss:  0.452
    Epoch  29 Batch   11/22 - Train Accuracy:  0.707, Validation Accuracy:  0.703, Loss:  0.429
    Epoch  29 Batch   12/22 - Train Accuracy:  0.705, Validation Accuracy:  0.701, Loss:  0.430
    Epoch  29 Batch   13/22 - Train Accuracy:  0.702, Validation Accuracy:  0.700, Loss:  0.433
    Epoch  29 Batch   14/22 - Train Accuracy:  0.732, Validation Accuracy:  0.702, Loss:  0.393
    Epoch  29 Batch   15/22 - Train Accuracy:  0.692, Validation Accuracy:  0.702, Loss:  0.449
    Epoch  29 Batch   16/22 - Train Accuracy:  0.706, Validation Accuracy:  0.703, Loss:  0.429
    Epoch  29 Batch   17/22 - Train Accuracy:  0.706, Validation Accuracy:  0.699, Loss:  0.428
    Epoch  29 Batch   18/22 - Train Accuracy:  0.731, Validation Accuracy:  0.702, Loss:  0.397
    Epoch  29 Batch   19/22 - Train Accuracy:  0.718, Validation Accuracy:  0.703, Loss:  0.408
    Epoch  29 Batch   20/22 - Train Accuracy:  0.707, Validation Accuracy:  0.705, Loss:  0.426
    Epoch  30 Batch    0/22 - Train Accuracy:  0.708, Validation Accuracy:  0.704, Loss:  0.424
    Epoch  30 Batch    1/22 - Train Accuracy:  0.708, Validation Accuracy:  0.703, Loss:  0.422
    Epoch  30 Batch    2/22 - Train Accuracy:  0.709, Validation Accuracy:  0.705, Loss:  0.427
    Epoch  30 Batch    3/22 - Train Accuracy:  0.711, Validation Accuracy:  0.703, Loss:  0.422
    Epoch  30 Batch    4/22 - Train Accuracy:  0.700, Validation Accuracy:  0.705, Loss:  0.426
    Epoch  30 Batch    5/22 - Train Accuracy:  0.710, Validation Accuracy:  0.704, Loss:  0.423
    Epoch  30 Batch    6/22 - Train Accuracy:  0.708, Validation Accuracy:  0.703, Loss:  0.421
    Epoch  30 Batch    7/22 - Train Accuracy:  0.694, Validation Accuracy:  0.704, Loss:  0.437
    Epoch  30 Batch    8/22 - Train Accuracy:  0.703, Validation Accuracy:  0.705, Loss:  0.420
    Epoch  30 Batch    9/22 - Train Accuracy:  0.697, Validation Accuracy:  0.704, Loss:  0.435
    Epoch  30 Batch   10/22 - Train Accuracy:  0.688, Validation Accuracy:  0.700, Loss:  0.442
    Epoch  30 Batch   11/22 - Train Accuracy:  0.707, Validation Accuracy:  0.702, Loss:  0.425
    Epoch  30 Batch   12/22 - Train Accuracy:  0.704, Validation Accuracy:  0.701, Loss:  0.431
    Epoch  30 Batch   13/22 - Train Accuracy:  0.710, Validation Accuracy:  0.708, Loss:  0.427
    Epoch  30 Batch   14/22 - Train Accuracy:  0.737, Validation Accuracy:  0.708, Loss:  0.380
    Epoch  30 Batch   15/22 - Train Accuracy:  0.693, Validation Accuracy:  0.704, Loss:  0.434
    Epoch  30 Batch   16/22 - Train Accuracy:  0.709, Validation Accuracy:  0.705, Loss:  0.420
    Epoch  30 Batch   17/22 - Train Accuracy:  0.711, Validation Accuracy:  0.704, Loss:  0.422
    Epoch  30 Batch   18/22 - Train Accuracy:  0.736, Validation Accuracy:  0.707, Loss:  0.383
    Epoch  30 Batch   19/22 - Train Accuracy:  0.723, Validation Accuracy:  0.707, Loss:  0.392
    Epoch  30 Batch   20/22 - Train Accuracy:  0.707, Validation Accuracy:  0.703, Loss:  0.418
    Epoch  31 Batch    0/22 - Train Accuracy:  0.710, Validation Accuracy:  0.706, Loss:  0.420
    Epoch  31 Batch    1/22 - Train Accuracy:  0.714, Validation Accuracy:  0.708, Loss:  0.416
    Epoch  31 Batch    2/22 - Train Accuracy:  0.715, Validation Accuracy:  0.711, Loss:  0.413
    Epoch  31 Batch    3/22 - Train Accuracy:  0.715, Validation Accuracy:  0.708, Loss:  0.406
    Epoch  31 Batch    4/22 - Train Accuracy:  0.702, Validation Accuracy:  0.707, Loss:  0.416
    Epoch  31 Batch    5/22 - Train Accuracy:  0.714, Validation Accuracy:  0.710, Loss:  0.413
    Epoch  31 Batch    6/22 - Train Accuracy:  0.711, Validation Accuracy:  0.708, Loss:  0.409
    Epoch  31 Batch    7/22 - Train Accuracy:  0.699, Validation Accuracy:  0.709, Loss:  0.426
    Epoch  31 Batch    8/22 - Train Accuracy:  0.708, Validation Accuracy:  0.709, Loss:  0.410
    Epoch  31 Batch    9/22 - Train Accuracy:  0.701, Validation Accuracy:  0.706, Loss:  0.426
    Epoch  31 Batch   10/22 - Train Accuracy:  0.697, Validation Accuracy:  0.710, Loss:  0.428
    Epoch  31 Batch   11/22 - Train Accuracy:  0.716, Validation Accuracy:  0.711, Loss:  0.405
    Epoch  31 Batch   12/22 - Train Accuracy:  0.712, Validation Accuracy:  0.709, Loss:  0.405
    Epoch  31 Batch   13/22 - Train Accuracy:  0.713, Validation Accuracy:  0.711, Loss:  0.407
    Epoch  31 Batch   14/22 - Train Accuracy:  0.742, Validation Accuracy:  0.712, Loss:  0.369
    Epoch  31 Batch   15/22 - Train Accuracy:  0.703, Validation Accuracy:  0.713, Loss:  0.420
    Epoch  31 Batch   16/22 - Train Accuracy:  0.719, Validation Accuracy:  0.713, Loss:  0.401
    Epoch  31 Batch   17/22 - Train Accuracy:  0.720, Validation Accuracy:  0.712, Loss:  0.400
    Epoch  31 Batch   18/22 - Train Accuracy:  0.741, Validation Accuracy:  0.713, Loss:  0.369
    Epoch  31 Batch   19/22 - Train Accuracy:  0.729, Validation Accuracy:  0.715, Loss:  0.379
    Epoch  31 Batch   20/22 - Train Accuracy:  0.719, Validation Accuracy:  0.715, Loss:  0.399
    Epoch  32 Batch    0/22 - Train Accuracy:  0.718, Validation Accuracy:  0.715, Loss:  0.399
    Epoch  32 Batch    1/22 - Train Accuracy:  0.719, Validation Accuracy:  0.712, Loss:  0.395
    Epoch  32 Batch    2/22 - Train Accuracy:  0.719, Validation Accuracy:  0.714, Loss:  0.400
    Epoch  32 Batch    3/22 - Train Accuracy:  0.720, Validation Accuracy:  0.713, Loss:  0.395
    Epoch  32 Batch    4/22 - Train Accuracy:  0.710, Validation Accuracy:  0.714, Loss:  0.400
    Epoch  32 Batch    5/22 - Train Accuracy:  0.720, Validation Accuracy:  0.714, Loss:  0.398
    Epoch  32 Batch    6/22 - Train Accuracy:  0.719, Validation Accuracy:  0.716, Loss:  0.396
    Epoch  32 Batch    7/22 - Train Accuracy:  0.706, Validation Accuracy:  0.716, Loss:  0.412
    Epoch  32 Batch    8/22 - Train Accuracy:  0.716, Validation Accuracy:  0.718, Loss:  0.393
    Epoch  32 Batch    9/22 - Train Accuracy:  0.711, Validation Accuracy:  0.717, Loss:  0.405
    Epoch  32 Batch   10/22 - Train Accuracy:  0.706, Validation Accuracy:  0.717, Loss:  0.410
    Epoch  32 Batch   11/22 - Train Accuracy:  0.722, Validation Accuracy:  0.717, Loss:  0.389
    Epoch  32 Batch   12/22 - Train Accuracy:  0.720, Validation Accuracy:  0.717, Loss:  0.391
    Epoch  32 Batch   13/22 - Train Accuracy:  0.717, Validation Accuracy:  0.714, Loss:  0.394
    Epoch  32 Batch   14/22 - Train Accuracy:  0.745, Validation Accuracy:  0.716, Loss:  0.359
    Epoch  32 Batch   15/22 - Train Accuracy:  0.698, Validation Accuracy:  0.706, Loss:  0.412
    Epoch  32 Batch   16/22 - Train Accuracy:  0.710, Validation Accuracy:  0.706, Loss:  0.403
    Epoch  32 Batch   17/22 - Train Accuracy:  0.708, Validation Accuracy:  0.698, Loss:  0.421
    Epoch  32 Batch   18/22 - Train Accuracy:  0.746, Validation Accuracy:  0.719, Loss:  0.385
    Epoch  32 Batch   19/22 - Train Accuracy:  0.723, Validation Accuracy:  0.708, Loss:  0.370
    Epoch  32 Batch   20/22 - Train Accuracy:  0.703, Validation Accuracy:  0.700, Loss:  0.409
    Epoch  33 Batch    0/22 - Train Accuracy:  0.719, Validation Accuracy:  0.713, Loss:  0.414
    Epoch  33 Batch    1/22 - Train Accuracy:  0.718, Validation Accuracy:  0.714, Loss:  0.390
    Epoch  33 Batch    2/22 - Train Accuracy:  0.711, Validation Accuracy:  0.708, Loss:  0.396
    Epoch  33 Batch    3/22 - Train Accuracy:  0.721, Validation Accuracy:  0.714, Loss:  0.400
    Epoch  33 Batch    4/22 - Train Accuracy:  0.712, Validation Accuracy:  0.717, Loss:  0.390
    Epoch  33 Batch    5/22 - Train Accuracy:  0.715, Validation Accuracy:  0.711, Loss:  0.392
    Epoch  33 Batch    6/22 - Train Accuracy:  0.718, Validation Accuracy:  0.720, Loss:  0.391
    Epoch  33 Batch    7/22 - Train Accuracy:  0.707, Validation Accuracy:  0.715, Loss:  0.400
    Epoch  33 Batch    8/22 - Train Accuracy:  0.718, Validation Accuracy:  0.720, Loss:  0.390
    Epoch  33 Batch    9/22 - Train Accuracy:  0.710, Validation Accuracy:  0.717, Loss:  0.396
    Epoch  33 Batch   10/22 - Train Accuracy:  0.707, Validation Accuracy:  0.719, Loss:  0.405
    Epoch  33 Batch   11/22 - Train Accuracy:  0.724, Validation Accuracy:  0.720, Loss:  0.381
    Epoch  33 Batch   12/22 - Train Accuracy:  0.722, Validation Accuracy:  0.722, Loss:  0.379
    Epoch  33 Batch   13/22 - Train Accuracy:  0.724, Validation Accuracy:  0.722, Loss:  0.380
    Epoch  33 Batch   14/22 - Train Accuracy:  0.751, Validation Accuracy:  0.723, Loss:  0.344
    Epoch  33 Batch   15/22 - Train Accuracy:  0.713, Validation Accuracy:  0.722, Loss:  0.390
    Epoch  33 Batch   16/22 - Train Accuracy:  0.728, Validation Accuracy:  0.724, Loss:  0.375
    Epoch  33 Batch   17/22 - Train Accuracy:  0.728, Validation Accuracy:  0.722, Loss:  0.371
    Epoch  33 Batch   18/22 - Train Accuracy:  0.751, Validation Accuracy:  0.725, Loss:  0.341
    Epoch  33 Batch   19/22 - Train Accuracy:  0.740, Validation Accuracy:  0.727, Loss:  0.348
    Epoch  33 Batch   20/22 - Train Accuracy:  0.729, Validation Accuracy:  0.727, Loss:  0.370
    Epoch  34 Batch    0/22 - Train Accuracy:  0.729, Validation Accuracy:  0.726, Loss:  0.368
    Epoch  34 Batch    1/22 - Train Accuracy:  0.733, Validation Accuracy:  0.729, Loss:  0.364
    Epoch  34 Batch    2/22 - Train Accuracy:  0.731, Validation Accuracy:  0.728, Loss:  0.366
    Epoch  34 Batch    3/22 - Train Accuracy:  0.732, Validation Accuracy:  0.726, Loss:  0.359
    Epoch  34 Batch    4/22 - Train Accuracy:  0.725, Validation Accuracy:  0.729, Loss:  0.363
    Epoch  34 Batch    5/22 - Train Accuracy:  0.735, Validation Accuracy:  0.730, Loss:  0.361
    Epoch  34 Batch    6/22 - Train Accuracy:  0.731, Validation Accuracy:  0.730, Loss:  0.360
    Epoch  34 Batch    7/22 - Train Accuracy:  0.719, Validation Accuracy:  0.729, Loss:  0.375
    Epoch  34 Batch    8/22 - Train Accuracy:  0.729, Validation Accuracy:  0.733, Loss:  0.359
    Epoch  34 Batch    9/22 - Train Accuracy:  0.723, Validation Accuracy:  0.730, Loss:  0.369
    Epoch  34 Batch   10/22 - Train Accuracy:  0.714, Validation Accuracy:  0.730, Loss:  0.373
    Epoch  34 Batch   11/22 - Train Accuracy:  0.731, Validation Accuracy:  0.727, Loss:  0.357
    Epoch  34 Batch   12/22 - Train Accuracy:  0.731, Validation Accuracy:  0.728, Loss:  0.359
    Epoch  34 Batch   13/22 - Train Accuracy:  0.730, Validation Accuracy:  0.729, Loss:  0.360
    Epoch  34 Batch   14/22 - Train Accuracy:  0.758, Validation Accuracy:  0.732, Loss:  0.324
    Epoch  34 Batch   15/22 - Train Accuracy:  0.724, Validation Accuracy:  0.734, Loss:  0.366
    Epoch  34 Batch   16/22 - Train Accuracy:  0.738, Validation Accuracy:  0.731, Loss:  0.351
    Epoch  34 Batch   17/22 - Train Accuracy:  0.736, Validation Accuracy:  0.731, Loss:  0.351
    Epoch  34 Batch   18/22 - Train Accuracy:  0.760, Validation Accuracy:  0.734, Loss:  0.322
    Epoch  34 Batch   19/22 - Train Accuracy:  0.749, Validation Accuracy:  0.734, Loss:  0.331
    Epoch  34 Batch   20/22 - Train Accuracy:  0.737, Validation Accuracy:  0.734, Loss:  0.348
    Epoch  35 Batch    0/22 - Train Accuracy:  0.736, Validation Accuracy:  0.734, Loss:  0.348
    Epoch  35 Batch    1/22 - Train Accuracy:  0.738, Validation Accuracy:  0.735, Loss:  0.345
    Epoch  35 Batch    2/22 - Train Accuracy:  0.739, Validation Accuracy:  0.738, Loss:  0.347
    Epoch  35 Batch    3/22 - Train Accuracy:  0.744, Validation Accuracy:  0.739, Loss:  0.341
    Epoch  35 Batch    4/22 - Train Accuracy:  0.732, Validation Accuracy:  0.736, Loss:  0.343
    Epoch  35 Batch    5/22 - Train Accuracy:  0.742, Validation Accuracy:  0.737, Loss:  0.344
    Epoch  35 Batch    6/22 - Train Accuracy:  0.737, Validation Accuracy:  0.736, Loss:  0.343
    Epoch  35 Batch    7/22 - Train Accuracy:  0.726, Validation Accuracy:  0.738, Loss:  0.357
    Epoch  35 Batch    8/22 - Train Accuracy:  0.734, Validation Accuracy:  0.738, Loss:  0.341
    Epoch  35 Batch    9/22 - Train Accuracy:  0.730, Validation Accuracy:  0.736, Loss:  0.352
    Epoch  35 Batch   10/22 - Train Accuracy:  0.725, Validation Accuracy:  0.737, Loss:  0.355
    Epoch  35 Batch   11/22 - Train Accuracy:  0.739, Validation Accuracy:  0.736, Loss:  0.337
    Epoch  35 Batch   12/22 - Train Accuracy:  0.742, Validation Accuracy:  0.740, Loss:  0.338
    Epoch  35 Batch   13/22 - Train Accuracy:  0.743, Validation Accuracy:  0.740, Loss:  0.340
    Epoch  35 Batch   14/22 - Train Accuracy:  0.766, Validation Accuracy:  0.739, Loss:  0.305
    Epoch  35 Batch   15/22 - Train Accuracy:  0.731, Validation Accuracy:  0.740, Loss:  0.350
    Epoch  35 Batch   16/22 - Train Accuracy:  0.744, Validation Accuracy:  0.740, Loss:  0.336
    Epoch  35 Batch   17/22 - Train Accuracy:  0.744, Validation Accuracy:  0.740, Loss:  0.334
    Epoch  35 Batch   18/22 - Train Accuracy:  0.763, Validation Accuracy:  0.740, Loss:  0.305
    Epoch  35 Batch   19/22 - Train Accuracy:  0.753, Validation Accuracy:  0.740, Loss:  0.316
    Epoch  35 Batch   20/22 - Train Accuracy:  0.745, Validation Accuracy:  0.743, Loss:  0.333
    Epoch  36 Batch    0/22 - Train Accuracy:  0.745, Validation Accuracy:  0.741, Loss:  0.330
    Epoch  36 Batch    1/22 - Train Accuracy:  0.745, Validation Accuracy:  0.741, Loss:  0.327
    Epoch  36 Batch    2/22 - Train Accuracy:  0.743, Validation Accuracy:  0.742, Loss:  0.331
    Epoch  36 Batch    3/22 - Train Accuracy:  0.748, Validation Accuracy:  0.741, Loss:  0.325
    Epoch  36 Batch    4/22 - Train Accuracy:  0.737, Validation Accuracy:  0.740, Loss:  0.330
    Epoch  36 Batch    5/22 - Train Accuracy:  0.736, Validation Accuracy:  0.734, Loss:  0.332
    Epoch  36 Batch    6/22 - Train Accuracy:  0.732, Validation Accuracy:  0.732, Loss:  0.339
    Epoch  36 Batch    7/22 - Train Accuracy:  0.697, Validation Accuracy:  0.710, Loss:  0.366
    Epoch  36 Batch    8/22 - Train Accuracy:  0.706, Validation Accuracy:  0.710, Loss:  0.388
    Epoch  36 Batch    9/22 - Train Accuracy:  0.688, Validation Accuracy:  0.696, Loss:  0.433
    Epoch  36 Batch   10/22 - Train Accuracy:  0.690, Validation Accuracy:  0.703, Loss:  0.437
    Epoch  36 Batch   11/22 - Train Accuracy:  0.703, Validation Accuracy:  0.702, Loss:  0.419
    Epoch  36 Batch   12/22 - Train Accuracy:  0.705, Validation Accuracy:  0.703, Loss:  0.424
    Epoch  36 Batch   13/22 - Train Accuracy:  0.696, Validation Accuracy:  0.694, Loss:  0.424
    Epoch  36 Batch   14/22 - Train Accuracy:  0.743, Validation Accuracy:  0.715, Loss:  0.370
    Epoch  36 Batch   15/22 - Train Accuracy:  0.698, Validation Accuracy:  0.709, Loss:  0.397
    Epoch  36 Batch   16/22 - Train Accuracy:  0.719, Validation Accuracy:  0.716, Loss:  0.379
    Epoch  36 Batch   17/22 - Train Accuracy:  0.717, Validation Accuracy:  0.714, Loss:  0.378
    Epoch  36 Batch   18/22 - Train Accuracy:  0.731, Validation Accuracy:  0.706, Loss:  0.344
    Epoch  36 Batch   19/22 - Train Accuracy:  0.733, Validation Accuracy:  0.722, Loss:  0.360
    Epoch  36 Batch   20/22 - Train Accuracy:  0.715, Validation Accuracy:  0.713, Loss:  0.361
    Epoch  37 Batch    0/22 - Train Accuracy:  0.718, Validation Accuracy:  0.718, Loss:  0.368
    Epoch  37 Batch    1/22 - Train Accuracy:  0.728, Validation Accuracy:  0.722, Loss:  0.358
    Epoch  37 Batch    2/22 - Train Accuracy:  0.726, Validation Accuracy:  0.721, Loss:  0.358
    Epoch  37 Batch    3/22 - Train Accuracy:  0.729, Validation Accuracy:  0.723, Loss:  0.354
    Epoch  37 Batch    4/22 - Train Accuracy:  0.723, Validation Accuracy:  0.728, Loss:  0.358
    Epoch  37 Batch    5/22 - Train Accuracy:  0.729, Validation Accuracy:  0.727, Loss:  0.349
    Epoch  37 Batch    6/22 - Train Accuracy:  0.724, Validation Accuracy:  0.726, Loss:  0.346
    Epoch  37 Batch    7/22 - Train Accuracy:  0.715, Validation Accuracy:  0.730, Loss:  0.364
    Epoch  37 Batch    8/22 - Train Accuracy:  0.726, Validation Accuracy:  0.730, Loss:  0.344
    Epoch  37 Batch    9/22 - Train Accuracy:  0.723, Validation Accuracy:  0.729, Loss:  0.353
    Epoch  37 Batch   10/22 - Train Accuracy:  0.721, Validation Accuracy:  0.737, Loss:  0.358
    Epoch  37 Batch   11/22 - Train Accuracy:  0.738, Validation Accuracy:  0.735, Loss:  0.334
    Epoch  37 Batch   12/22 - Train Accuracy:  0.740, Validation Accuracy:  0.737, Loss:  0.334
    Epoch  37 Batch   13/22 - Train Accuracy:  0.740, Validation Accuracy:  0.739, Loss:  0.335
    Epoch  37 Batch   14/22 - Train Accuracy:  0.767, Validation Accuracy:  0.741, Loss:  0.301
    Epoch  37 Batch   15/22 - Train Accuracy:  0.733, Validation Accuracy:  0.742, Loss:  0.343
    Epoch  37 Batch   16/22 - Train Accuracy:  0.747, Validation Accuracy:  0.743, Loss:  0.326
    Epoch  37 Batch   17/22 - Train Accuracy:  0.747, Validation Accuracy:  0.744, Loss:  0.326
    Epoch  37 Batch   18/22 - Train Accuracy:  0.765, Validation Accuracy:  0.741, Loss:  0.297
    Epoch  37 Batch   19/22 - Train Accuracy:  0.755, Validation Accuracy:  0.741, Loss:  0.305
    Epoch  37 Batch   20/22 - Train Accuracy:  0.745, Validation Accuracy:  0.745, Loss:  0.323
    Epoch  38 Batch    0/22 - Train Accuracy:  0.747, Validation Accuracy:  0.743, Loss:  0.320
    Epoch  38 Batch    1/22 - Train Accuracy:  0.751, Validation Accuracy:  0.745, Loss:  0.316
    Epoch  38 Batch    2/22 - Train Accuracy:  0.749, Validation Accuracy:  0.747, Loss:  0.318
    Epoch  38 Batch    3/22 - Train Accuracy:  0.755, Validation Accuracy:  0.747, Loss:  0.312
    Epoch  38 Batch    4/22 - Train Accuracy:  0.745, Validation Accuracy:  0.747, Loss:  0.315
    Epoch  38 Batch    5/22 - Train Accuracy:  0.754, Validation Accuracy:  0.747, Loss:  0.314
    Epoch  38 Batch    6/22 - Train Accuracy:  0.749, Validation Accuracy:  0.747, Loss:  0.313
    Epoch  38 Batch    7/22 - Train Accuracy:  0.738, Validation Accuracy:  0.748, Loss:  0.325
    Epoch  38 Batch    8/22 - Train Accuracy:  0.741, Validation Accuracy:  0.747, Loss:  0.312
    Epoch  38 Batch    9/22 - Train Accuracy:  0.742, Validation Accuracy:  0.748, Loss:  0.321
    Epoch  38 Batch   10/22 - Train Accuracy:  0.735, Validation Accuracy:  0.748, Loss:  0.324
    Epoch  38 Batch   11/22 - Train Accuracy:  0.755, Validation Accuracy:  0.749, Loss:  0.308
    Epoch  38 Batch   12/22 - Train Accuracy:  0.754, Validation Accuracy:  0.749, Loss:  0.308
    Epoch  38 Batch   13/22 - Train Accuracy:  0.753, Validation Accuracy:  0.752, Loss:  0.308
    Epoch  38 Batch   14/22 - Train Accuracy:  0.779, Validation Accuracy:  0.753, Loss:  0.278
    Epoch  38 Batch   15/22 - Train Accuracy:  0.748, Validation Accuracy:  0.754, Loss:  0.317
    Epoch  38 Batch   16/22 - Train Accuracy:  0.760, Validation Accuracy:  0.754, Loss:  0.303
    Epoch  38 Batch   17/22 - Train Accuracy:  0.759, Validation Accuracy:  0.754, Loss:  0.302
    Epoch  38 Batch   18/22 - Train Accuracy:  0.779, Validation Accuracy:  0.755, Loss:  0.278
    Epoch  38 Batch   19/22 - Train Accuracy:  0.770, Validation Accuracy:  0.756, Loss:  0.286
    Epoch  38 Batch   20/22 - Train Accuracy:  0.759, Validation Accuracy:  0.754, Loss:  0.301
    Epoch  39 Batch    0/22 - Train Accuracy:  0.762, Validation Accuracy:  0.757, Loss:  0.300
    Epoch  39 Batch    1/22 - Train Accuracy:  0.763, Validation Accuracy:  0.759, Loss:  0.295
    Epoch  39 Batch    2/22 - Train Accuracy:  0.763, Validation Accuracy:  0.758, Loss:  0.299
    Epoch  39 Batch    3/22 - Train Accuracy:  0.766, Validation Accuracy:  0.759, Loss:  0.294
    Epoch  39 Batch    4/22 - Train Accuracy:  0.759, Validation Accuracy:  0.761, Loss:  0.299
    Epoch  39 Batch    5/22 - Train Accuracy:  0.765, Validation Accuracy:  0.760, Loss:  0.296
    Epoch  39 Batch    6/22 - Train Accuracy:  0.762, Validation Accuracy:  0.762, Loss:  0.296
    Epoch  39 Batch    7/22 - Train Accuracy:  0.755, Validation Accuracy:  0.763, Loss:  0.308
    Epoch  39 Batch    8/22 - Train Accuracy:  0.761, Validation Accuracy:  0.764, Loss:  0.293
    Epoch  39 Batch    9/22 - Train Accuracy:  0.760, Validation Accuracy:  0.765, Loss:  0.304
    Epoch  39 Batch   10/22 - Train Accuracy:  0.755, Validation Accuracy:  0.766, Loss:  0.305
    Epoch  39 Batch   11/22 - Train Accuracy:  0.768, Validation Accuracy:  0.764, Loss:  0.290
    Epoch  39 Batch   12/22 - Train Accuracy:  0.770, Validation Accuracy:  0.765, Loss:  0.292
    Epoch  39 Batch   13/22 - Train Accuracy:  0.768, Validation Accuracy:  0.765, Loss:  0.292
    Epoch  39 Batch   14/22 - Train Accuracy:  0.792, Validation Accuracy:  0.767, Loss:  0.264
    Epoch  39 Batch   15/22 - Train Accuracy:  0.761, Validation Accuracy:  0.768, Loss:  0.301
    Epoch  39 Batch   16/22 - Train Accuracy:  0.774, Validation Accuracy:  0.770, Loss:  0.287
    Epoch  39 Batch   17/22 - Train Accuracy:  0.776, Validation Accuracy:  0.770, Loss:  0.288
    Epoch  39 Batch   18/22 - Train Accuracy:  0.793, Validation Accuracy:  0.770, Loss:  0.264
    Epoch  39 Batch   19/22 - Train Accuracy:  0.783, Validation Accuracy:  0.770, Loss:  0.272
    Epoch  39 Batch   20/22 - Train Accuracy:  0.774, Validation Accuracy:  0.772, Loss:  0.285
    Epoch  40 Batch    0/22 - Train Accuracy:  0.772, Validation Accuracy:  0.772, Loss:  0.285
    Epoch  40 Batch    1/22 - Train Accuracy:  0.778, Validation Accuracy:  0.774, Loss:  0.281
    Epoch  40 Batch    2/22 - Train Accuracy:  0.775, Validation Accuracy:  0.771, Loss:  0.285
    Epoch  40 Batch    3/22 - Train Accuracy:  0.780, Validation Accuracy:  0.774, Loss:  0.279
    Epoch  40 Batch    4/22 - Train Accuracy:  0.771, Validation Accuracy:  0.774, Loss:  0.283
    Epoch  40 Batch    5/22 - Train Accuracy:  0.781, Validation Accuracy:  0.776, Loss:  0.282
    Epoch  40 Batch    6/22 - Train Accuracy:  0.773, Validation Accuracy:  0.775, Loss:  0.281
    Epoch  40 Batch    7/22 - Train Accuracy:  0.767, Validation Accuracy:  0.776, Loss:  0.294
    Epoch  40 Batch    8/22 - Train Accuracy:  0.760, Validation Accuracy:  0.762, Loss:  0.282
    Epoch  40 Batch    9/22 - Train Accuracy:  0.719, Validation Accuracy:  0.729, Loss:  0.295
    Epoch  40 Batch   10/22 - Train Accuracy:  0.712, Validation Accuracy:  0.727, Loss:  0.318
    Epoch  40 Batch   11/22 - Train Accuracy:  0.725, Validation Accuracy:  0.721, Loss:  0.337
    Epoch  40 Batch   12/22 - Train Accuracy:  0.717, Validation Accuracy:  0.712, Loss:  0.358
    Epoch  40 Batch   13/22 - Train Accuracy:  0.683, Validation Accuracy:  0.685, Loss:  0.382
    Epoch  40 Batch   14/22 - Train Accuracy:  0.746, Validation Accuracy:  0.719, Loss:  0.408
    Epoch  40 Batch   15/22 - Train Accuracy:  0.685, Validation Accuracy:  0.695, Loss:  0.376
    Epoch  40 Batch   16/22 - Train Accuracy:  0.705, Validation Accuracy:  0.700, Loss:  0.433
    Epoch  40 Batch   17/22 - Train Accuracy:  0.718, Validation Accuracy:  0.719, Loss:  0.380
    Epoch  40 Batch   18/22 - Train Accuracy:  0.729, Validation Accuracy:  0.707, Loss:  0.334
    Epoch  40 Batch   19/22 - Train Accuracy:  0.745, Validation Accuracy:  0.733, Loss:  0.366
    Epoch  40 Batch   20/22 - Train Accuracy:  0.723, Validation Accuracy:  0.723, Loss:  0.331
    Epoch  41 Batch    0/22 - Train Accuracy:  0.728, Validation Accuracy:  0.728, Loss:  0.347
    Epoch  41 Batch    1/22 - Train Accuracy:  0.733, Validation Accuracy:  0.729, Loss:  0.349
    Epoch  41 Batch    2/22 - Train Accuracy:  0.742, Validation Accuracy:  0.744, Loss:  0.336
    Epoch  41 Batch    3/22 - Train Accuracy:  0.743, Validation Accuracy:  0.740, Loss:  0.314
    Epoch  41 Batch    4/22 - Train Accuracy:  0.726, Validation Accuracy:  0.731, Loss:  0.329
    Epoch  41 Batch    5/22 - Train Accuracy:  0.738, Validation Accuracy:  0.734, Loss:  0.328
    Epoch  41 Batch    6/22 - Train Accuracy:  0.741, Validation Accuracy:  0.741, Loss:  0.317
    Epoch  41 Batch    7/22 - Train Accuracy:  0.732, Validation Accuracy:  0.744, Loss:  0.320
    Epoch  41 Batch    8/22 - Train Accuracy:  0.737, Validation Accuracy:  0.740, Loss:  0.308
    Epoch  41 Batch    9/22 - Train Accuracy:  0.741, Validation Accuracy:  0.750, Loss:  0.319
    Epoch  41 Batch   10/22 - Train Accuracy:  0.737, Validation Accuracy:  0.751, Loss:  0.318
    Epoch  41 Batch   11/22 - Train Accuracy:  0.749, Validation Accuracy:  0.748, Loss:  0.296
    Epoch  41 Batch   12/22 - Train Accuracy:  0.757, Validation Accuracy:  0.756, Loss:  0.295
    Epoch  41 Batch   13/22 - Train Accuracy:  0.754, Validation Accuracy:  0.757, Loss:  0.293
    Epoch  41 Batch   14/22 - Train Accuracy:  0.781, Validation Accuracy:  0.762, Loss:  0.269
    Epoch  41 Batch   15/22 - Train Accuracy:  0.756, Validation Accuracy:  0.763, Loss:  0.301
    Epoch  41 Batch   16/22 - Train Accuracy:  0.768, Validation Accuracy:  0.763, Loss:  0.288
    Epoch  41 Batch   17/22 - Train Accuracy:  0.768, Validation Accuracy:  0.761, Loss:  0.283
    Epoch  41 Batch   18/22 - Train Accuracy:  0.785, Validation Accuracy:  0.762, Loss:  0.260
    Epoch  41 Batch   19/22 - Train Accuracy:  0.774, Validation Accuracy:  0.764, Loss:  0.269
    Epoch  41 Batch   20/22 - Train Accuracy:  0.769, Validation Accuracy:  0.768, Loss:  0.280
    Epoch  42 Batch    0/22 - Train Accuracy:  0.772, Validation Accuracy:  0.772, Loss:  0.276
    Epoch  42 Batch    1/22 - Train Accuracy:  0.776, Validation Accuracy:  0.775, Loss:  0.273
    Epoch  42 Batch    2/22 - Train Accuracy:  0.779, Validation Accuracy:  0.776, Loss:  0.277
    Epoch  42 Batch    3/22 - Train Accuracy:  0.781, Validation Accuracy:  0.777, Loss:  0.270
    Epoch  42 Batch    4/22 - Train Accuracy:  0.769, Validation Accuracy:  0.772, Loss:  0.274
    Epoch  42 Batch    5/22 - Train Accuracy:  0.772, Validation Accuracy:  0.770, Loss:  0.272
    Epoch  42 Batch    6/22 - Train Accuracy:  0.775, Validation Accuracy:  0.775, Loss:  0.271
    Epoch  42 Batch    7/22 - Train Accuracy:  0.769, Validation Accuracy:  0.778, Loss:  0.281
    Epoch  42 Batch    8/22 - Train Accuracy:  0.779, Validation Accuracy:  0.781, Loss:  0.268
    Epoch  42 Batch    9/22 - Train Accuracy:  0.778, Validation Accuracy:  0.784, Loss:  0.275
    Epoch  42 Batch   10/22 - Train Accuracy:  0.771, Validation Accuracy:  0.783, Loss:  0.277
    Epoch  42 Batch   11/22 - Train Accuracy:  0.786, Validation Accuracy:  0.785, Loss:  0.264
    Epoch  42 Batch   12/22 - Train Accuracy:  0.784, Validation Accuracy:  0.786, Loss:  0.263
    Epoch  42 Batch   13/22 - Train Accuracy:  0.785, Validation Accuracy:  0.785, Loss:  0.265
    Epoch  42 Batch   14/22 - Train Accuracy:  0.806, Validation Accuracy:  0.785, Loss:  0.238
    Epoch  42 Batch   15/22 - Train Accuracy:  0.777, Validation Accuracy:  0.784, Loss:  0.271
    Epoch  42 Batch   16/22 - Train Accuracy:  0.788, Validation Accuracy:  0.785, Loss:  0.260
    Epoch  42 Batch   17/22 - Train Accuracy:  0.793, Validation Accuracy:  0.787, Loss:  0.261
    Epoch  42 Batch   18/22 - Train Accuracy:  0.810, Validation Accuracy:  0.790, Loss:  0.237
    Epoch  42 Batch   19/22 - Train Accuracy:  0.800, Validation Accuracy:  0.791, Loss:  0.244
    Epoch  42 Batch   20/22 - Train Accuracy:  0.793, Validation Accuracy:  0.792, Loss:  0.258
    Epoch  43 Batch    0/22 - Train Accuracy:  0.793, Validation Accuracy:  0.793, Loss:  0.254
    Epoch  43 Batch    1/22 - Train Accuracy:  0.796, Validation Accuracy:  0.793, Loss:  0.252
    Epoch  43 Batch    2/22 - Train Accuracy:  0.795, Validation Accuracy:  0.791, Loss:  0.254
    Epoch  43 Batch    3/22 - Train Accuracy:  0.799, Validation Accuracy:  0.793, Loss:  0.250
    Epoch  43 Batch    4/22 - Train Accuracy:  0.793, Validation Accuracy:  0.795, Loss:  0.255
    Epoch  43 Batch    5/22 - Train Accuracy:  0.799, Validation Accuracy:  0.796, Loss:  0.253
    Epoch  43 Batch    6/22 - Train Accuracy:  0.795, Validation Accuracy:  0.798, Loss:  0.252
    Epoch  43 Batch    7/22 - Train Accuracy:  0.788, Validation Accuracy:  0.797, Loss:  0.262
    Epoch  43 Batch    8/22 - Train Accuracy:  0.797, Validation Accuracy:  0.798, Loss:  0.251
    Epoch  43 Batch    9/22 - Train Accuracy:  0.794, Validation Accuracy:  0.799, Loss:  0.257
    Epoch  43 Batch   10/22 - Train Accuracy:  0.786, Validation Accuracy:  0.799, Loss:  0.260
    Epoch  43 Batch   11/22 - Train Accuracy:  0.799, Validation Accuracy:  0.800, Loss:  0.247
    Epoch  43 Batch   12/22 - Train Accuracy:  0.799, Validation Accuracy:  0.800, Loss:  0.248
    Epoch  43 Batch   13/22 - Train Accuracy:  0.799, Validation Accuracy:  0.800, Loss:  0.250
    Epoch  43 Batch   14/22 - Train Accuracy:  0.820, Validation Accuracy:  0.800, Loss:  0.223
    Epoch  43 Batch   15/22 - Train Accuracy:  0.790, Validation Accuracy:  0.801, Loss:  0.255
    Epoch  43 Batch   16/22 - Train Accuracy:  0.802, Validation Accuracy:  0.803, Loss:  0.246
    Epoch  43 Batch   17/22 - Train Accuracy:  0.806, Validation Accuracy:  0.804, Loss:  0.244
    Epoch  43 Batch   18/22 - Train Accuracy:  0.820, Validation Accuracy:  0.805, Loss:  0.224
    Epoch  43 Batch   19/22 - Train Accuracy:  0.811, Validation Accuracy:  0.804, Loss:  0.230
    Epoch  43 Batch   20/22 - Train Accuracy:  0.801, Validation Accuracy:  0.804, Loss:  0.242
    Epoch  44 Batch    0/22 - Train Accuracy:  0.801, Validation Accuracy:  0.803, Loss:  0.240
    Epoch  44 Batch    1/22 - Train Accuracy:  0.805, Validation Accuracy:  0.806, Loss:  0.239
    Epoch  44 Batch    2/22 - Train Accuracy:  0.807, Validation Accuracy:  0.804, Loss:  0.242
    Epoch  44 Batch    3/22 - Train Accuracy:  0.808, Validation Accuracy:  0.805, Loss:  0.237
    Epoch  44 Batch    4/22 - Train Accuracy:  0.803, Validation Accuracy:  0.807, Loss:  0.240
    Epoch  44 Batch    5/22 - Train Accuracy:  0.806, Validation Accuracy:  0.804, Loss:  0.240
    Epoch  44 Batch    6/22 - Train Accuracy:  0.801, Validation Accuracy:  0.807, Loss:  0.238
    Epoch  44 Batch    7/22 - Train Accuracy:  0.792, Validation Accuracy:  0.807, Loss:  0.248
    Epoch  44 Batch    8/22 - Train Accuracy:  0.804, Validation Accuracy:  0.808, Loss:  0.237
    Epoch  44 Batch    9/22 - Train Accuracy:  0.801, Validation Accuracy:  0.809, Loss:  0.243
    Epoch  44 Batch   10/22 - Train Accuracy:  0.791, Validation Accuracy:  0.808, Loss:  0.248
    Epoch  44 Batch   11/22 - Train Accuracy:  0.804, Validation Accuracy:  0.807, Loss:  0.237
    Epoch  44 Batch   12/22 - Train Accuracy:  0.804, Validation Accuracy:  0.807, Loss:  0.240
    Epoch  44 Batch   13/22 - Train Accuracy:  0.803, Validation Accuracy:  0.808, Loss:  0.239
    Epoch  44 Batch   14/22 - Train Accuracy:  0.824, Validation Accuracy:  0.809, Loss:  0.212
    Epoch  44 Batch   15/22 - Train Accuracy:  0.794, Validation Accuracy:  0.804, Loss:  0.245
    Epoch  44 Batch   16/22 - Train Accuracy:  0.804, Validation Accuracy:  0.806, Loss:  0.239
    Epoch  44 Batch   17/22 - Train Accuracy:  0.811, Validation Accuracy:  0.809, Loss:  0.238
    Epoch  44 Batch   18/22 - Train Accuracy:  0.824, Validation Accuracy:  0.807, Loss:  0.214
    Epoch  44 Batch   19/22 - Train Accuracy:  0.808, Validation Accuracy:  0.806, Loss:  0.221
    Epoch  44 Batch   20/22 - Train Accuracy:  0.807, Validation Accuracy:  0.809, Loss:  0.233
    Epoch  45 Batch    0/22 - Train Accuracy:  0.809, Validation Accuracy:  0.811, Loss:  0.230
    Epoch  45 Batch    1/22 - Train Accuracy:  0.807, Validation Accuracy:  0.808, Loss:  0.227
    Epoch  45 Batch    2/22 - Train Accuracy:  0.812, Validation Accuracy:  0.810, Loss:  0.232
    Epoch  45 Batch    3/22 - Train Accuracy:  0.814, Validation Accuracy:  0.814, Loss:  0.226
    Epoch  45 Batch    4/22 - Train Accuracy:  0.805, Validation Accuracy:  0.812, Loss:  0.229
    Epoch  45 Batch    5/22 - Train Accuracy:  0.811, Validation Accuracy:  0.812, Loss:  0.229
    Epoch  45 Batch    6/22 - Train Accuracy:  0.809, Validation Accuracy:  0.814, Loss:  0.228
    Epoch  45 Batch    7/22 - Train Accuracy:  0.797, Validation Accuracy:  0.814, Loss:  0.237
    Epoch  45 Batch    8/22 - Train Accuracy:  0.809, Validation Accuracy:  0.813, Loss:  0.227
    Epoch  45 Batch    9/22 - Train Accuracy:  0.801, Validation Accuracy:  0.813, Loss:  0.232
    Epoch  45 Batch   10/22 - Train Accuracy:  0.795, Validation Accuracy:  0.813, Loss:  0.234
    Epoch  45 Batch   11/22 - Train Accuracy:  0.811, Validation Accuracy:  0.813, Loss:  0.225
    Epoch  45 Batch   12/22 - Train Accuracy:  0.811, Validation Accuracy:  0.814, Loss:  0.224
    Epoch  45 Batch   13/22 - Train Accuracy:  0.811, Validation Accuracy:  0.815, Loss:  0.227
    Epoch  45 Batch   14/22 - Train Accuracy:  0.831, Validation Accuracy:  0.816, Loss:  0.204
    Epoch  45 Batch   15/22 - Train Accuracy:  0.804, Validation Accuracy:  0.816, Loss:  0.230
    Epoch  45 Batch   16/22 - Train Accuracy:  0.816, Validation Accuracy:  0.818, Loss:  0.224
    Epoch  45 Batch   17/22 - Train Accuracy:  0.816, Validation Accuracy:  0.817, Loss:  0.222
    Epoch  45 Batch   18/22 - Train Accuracy:  0.826, Validation Accuracy:  0.817, Loss:  0.203
    Epoch  45 Batch   19/22 - Train Accuracy:  0.820, Validation Accuracy:  0.817, Loss:  0.208
    Epoch  45 Batch   20/22 - Train Accuracy:  0.813, Validation Accuracy:  0.816, Loss:  0.220
    Epoch  46 Batch    0/22 - Train Accuracy:  0.814, Validation Accuracy:  0.819, Loss:  0.218
    Epoch  46 Batch    1/22 - Train Accuracy:  0.818, Validation Accuracy:  0.818, Loss:  0.215
    Epoch  46 Batch    2/22 - Train Accuracy:  0.818, Validation Accuracy:  0.819, Loss:  0.219
    Epoch  46 Batch    3/22 - Train Accuracy:  0.819, Validation Accuracy:  0.820, Loss:  0.214
    Epoch  46 Batch    4/22 - Train Accuracy:  0.816, Validation Accuracy:  0.820, Loss:  0.217
    Epoch  46 Batch    5/22 - Train Accuracy:  0.820, Validation Accuracy:  0.822, Loss:  0.216
    Epoch  46 Batch    6/22 - Train Accuracy:  0.815, Validation Accuracy:  0.822, Loss:  0.214
    Epoch  46 Batch    7/22 - Train Accuracy:  0.807, Validation Accuracy:  0.823, Loss:  0.225
    Epoch  46 Batch    8/22 - Train Accuracy:  0.817, Validation Accuracy:  0.821, Loss:  0.214
    Epoch  46 Batch    9/22 - Train Accuracy:  0.810, Validation Accuracy:  0.821, Loss:  0.221
    Epoch  46 Batch   10/22 - Train Accuracy:  0.805, Validation Accuracy:  0.823, Loss:  0.223
    Epoch  46 Batch   11/22 - Train Accuracy:  0.818, Validation Accuracy:  0.821, Loss:  0.212
    Epoch  46 Batch   12/22 - Train Accuracy:  0.820, Validation Accuracy:  0.821, Loss:  0.212
    Epoch  46 Batch   13/22 - Train Accuracy:  0.817, Validation Accuracy:  0.823, Loss:  0.214
    Epoch  46 Batch   14/22 - Train Accuracy:  0.836, Validation Accuracy:  0.824, Loss:  0.193
    Epoch  46 Batch   15/22 - Train Accuracy:  0.812, Validation Accuracy:  0.825, Loss:  0.217
    Epoch  46 Batch   16/22 - Train Accuracy:  0.821, Validation Accuracy:  0.825, Loss:  0.210
    Epoch  46 Batch   17/22 - Train Accuracy:  0.826, Validation Accuracy:  0.826, Loss:  0.211
    Epoch  46 Batch   18/22 - Train Accuracy:  0.835, Validation Accuracy:  0.826, Loss:  0.192
    Epoch  46 Batch   19/22 - Train Accuracy:  0.829, Validation Accuracy:  0.826, Loss:  0.199
    Epoch  46 Batch   20/22 - Train Accuracy:  0.821, Validation Accuracy:  0.825, Loss:  0.208
    Epoch  47 Batch    0/22 - Train Accuracy:  0.820, Validation Accuracy:  0.826, Loss:  0.207
    Epoch  47 Batch    1/22 - Train Accuracy:  0.827, Validation Accuracy:  0.826, Loss:  0.204
    Epoch  47 Batch    2/22 - Train Accuracy:  0.822, Validation Accuracy:  0.824, Loss:  0.208
    Epoch  47 Batch    3/22 - Train Accuracy:  0.828, Validation Accuracy:  0.826, Loss:  0.203
    Epoch  47 Batch    4/22 - Train Accuracy:  0.818, Validation Accuracy:  0.824, Loss:  0.207
    Epoch  47 Batch    5/22 - Train Accuracy:  0.825, Validation Accuracy:  0.826, Loss:  0.206
    Epoch  47 Batch    6/22 - Train Accuracy:  0.816, Validation Accuracy:  0.823, Loss:  0.206
    Epoch  47 Batch    7/22 - Train Accuracy:  0.813, Validation Accuracy:  0.827, Loss:  0.216
    Epoch  47 Batch    8/22 - Train Accuracy:  0.815, Validation Accuracy:  0.823, Loss:  0.205
    Epoch  47 Batch    9/22 - Train Accuracy:  0.821, Validation Accuracy:  0.826, Loss:  0.211
    Epoch  47 Batch   10/22 - Train Accuracy:  0.803, Validation Accuracy:  0.822, Loss:  0.215
    Epoch  47 Batch   11/22 - Train Accuracy:  0.826, Validation Accuracy:  0.827, Loss:  0.203
    Epoch  47 Batch   12/22 - Train Accuracy:  0.826, Validation Accuracy:  0.830, Loss:  0.203
    Epoch  47 Batch   13/22 - Train Accuracy:  0.826, Validation Accuracy:  0.833, Loss:  0.204
    Epoch  47 Batch   14/22 - Train Accuracy:  0.845, Validation Accuracy:  0.833, Loss:  0.183
    Epoch  47 Batch   15/22 - Train Accuracy:  0.815, Validation Accuracy:  0.829, Loss:  0.209
    Epoch  47 Batch   16/22 - Train Accuracy:  0.830, Validation Accuracy:  0.832, Loss:  0.202
    Epoch  47 Batch   17/22 - Train Accuracy:  0.824, Validation Accuracy:  0.828, Loss:  0.202
    Epoch  47 Batch   18/22 - Train Accuracy:  0.841, Validation Accuracy:  0.833, Loss:  0.184
    Epoch  47 Batch   19/22 - Train Accuracy:  0.838, Validation Accuracy:  0.833, Loss:  0.189
    Epoch  47 Batch   20/22 - Train Accuracy:  0.820, Validation Accuracy:  0.827, Loss:  0.199
    Epoch  48 Batch    0/22 - Train Accuracy:  0.830, Validation Accuracy:  0.832, Loss:  0.198
    Epoch  48 Batch    1/22 - Train Accuracy:  0.827, Validation Accuracy:  0.829, Loss:  0.196
    Epoch  48 Batch    2/22 - Train Accuracy:  0.832, Validation Accuracy:  0.832, Loss:  0.198
    Epoch  48 Batch    3/22 - Train Accuracy:  0.834, Validation Accuracy:  0.833, Loss:  0.193
    Epoch  48 Batch    4/22 - Train Accuracy:  0.826, Validation Accuracy:  0.830, Loss:  0.197
    Epoch  48 Batch    5/22 - Train Accuracy:  0.831, Validation Accuracy:  0.834, Loss:  0.197
    Epoch  48 Batch    6/22 - Train Accuracy:  0.824, Validation Accuracy:  0.830, Loss:  0.196
    Epoch  48 Batch    7/22 - Train Accuracy:  0.824, Validation Accuracy:  0.835, Loss:  0.206
    Epoch  48 Batch    8/22 - Train Accuracy:  0.830, Validation Accuracy:  0.833, Loss:  0.196
    Epoch  48 Batch    9/22 - Train Accuracy:  0.827, Validation Accuracy:  0.837, Loss:  0.202
    Epoch  48 Batch   10/22 - Train Accuracy:  0.824, Validation Accuracy:  0.839, Loss:  0.202
    Epoch  48 Batch   11/22 - Train Accuracy:  0.827, Validation Accuracy:  0.832, Loss:  0.192
    Epoch  48 Batch   12/22 - Train Accuracy:  0.838, Validation Accuracy:  0.839, Loss:  0.193
    Epoch  48 Batch   13/22 - Train Accuracy:  0.827, Validation Accuracy:  0.833, Loss:  0.195
    Epoch  48 Batch   14/22 - Train Accuracy:  0.849, Validation Accuracy:  0.837, Loss:  0.175
    Epoch  48 Batch   15/22 - Train Accuracy:  0.817, Validation Accuracy:  0.833, Loss:  0.201
    Epoch  48 Batch   16/22 - Train Accuracy:  0.836, Validation Accuracy:  0.836, Loss:  0.195
    Epoch  48 Batch   17/22 - Train Accuracy:  0.832, Validation Accuracy:  0.833, Loss:  0.195
    Epoch  48 Batch   18/22 - Train Accuracy:  0.848, Validation Accuracy:  0.842, Loss:  0.178
    Epoch  48 Batch   19/22 - Train Accuracy:  0.845, Validation Accuracy:  0.841, Loss:  0.182
    Epoch  48 Batch   20/22 - Train Accuracy:  0.831, Validation Accuracy:  0.837, Loss:  0.190
    Epoch  49 Batch    0/22 - Train Accuracy:  0.834, Validation Accuracy:  0.837, Loss:  0.190
    Epoch  49 Batch    1/22 - Train Accuracy:  0.807, Validation Accuracy:  0.813, Loss:  0.197
    Epoch  49 Batch    2/22 - Train Accuracy:  0.821, Validation Accuracy:  0.820, Loss:  0.207
    Epoch  49 Batch    3/22 - Train Accuracy:  0.782, Validation Accuracy:  0.782, Loss:  0.202
    Epoch  49 Batch    4/22 - Train Accuracy:  0.799, Validation Accuracy:  0.802, Loss:  0.220
    Epoch  49 Batch    5/22 - Train Accuracy:  0.786, Validation Accuracy:  0.788, Loss:  0.224
    Epoch  49 Batch    6/22 - Train Accuracy:  0.748, Validation Accuracy:  0.752, Loss:  0.270
    Epoch  49 Batch    7/22 - Train Accuracy:  0.731, Validation Accuracy:  0.743, Loss:  0.447
    Epoch  49 Batch    8/22 - Train Accuracy:  0.794, Validation Accuracy:  0.803, Loss:  0.461
    Epoch  49 Batch    9/22 - Train Accuracy:  0.763, Validation Accuracy:  0.776, Loss:  0.292
    Epoch  49 Batch   10/22 - Train Accuracy:  0.773, Validation Accuracy:  0.794, Loss:  0.336
    Epoch  49 Batch   11/22 - Train Accuracy:  0.783, Validation Accuracy:  0.787, Loss:  0.275
    Epoch  49 Batch   12/22 - Train Accuracy:  0.794, Validation Accuracy:  0.794, Loss:  0.286
    Epoch  49 Batch   13/22 - Train Accuracy:  0.784, Validation Accuracy:  0.791, Loss:  0.276
    Epoch  49 Batch   14/22 - Train Accuracy:  0.805, Validation Accuracy:  0.792, Loss:  0.235
    Epoch  49 Batch   15/22 - Train Accuracy:  0.787, Validation Accuracy:  0.802, Loss:  0.274
    Epoch  49 Batch   16/22 - Train Accuracy:  0.797, Validation Accuracy:  0.798, Loss:  0.243
    Epoch  49 Batch   17/22 - Train Accuracy:  0.803, Validation Accuracy:  0.804, Loss:  0.252
    Epoch  49 Batch   18/22 - Train Accuracy:  0.815, Validation Accuracy:  0.801, Loss:  0.221
    Epoch  49 Batch   19/22 - Train Accuracy:  0.803, Validation Accuracy:  0.799, Loss:  0.222
    Epoch  49 Batch   20/22 - Train Accuracy:  0.804, Validation Accuracy:  0.810, Loss:  0.232
    Epoch  50 Batch    0/22 - Train Accuracy:  0.814, Validation Accuracy:  0.817, Loss:  0.224
    Epoch  50 Batch    1/22 - Train Accuracy:  0.818, Validation Accuracy:  0.819, Loss:  0.217
    Epoch  50 Batch    2/22 - Train Accuracy:  0.822, Validation Accuracy:  0.823, Loss:  0.219
    Epoch  50 Batch    3/22 - Train Accuracy:  0.820, Validation Accuracy:  0.821, Loss:  0.212
    Epoch  50 Batch    4/22 - Train Accuracy:  0.815, Validation Accuracy:  0.815, Loss:  0.211
    Epoch  50 Batch    5/22 - Train Accuracy:  0.818, Validation Accuracy:  0.819, Loss:  0.210
    Epoch  50 Batch    6/22 - Train Accuracy:  0.822, Validation Accuracy:  0.826, Loss:  0.206
    Epoch  50 Batch    7/22 - Train Accuracy:  0.814, Validation Accuracy:  0.830, Loss:  0.213
    Epoch  50 Batch    8/22 - Train Accuracy:  0.830, Validation Accuracy:  0.834, Loss:  0.201
    Epoch  50 Batch    9/22 - Train Accuracy:  0.825, Validation Accuracy:  0.836, Loss:  0.206
    Epoch  50 Batch   10/22 - Train Accuracy:  0.818, Validation Accuracy:  0.833, Loss:  0.208
    Epoch  50 Batch   11/22 - Train Accuracy:  0.835, Validation Accuracy:  0.841, Loss:  0.194
    Epoch  50 Batch   12/22 - Train Accuracy:  0.839, Validation Accuracy:  0.845, Loss:  0.194
    Epoch  50 Batch   13/22 - Train Accuracy:  0.838, Validation Accuracy:  0.843, Loss:  0.195
    Epoch  50 Batch   14/22 - Train Accuracy:  0.855, Validation Accuracy:  0.844, Loss:  0.175
    Epoch  50 Batch   15/22 - Train Accuracy:  0.834, Validation Accuracy:  0.848, Loss:  0.197
    Epoch  50 Batch   16/22 - Train Accuracy:  0.844, Validation Accuracy:  0.847, Loss:  0.189
    Epoch  50 Batch   17/22 - Train Accuracy:  0.847, Validation Accuracy:  0.848, Loss:  0.190
    Epoch  50 Batch   18/22 - Train Accuracy:  0.856, Validation Accuracy:  0.850, Loss:  0.172
    Epoch  50 Batch   19/22 - Train Accuracy:  0.851, Validation Accuracy:  0.849, Loss:  0.176
    Epoch  50 Batch   20/22 - Train Accuracy:  0.841, Validation Accuracy:  0.849, Loss:  0.185
    Epoch  51 Batch    0/22 - Train Accuracy:  0.845, Validation Accuracy:  0.850, Loss:  0.182
    Epoch  51 Batch    1/22 - Train Accuracy:  0.851, Validation Accuracy:  0.850, Loss:  0.179
    Epoch  51 Batch    2/22 - Train Accuracy:  0.846, Validation Accuracy:  0.846, Loss:  0.183
    Epoch  51 Batch    3/22 - Train Accuracy:  0.853, Validation Accuracy:  0.853, Loss:  0.178
    Epoch  51 Batch    4/22 - Train Accuracy:  0.851, Validation Accuracy:  0.853, Loss:  0.180
    Epoch  51 Batch    5/22 - Train Accuracy:  0.851, Validation Accuracy:  0.854, Loss:  0.180
    Epoch  51 Batch    6/22 - Train Accuracy:  0.850, Validation Accuracy:  0.855, Loss:  0.179
    Epoch  51 Batch    7/22 - Train Accuracy:  0.843, Validation Accuracy:  0.856, Loss:  0.185
    Epoch  51 Batch    8/22 - Train Accuracy:  0.855, Validation Accuracy:  0.859, Loss:  0.177
    Epoch  51 Batch    9/22 - Train Accuracy:  0.849, Validation Accuracy:  0.861, Loss:  0.181
    Epoch  51 Batch   10/22 - Train Accuracy:  0.843, Validation Accuracy:  0.858, Loss:  0.184
    Epoch  51 Batch   11/22 - Train Accuracy:  0.853, Validation Accuracy:  0.859, Loss:  0.173
    Epoch  51 Batch   12/22 - Train Accuracy:  0.854, Validation Accuracy:  0.860, Loss:  0.173
    Epoch  51 Batch   13/22 - Train Accuracy:  0.855, Validation Accuracy:  0.860, Loss:  0.175
    Epoch  51 Batch   14/22 - Train Accuracy:  0.872, Validation Accuracy:  0.864, Loss:  0.157
    Epoch  51 Batch   15/22 - Train Accuracy:  0.852, Validation Accuracy:  0.865, Loss:  0.179
    Epoch  51 Batch   16/22 - Train Accuracy:  0.859, Validation Accuracy:  0.861, Loss:  0.172
    Epoch  51 Batch   17/22 - Train Accuracy:  0.860, Validation Accuracy:  0.864, Loss:  0.172
    Epoch  51 Batch   18/22 - Train Accuracy:  0.869, Validation Accuracy:  0.864, Loss:  0.157
    Epoch  51 Batch   19/22 - Train Accuracy:  0.865, Validation Accuracy:  0.862, Loss:  0.162
    Epoch  51 Batch   20/22 - Train Accuracy:  0.862, Validation Accuracy:  0.866, Loss:  0.170
    Epoch  52 Batch    0/22 - Train Accuracy:  0.857, Validation Accuracy:  0.865, Loss:  0.168
    Epoch  52 Batch    1/22 - Train Accuracy:  0.862, Validation Accuracy:  0.863, Loss:  0.165
    Epoch  52 Batch    2/22 - Train Accuracy:  0.861, Validation Accuracy:  0.865, Loss:  0.168
    Epoch  52 Batch    3/22 - Train Accuracy:  0.862, Validation Accuracy:  0.864, Loss:  0.164
    Epoch  52 Batch    4/22 - Train Accuracy:  0.863, Validation Accuracy:  0.867, Loss:  0.168
    Epoch  52 Batch    5/22 - Train Accuracy:  0.861, Validation Accuracy:  0.869, Loss:  0.167
    Epoch  52 Batch    6/22 - Train Accuracy:  0.861, Validation Accuracy:  0.865, Loss:  0.166
    Epoch  52 Batch    7/22 - Train Accuracy:  0.858, Validation Accuracy:  0.870, Loss:  0.175
    Epoch  52 Batch    8/22 - Train Accuracy:  0.868, Validation Accuracy:  0.869, Loss:  0.166
    Epoch  52 Batch    9/22 - Train Accuracy:  0.857, Validation Accuracy:  0.867, Loss:  0.170
    Epoch  52 Batch   10/22 - Train Accuracy:  0.852, Validation Accuracy:  0.869, Loss:  0.172
    Epoch  52 Batch   11/22 - Train Accuracy:  0.863, Validation Accuracy:  0.868, Loss:  0.163
    Epoch  52 Batch   12/22 - Train Accuracy:  0.865, Validation Accuracy:  0.870, Loss:  0.162
    Epoch  52 Batch   13/22 - Train Accuracy:  0.866, Validation Accuracy:  0.871, Loss:  0.165
    Epoch  52 Batch   14/22 - Train Accuracy:  0.876, Validation Accuracy:  0.869, Loss:  0.147
    Epoch  52 Batch   15/22 - Train Accuracy:  0.862, Validation Accuracy:  0.872, Loss:  0.169
    Epoch  52 Batch   16/22 - Train Accuracy:  0.867, Validation Accuracy:  0.870, Loss:  0.162
    Epoch  52 Batch   17/22 - Train Accuracy:  0.869, Validation Accuracy:  0.872, Loss:  0.162
    Epoch  52 Batch   18/22 - Train Accuracy:  0.875, Validation Accuracy:  0.871, Loss:  0.148
    Epoch  52 Batch   19/22 - Train Accuracy:  0.872, Validation Accuracy:  0.869, Loss:  0.151
    Epoch  52 Batch   20/22 - Train Accuracy:  0.869, Validation Accuracy:  0.873, Loss:  0.161
    Epoch  53 Batch    0/22 - Train Accuracy:  0.862, Validation Accuracy:  0.870, Loss:  0.160
    Epoch  53 Batch    1/22 - Train Accuracy:  0.872, Validation Accuracy:  0.873, Loss:  0.156
    Epoch  53 Batch    2/22 - Train Accuracy:  0.868, Validation Accuracy:  0.871, Loss:  0.160
    Epoch  53 Batch    3/22 - Train Accuracy:  0.872, Validation Accuracy:  0.874, Loss:  0.156
    Epoch  53 Batch    4/22 - Train Accuracy:  0.870, Validation Accuracy:  0.873, Loss:  0.158
    Epoch  53 Batch    5/22 - Train Accuracy:  0.869, Validation Accuracy:  0.874, Loss:  0.158
    Epoch  53 Batch    6/22 - Train Accuracy:  0.868, Validation Accuracy:  0.871, Loss:  0.158
    Epoch  53 Batch    7/22 - Train Accuracy:  0.864, Validation Accuracy:  0.876, Loss:  0.165
    Epoch  53 Batch    8/22 - Train Accuracy:  0.870, Validation Accuracy:  0.874, Loss:  0.157
    Epoch  53 Batch    9/22 - Train Accuracy:  0.863, Validation Accuracy:  0.874, Loss:  0.161
    Epoch  53 Batch   10/22 - Train Accuracy:  0.857, Validation Accuracy:  0.873, Loss:  0.164
    Epoch  53 Batch   11/22 - Train Accuracy:  0.868, Validation Accuracy:  0.874, Loss:  0.156
    Epoch  53 Batch   12/22 - Train Accuracy:  0.872, Validation Accuracy:  0.875, Loss:  0.155
    Epoch  53 Batch   13/22 - Train Accuracy:  0.869, Validation Accuracy:  0.872, Loss:  0.157
    Epoch  53 Batch   14/22 - Train Accuracy:  0.886, Validation Accuracy:  0.877, Loss:  0.141
    Epoch  53 Batch   15/22 - Train Accuracy:  0.861, Validation Accuracy:  0.874, Loss:  0.161
    Epoch  53 Batch   16/22 - Train Accuracy:  0.875, Validation Accuracy:  0.878, Loss:  0.155
    Epoch  53 Batch   17/22 - Train Accuracy:  0.870, Validation Accuracy:  0.875, Loss:  0.156
    Epoch  53 Batch   18/22 - Train Accuracy:  0.881, Validation Accuracy:  0.877, Loss:  0.142
    Epoch  53 Batch   19/22 - Train Accuracy:  0.879, Validation Accuracy:  0.877, Loss:  0.146
    Epoch  53 Batch   20/22 - Train Accuracy:  0.868, Validation Accuracy:  0.870, Loss:  0.154
    Epoch  54 Batch    0/22 - Train Accuracy:  0.866, Validation Accuracy:  0.871, Loss:  0.152
    Epoch  54 Batch    1/22 - Train Accuracy:  0.852, Validation Accuracy:  0.855, Loss:  0.152
    Epoch  54 Batch    2/22 - Train Accuracy:  0.861, Validation Accuracy:  0.865, Loss:  0.157
    Epoch  54 Batch    3/22 - Train Accuracy:  0.847, Validation Accuracy:  0.847, Loss:  0.156
    Epoch  54 Batch    4/22 - Train Accuracy:  0.864, Validation Accuracy:  0.868, Loss:  0.160
    Epoch  54 Batch    5/22 - Train Accuracy:  0.872, Validation Accuracy:  0.879, Loss:  0.156
    Epoch  54 Batch    6/22 - Train Accuracy:  0.869, Validation Accuracy:  0.871, Loss:  0.150
    Epoch  54 Batch    7/22 - Train Accuracy:  0.862, Validation Accuracy:  0.870, Loss:  0.160
    Epoch  54 Batch    8/22 - Train Accuracy:  0.863, Validation Accuracy:  0.868, Loss:  0.155
    Epoch  54 Batch    9/22 - Train Accuracy:  0.871, Validation Accuracy:  0.879, Loss:  0.157
    Epoch  54 Batch   10/22 - Train Accuracy:  0.865, Validation Accuracy:  0.877, Loss:  0.157
    Epoch  54 Batch   11/22 - Train Accuracy:  0.863, Validation Accuracy:  0.869, Loss:  0.149
    Epoch  54 Batch   12/22 - Train Accuracy:  0.875, Validation Accuracy:  0.879, Loss:  0.152
    Epoch  54 Batch   13/22 - Train Accuracy:  0.878, Validation Accuracy:  0.882, Loss:  0.150
    Epoch  54 Batch   14/22 - Train Accuracy:  0.883, Validation Accuracy:  0.875, Loss:  0.133
    Epoch  54 Batch   15/22 - Train Accuracy:  0.870, Validation Accuracy:  0.880, Loss:  0.153
    Epoch  54 Batch   16/22 - Train Accuracy:  0.877, Validation Accuracy:  0.883, Loss:  0.148
    Epoch  54 Batch   17/22 - Train Accuracy:  0.878, Validation Accuracy:  0.881, Loss:  0.148
    Epoch  54 Batch   18/22 - Train Accuracy:  0.886, Validation Accuracy:  0.881, Loss:  0.135
    Epoch  54 Batch   19/22 - Train Accuracy:  0.882, Validation Accuracy:  0.882, Loss:  0.140
    Epoch  54 Batch   20/22 - Train Accuracy:  0.879, Validation Accuracy:  0.882, Loss:  0.147
    Epoch  55 Batch    0/22 - Train Accuracy:  0.877, Validation Accuracy:  0.882, Loss:  0.145
    Epoch  55 Batch    1/22 - Train Accuracy:  0.877, Validation Accuracy:  0.878, Loss:  0.143
    Epoch  55 Batch    2/22 - Train Accuracy:  0.878, Validation Accuracy:  0.882, Loss:  0.146
    Epoch  55 Batch    3/22 - Train Accuracy:  0.882, Validation Accuracy:  0.883, Loss:  0.142
    Epoch  55 Batch    4/22 - Train Accuracy:  0.875, Validation Accuracy:  0.879, Loss:  0.144
    Epoch  55 Batch    5/22 - Train Accuracy:  0.877, Validation Accuracy:  0.882, Loss:  0.145
    Epoch  55 Batch    6/22 - Train Accuracy:  0.879, Validation Accuracy:  0.883, Loss:  0.145
    Epoch  55 Batch    7/22 - Train Accuracy:  0.871, Validation Accuracy:  0.883, Loss:  0.150
    Epoch  55 Batch    8/22 - Train Accuracy:  0.880, Validation Accuracy:  0.883, Loss:  0.141
    Epoch  55 Batch    9/22 - Train Accuracy:  0.870, Validation Accuracy:  0.879, Loss:  0.146
    Epoch  55 Batch   10/22 - Train Accuracy:  0.869, Validation Accuracy:  0.883, Loss:  0.150
    Epoch  55 Batch   11/22 - Train Accuracy:  0.879, Validation Accuracy:  0.885, Loss:  0.141
    Epoch  55 Batch   12/22 - Train Accuracy:  0.880, Validation Accuracy:  0.885, Loss:  0.140
    Epoch  55 Batch   13/22 - Train Accuracy:  0.880, Validation Accuracy:  0.885, Loss:  0.142
    Epoch  55 Batch   14/22 - Train Accuracy:  0.893, Validation Accuracy:  0.886, Loss:  0.128
    Epoch  55 Batch   15/22 - Train Accuracy:  0.877, Validation Accuracy:  0.887, Loss:  0.145
    Epoch  55 Batch   16/22 - Train Accuracy:  0.884, Validation Accuracy:  0.886, Loss:  0.140
    Epoch  55 Batch   17/22 - Train Accuracy:  0.882, Validation Accuracy:  0.885, Loss:  0.141
    Epoch  55 Batch   18/22 - Train Accuracy:  0.892, Validation Accuracy:  0.887, Loss:  0.128
    Epoch  55 Batch   19/22 - Train Accuracy:  0.886, Validation Accuracy:  0.887, Loss:  0.132
    Epoch  55 Batch   20/22 - Train Accuracy:  0.882, Validation Accuracy:  0.886, Loss:  0.139
    Epoch  56 Batch    0/22 - Train Accuracy:  0.879, Validation Accuracy:  0.886, Loss:  0.137
    Epoch  56 Batch    1/22 - Train Accuracy:  0.887, Validation Accuracy:  0.887, Loss:  0.134
    Epoch  56 Batch    2/22 - Train Accuracy:  0.882, Validation Accuracy:  0.888, Loss:  0.139
    Epoch  56 Batch    3/22 - Train Accuracy:  0.884, Validation Accuracy:  0.888, Loss:  0.134
    Epoch  56 Batch    4/22 - Train Accuracy:  0.886, Validation Accuracy:  0.889, Loss:  0.138
    Epoch  56 Batch    5/22 - Train Accuracy:  0.883, Validation Accuracy:  0.890, Loss:  0.138
    Epoch  56 Batch    6/22 - Train Accuracy:  0.886, Validation Accuracy:  0.890, Loss:  0.134
    Epoch  56 Batch    7/22 - Train Accuracy:  0.879, Validation Accuracy:  0.890, Loss:  0.143
    Epoch  56 Batch    8/22 - Train Accuracy:  0.886, Validation Accuracy:  0.890, Loss:  0.135
    Epoch  56 Batch    9/22 - Train Accuracy:  0.880, Validation Accuracy:  0.890, Loss:  0.140
    Epoch  56 Batch   10/22 - Train Accuracy:  0.869, Validation Accuracy:  0.884, Loss:  0.142
    Epoch  56 Batch   11/22 - Train Accuracy:  0.878, Validation Accuracy:  0.885, Loss:  0.134
    Epoch  56 Batch   12/22 - Train Accuracy:  0.877, Validation Accuracy:  0.880, Loss:  0.135
    Epoch  56 Batch   13/22 - Train Accuracy:  0.879, Validation Accuracy:  0.884, Loss:  0.139
    Epoch  56 Batch   14/22 - Train Accuracy:  0.891, Validation Accuracy:  0.886, Loss:  0.123
    Epoch  56 Batch   15/22 - Train Accuracy:  0.879, Validation Accuracy:  0.890, Loss:  0.139
    Epoch  56 Batch   16/22 - Train Accuracy:  0.886, Validation Accuracy:  0.890, Loss:  0.134
    Epoch  56 Batch   17/22 - Train Accuracy:  0.888, Validation Accuracy:  0.892, Loss:  0.134
    Epoch  56 Batch   18/22 - Train Accuracy:  0.895, Validation Accuracy:  0.891, Loss:  0.122
    Epoch  56 Batch   19/22 - Train Accuracy:  0.890, Validation Accuracy:  0.890, Loss:  0.125
    Epoch  56 Batch   20/22 - Train Accuracy:  0.884, Validation Accuracy:  0.888, Loss:  0.132
    Epoch  57 Batch    0/22 - Train Accuracy:  0.876, Validation Accuracy:  0.886, Loss:  0.131
    Epoch  57 Batch    1/22 - Train Accuracy:  0.887, Validation Accuracy:  0.888, Loss:  0.128
    Epoch  57 Batch    2/22 - Train Accuracy:  0.883, Validation Accuracy:  0.887, Loss:  0.134
    Epoch  57 Batch    3/22 - Train Accuracy:  0.889, Validation Accuracy:  0.891, Loss:  0.130
    Epoch  57 Batch    4/22 - Train Accuracy:  0.882, Validation Accuracy:  0.887, Loss:  0.132
    Epoch  57 Batch    5/22 - Train Accuracy:  0.885, Validation Accuracy:  0.889, Loss:  0.132
    Epoch  57 Batch    6/22 - Train Accuracy:  0.888, Validation Accuracy:  0.891, Loss:  0.130
    Epoch  57 Batch    7/22 - Train Accuracy:  0.880, Validation Accuracy:  0.892, Loss:  0.136
    Epoch  57 Batch    8/22 - Train Accuracy:  0.886, Validation Accuracy:  0.890, Loss:  0.130
    Epoch  57 Batch    9/22 - Train Accuracy:  0.878, Validation Accuracy:  0.888, Loss:  0.134
    Epoch  57 Batch   10/22 - Train Accuracy:  0.875, Validation Accuracy:  0.888, Loss:  0.137
    Epoch  57 Batch   11/22 - Train Accuracy:  0.881, Validation Accuracy:  0.885, Loss:  0.130
    Epoch  57 Batch   12/22 - Train Accuracy:  0.886, Validation Accuracy:  0.892, Loss:  0.129
    Epoch  57 Batch   13/22 - Train Accuracy:  0.890, Validation Accuracy:  0.893, Loss:  0.131
    Epoch  57 Batch   14/22 - Train Accuracy:  0.899, Validation Accuracy:  0.895, Loss:  0.115
    Epoch  57 Batch   15/22 - Train Accuracy:  0.880, Validation Accuracy:  0.892, Loss:  0.130
    Epoch  57 Batch   16/22 - Train Accuracy:  0.883, Validation Accuracy:  0.887, Loss:  0.128
    Epoch  57 Batch   17/22 - Train Accuracy:  0.889, Validation Accuracy:  0.892, Loss:  0.129
    Epoch  57 Batch   18/22 - Train Accuracy:  0.898, Validation Accuracy:  0.894, Loss:  0.119
    Epoch  57 Batch   19/22 - Train Accuracy:  0.893, Validation Accuracy:  0.893, Loss:  0.120
    Epoch  57 Batch   20/22 - Train Accuracy:  0.885, Validation Accuracy:  0.892, Loss:  0.125
    Epoch  58 Batch    0/22 - Train Accuracy:  0.876, Validation Accuracy:  0.885, Loss:  0.127
    Epoch  58 Batch    1/22 - Train Accuracy:  0.890, Validation Accuracy:  0.890, Loss:  0.125
    Epoch  58 Batch    2/22 - Train Accuracy:  0.889, Validation Accuracy:  0.893, Loss:  0.128
    Epoch  58 Batch    3/22 - Train Accuracy:  0.892, Validation Accuracy:  0.897, Loss:  0.122
    Epoch  58 Batch    4/22 - Train Accuracy:  0.890, Validation Accuracy:  0.894, Loss:  0.125
    Epoch  58 Batch    5/22 - Train Accuracy:  0.888, Validation Accuracy:  0.895, Loss:  0.125
    Epoch  58 Batch    6/22 - Train Accuracy:  0.892, Validation Accuracy:  0.895, Loss:  0.123
    Epoch  58 Batch    7/22 - Train Accuracy:  0.886, Validation Accuracy:  0.897, Loss:  0.130
    Epoch  58 Batch    8/22 - Train Accuracy:  0.891, Validation Accuracy:  0.896, Loss:  0.123
    Epoch  58 Batch    9/22 - Train Accuracy:  0.886, Validation Accuracy:  0.892, Loss:  0.128
    Epoch  58 Batch   10/22 - Train Accuracy:  0.875, Validation Accuracy:  0.889, Loss:  0.129
    Epoch  58 Batch   11/22 - Train Accuracy:  0.888, Validation Accuracy:  0.895, Loss:  0.123
    Epoch  58 Batch   12/22 - Train Accuracy:  0.891, Validation Accuracy:  0.897, Loss:  0.121
    Epoch  58 Batch   13/22 - Train Accuracy:  0.890, Validation Accuracy:  0.895, Loss:  0.122
    Epoch  58 Batch   14/22 - Train Accuracy:  0.903, Validation Accuracy:  0.898, Loss:  0.109
    Epoch  58 Batch   15/22 - Train Accuracy:  0.887, Validation Accuracy:  0.897, Loss:  0.124
    Epoch  58 Batch   16/22 - Train Accuracy:  0.894, Validation Accuracy:  0.897, Loss:  0.121
    Epoch  58 Batch   17/22 - Train Accuracy:  0.894, Validation Accuracy:  0.899, Loss:  0.122
    Epoch  58 Batch   18/22 - Train Accuracy:  0.900, Validation Accuracy:  0.897, Loss:  0.111
    Epoch  58 Batch   19/22 - Train Accuracy:  0.896, Validation Accuracy:  0.897, Loss:  0.113
    Epoch  58 Batch   20/22 - Train Accuracy:  0.893, Validation Accuracy:  0.898, Loss:  0.120
    Epoch  59 Batch    0/22 - Train Accuracy:  0.888, Validation Accuracy:  0.897, Loss:  0.119
    Epoch  59 Batch    1/22 - Train Accuracy:  0.897, Validation Accuracy:  0.897, Loss:  0.115
    Epoch  59 Batch    2/22 - Train Accuracy:  0.893, Validation Accuracy:  0.897, Loss:  0.121
    Epoch  59 Batch    3/22 - Train Accuracy:  0.897, Validation Accuracy:  0.899, Loss:  0.116
    Epoch  59 Batch    4/22 - Train Accuracy:  0.895, Validation Accuracy:  0.899, Loss:  0.119
    Epoch  59 Batch    5/22 - Train Accuracy:  0.894, Validation Accuracy:  0.899, Loss:  0.119
    Epoch  59 Batch    6/22 - Train Accuracy:  0.896, Validation Accuracy:  0.900, Loss:  0.117
    Epoch  59 Batch    7/22 - Train Accuracy:  0.889, Validation Accuracy:  0.900, Loss:  0.124
    Epoch  59 Batch    8/22 - Train Accuracy:  0.896, Validation Accuracy:  0.899, Loss:  0.117
    Epoch  59 Batch    9/22 - Train Accuracy:  0.889, Validation Accuracy:  0.897, Loss:  0.121
    Epoch  59 Batch   10/22 - Train Accuracy:  0.882, Validation Accuracy:  0.895, Loss:  0.124
    Epoch  59 Batch   11/22 - Train Accuracy:  0.892, Validation Accuracy:  0.896, Loss:  0.118
    Epoch  59 Batch   12/22 - Train Accuracy:  0.891, Validation Accuracy:  0.899, Loss:  0.115
    Epoch  59 Batch   13/22 - Train Accuracy:  0.897, Validation Accuracy:  0.901, Loss:  0.118
    Epoch  59 Batch   14/22 - Train Accuracy:  0.907, Validation Accuracy:  0.900, Loss:  0.104
    Epoch  59 Batch   15/22 - Train Accuracy:  0.889, Validation Accuracy:  0.899, Loss:  0.119
    Epoch  59 Batch   16/22 - Train Accuracy:  0.897, Validation Accuracy:  0.901, Loss:  0.116
    Epoch  59 Batch   17/22 - Train Accuracy:  0.895, Validation Accuracy:  0.901, Loss:  0.116
    Epoch  59 Batch   18/22 - Train Accuracy:  0.902, Validation Accuracy:  0.901, Loss:  0.107
    Epoch  59 Batch   19/22 - Train Accuracy:  0.899, Validation Accuracy:  0.900, Loss:  0.108
    Epoch  59 Batch   20/22 - Train Accuracy:  0.892, Validation Accuracy:  0.899, Loss:  0.114
    Epoch  60 Batch    0/22 - Train Accuracy:  0.893, Validation Accuracy:  0.901, Loss:  0.114
    Epoch  60 Batch    1/22 - Train Accuracy:  0.899, Validation Accuracy:  0.902, Loss:  0.110
    Epoch  60 Batch    2/22 - Train Accuracy:  0.897, Validation Accuracy:  0.902, Loss:  0.115
    Epoch  60 Batch    3/22 - Train Accuracy:  0.899, Validation Accuracy:  0.903, Loss:  0.111
    Epoch  60 Batch    4/22 - Train Accuracy:  0.898, Validation Accuracy:  0.903, Loss:  0.113
    Epoch  60 Batch    5/22 - Train Accuracy:  0.895, Validation Accuracy:  0.900, Loss:  0.115
    Epoch  60 Batch    6/22 - Train Accuracy:  0.899, Validation Accuracy:  0.901, Loss:  0.113
    Epoch  60 Batch    7/22 - Train Accuracy:  0.892, Validation Accuracy:  0.902, Loss:  0.118
    Epoch  60 Batch    8/22 - Train Accuracy:  0.897, Validation Accuracy:  0.901, Loss:  0.112
    Epoch  60 Batch    9/22 - Train Accuracy:  0.895, Validation Accuracy:  0.903, Loss:  0.115
    Epoch  60 Batch   10/22 - Train Accuracy:  0.888, Validation Accuracy:  0.903, Loss:  0.118
    Epoch  60 Batch   11/22 - Train Accuracy:  0.898, Validation Accuracy:  0.903, Loss:  0.111
    Epoch  60 Batch   12/22 - Train Accuracy:  0.898, Validation Accuracy:  0.905, Loss:  0.110
    Epoch  60 Batch   13/22 - Train Accuracy:  0.901, Validation Accuracy:  0.905, Loss:  0.113
    Epoch  60 Batch   14/22 - Train Accuracy:  0.909, Validation Accuracy:  0.904, Loss:  0.099
    Epoch  60 Batch   15/22 - Train Accuracy:  0.896, Validation Accuracy:  0.906, Loss:  0.113
    Epoch  60 Batch   16/22 - Train Accuracy:  0.900, Validation Accuracy:  0.905, Loss:  0.110
    Epoch  60 Batch   17/22 - Train Accuracy:  0.901, Validation Accuracy:  0.904, Loss:  0.111
    Epoch  60 Batch   18/22 - Train Accuracy:  0.905, Validation Accuracy:  0.904, Loss:  0.101
    Epoch  60 Batch   19/22 - Train Accuracy:  0.902, Validation Accuracy:  0.904, Loss:  0.103
    Epoch  60 Batch   20/22 - Train Accuracy:  0.897, Validation Accuracy:  0.905, Loss:  0.109
    Epoch  61 Batch    0/22 - Train Accuracy:  0.895, Validation Accuracy:  0.905, Loss:  0.108
    Epoch  61 Batch    1/22 - Train Accuracy:  0.902, Validation Accuracy:  0.905, Loss:  0.105
    Epoch  61 Batch    2/22 - Train Accuracy:  0.900, Validation Accuracy:  0.905, Loss:  0.109
    Epoch  61 Batch    3/22 - Train Accuracy:  0.903, Validation Accuracy:  0.904, Loss:  0.107
    Epoch  61 Batch    4/22 - Train Accuracy:  0.896, Validation Accuracy:  0.900, Loss:  0.110
    Epoch  61 Batch    5/22 - Train Accuracy:  0.898, Validation Accuracy:  0.901, Loss:  0.110
    Epoch  61 Batch    6/22 - Train Accuracy:  0.902, Validation Accuracy:  0.904, Loss:  0.109
    Epoch  61 Batch    7/22 - Train Accuracy:  0.895, Validation Accuracy:  0.904, Loss:  0.114
    Epoch  61 Batch    8/22 - Train Accuracy:  0.901, Validation Accuracy:  0.905, Loss:  0.107
    Epoch  61 Batch    9/22 - Train Accuracy:  0.895, Validation Accuracy:  0.902, Loss:  0.109
    Epoch  61 Batch   10/22 - Train Accuracy:  0.893, Validation Accuracy:  0.905, Loss:  0.112
    Epoch  61 Batch   11/22 - Train Accuracy:  0.900, Validation Accuracy:  0.903, Loss:  0.107
    Epoch  61 Batch   12/22 - Train Accuracy:  0.901, Validation Accuracy:  0.906, Loss:  0.105
    Epoch  61 Batch   13/22 - Train Accuracy:  0.903, Validation Accuracy:  0.907, Loss:  0.108
    Epoch  61 Batch   14/22 - Train Accuracy:  0.914, Validation Accuracy:  0.907, Loss:  0.095
    Epoch  61 Batch   15/22 - Train Accuracy:  0.896, Validation Accuracy:  0.906, Loss:  0.108
    Epoch  61 Batch   16/22 - Train Accuracy:  0.899, Validation Accuracy:  0.904, Loss:  0.107
    Epoch  61 Batch   17/22 - Train Accuracy:  0.901, Validation Accuracy:  0.903, Loss:  0.109
    Epoch  61 Batch   18/22 - Train Accuracy:  0.907, Validation Accuracy:  0.904, Loss:  0.101
    Epoch  61 Batch   19/22 - Train Accuracy:  0.905, Validation Accuracy:  0.908, Loss:  0.101
    Epoch  61 Batch   20/22 - Train Accuracy:  0.902, Validation Accuracy:  0.906, Loss:  0.105
    Epoch  62 Batch    0/22 - Train Accuracy:  0.893, Validation Accuracy:  0.901, Loss:  0.105
    Epoch  62 Batch    1/22 - Train Accuracy:  0.900, Validation Accuracy:  0.902, Loss:  0.105
    Epoch  62 Batch    2/22 - Train Accuracy:  0.902, Validation Accuracy:  0.906, Loss:  0.107
    Epoch  62 Batch    3/22 - Train Accuracy:  0.905, Validation Accuracy:  0.909, Loss:  0.102
    Epoch  62 Batch    4/22 - Train Accuracy:  0.903, Validation Accuracy:  0.909, Loss:  0.103
    Epoch  62 Batch    5/22 - Train Accuracy:  0.900, Validation Accuracy:  0.907, Loss:  0.104
    Epoch  62 Batch    6/22 - Train Accuracy:  0.905, Validation Accuracy:  0.909, Loss:  0.104
    Epoch  62 Batch    7/22 - Train Accuracy:  0.898, Validation Accuracy:  0.908, Loss:  0.109
    Epoch  62 Batch    8/22 - Train Accuracy:  0.900, Validation Accuracy:  0.907, Loss:  0.102
    Epoch  62 Batch    9/22 - Train Accuracy:  0.901, Validation Accuracy:  0.908, Loss:  0.105
    Epoch  62 Batch   10/22 - Train Accuracy:  0.893, Validation Accuracy:  0.907, Loss:  0.108
    Epoch  62 Batch   11/22 - Train Accuracy:  0.906, Validation Accuracy:  0.911, Loss:  0.103
    Epoch  62 Batch   12/22 - Train Accuracy:  0.905, Validation Accuracy:  0.911, Loss:  0.100
    Epoch  62 Batch   13/22 - Train Accuracy:  0.907, Validation Accuracy:  0.910, Loss:  0.103
    Epoch  62 Batch   14/22 - Train Accuracy:  0.916, Validation Accuracy:  0.910, Loss:  0.092
    Epoch  62 Batch   15/22 - Train Accuracy:  0.900, Validation Accuracy:  0.909, Loss:  0.103
    Epoch  62 Batch   16/22 - Train Accuracy:  0.903, Validation Accuracy:  0.909, Loss:  0.101
    Epoch  62 Batch   17/22 - Train Accuracy:  0.907, Validation Accuracy:  0.910, Loss:  0.104
    Epoch  62 Batch   18/22 - Train Accuracy:  0.912, Validation Accuracy:  0.911, Loss:  0.094
    Epoch  62 Batch   19/22 - Train Accuracy:  0.909, Validation Accuracy:  0.911, Loss:  0.095
    Epoch  62 Batch   20/22 - Train Accuracy:  0.905, Validation Accuracy:  0.910, Loss:  0.099
    Epoch  63 Batch    0/22 - Train Accuracy:  0.901, Validation Accuracy:  0.909, Loss:  0.100
    Epoch  63 Batch    1/22 - Train Accuracy:  0.909, Validation Accuracy:  0.910, Loss:  0.097
    Epoch  63 Batch    2/22 - Train Accuracy:  0.908, Validation Accuracy:  0.910, Loss:  0.101
    Epoch  63 Batch    3/22 - Train Accuracy:  0.908, Validation Accuracy:  0.910, Loss:  0.098
    Epoch  63 Batch    4/22 - Train Accuracy:  0.906, Validation Accuracy:  0.912, Loss:  0.100
    Epoch  63 Batch    5/22 - Train Accuracy:  0.908, Validation Accuracy:  0.911, Loss:  0.100
    Epoch  63 Batch    6/22 - Train Accuracy:  0.907, Validation Accuracy:  0.910, Loss:  0.098
    Epoch  63 Batch    7/22 - Train Accuracy:  0.901, Validation Accuracy:  0.911, Loss:  0.103
    Epoch  63 Batch    8/22 - Train Accuracy:  0.904, Validation Accuracy:  0.910, Loss:  0.097
    Epoch  63 Batch    9/22 - Train Accuracy:  0.904, Validation Accuracy:  0.911, Loss:  0.099
    Epoch  63 Batch   10/22 - Train Accuracy:  0.899, Validation Accuracy:  0.909, Loss:  0.103
    Epoch  63 Batch   11/22 - Train Accuracy:  0.906, Validation Accuracy:  0.912, Loss:  0.098
    Epoch  63 Batch   12/22 - Train Accuracy:  0.909, Validation Accuracy:  0.912, Loss:  0.097
    Epoch  63 Batch   13/22 - Train Accuracy:  0.909, Validation Accuracy:  0.911, Loss:  0.098
    Epoch  63 Batch   14/22 - Train Accuracy:  0.918, Validation Accuracy:  0.912, Loss:  0.087
    Epoch  63 Batch   15/22 - Train Accuracy:  0.903, Validation Accuracy:  0.912, Loss:  0.099
    Epoch  63 Batch   16/22 - Train Accuracy:  0.908, Validation Accuracy:  0.912, Loss:  0.096
    Epoch  63 Batch   17/22 - Train Accuracy:  0.908, Validation Accuracy:  0.911, Loss:  0.099
    Epoch  63 Batch   18/22 - Train Accuracy:  0.911, Validation Accuracy:  0.908, Loss:  0.089
    Epoch  63 Batch   19/22 - Train Accuracy:  0.909, Validation Accuracy:  0.909, Loss:  0.093
    Epoch  63 Batch   20/22 - Train Accuracy:  0.904, Validation Accuracy:  0.909, Loss:  0.097
    Model Trained and Saved


### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    # TODO: Implement Function
    return [vocab_to_int[word] if word in vocab_to_int.keys() else vocab_to_int["<UNK>"] for word in sentence.split()]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed


## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
```

    Input
      Word Ids:      [91, 142, 176, 32, 144, 27, 113]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [327, 144, 326, 149, 98, 55, 129, 1]
      French Words: ['il', 'les', 'une', 'camion', 'jaune', 'brillant', '.', '<EOS>']



```python
translate_sentence = 'she is beautiful.' 

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('logits:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence], keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in np.argmax(translate_logits, 1)]))
print('  French Words: {}'.format([target_int_to_vocab[i] for i in np.argmax(translate_logits, 1)]))
```

    Input
      Word Ids:      [182, 52, 2]
      English Words: ['she', 'is', '<UNK>']
    
    Prediction
      Word Ids:      [106, 284, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      French Words: ['est', 'le', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']


## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
