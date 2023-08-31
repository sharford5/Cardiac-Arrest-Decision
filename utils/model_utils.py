from keras.models import Model
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Reshape
from keras.layers import Conv1D, BatchNormalization, Activation, concatenate, multiply, Dropout
from keras.regularizers import l2



def build_embedding_model(num_classes, metadata, conv_layers=3, conv_nodes=64, embedding_dim=20, dropout_rate=0.9):
    inputs = []
    categorical_layers = []

    for i, cat_input_dim in enumerate(metadata):
        ip = Input(shape=(1,), dtype='int32', name='input_%s' % i)
        embed = Embedding(cat_input_dim, embedding_dim)(ip)
        inputs.append(ip)
        categorical_layers.append(embed)

    #Combine Embedding Layers
    embedding = concatenate(categorical_layers, axis=1)
    #Send Embedding through Conv Block
    x = conv_block(embedding, layers=conv_layers, nodes=conv_nodes, dropout_rate= dropout_rate)

    x = GlobalAveragePooling1D()(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)
    model = Model(inputs, out)
    # model.summary()
    return model

def conv_block(x, layers=3, dropout_rate=0.9, nodes=128):
    for _ in range(layers):
        x = Conv1D(nodes, 3, padding='same', kernel_regularizer=l2(0.01), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = squeeze_excite_block(x)
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, layers=3, dropout_rate=0.6, nodes=32):
    for _ in range(layers):
        x = Dense(nodes, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    return x

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input.shape[-1]  # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
