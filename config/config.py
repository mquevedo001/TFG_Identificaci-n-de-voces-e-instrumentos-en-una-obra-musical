config = {
    'DEVICE': 'cuda',
    'MAX_MIXTURES': int(1e8),                   #Maximo de mezclas generadas (get_data)
    'MAX_EPOCHS':  25,                          #Maximo de iteraciones (train)
    'EPOCH_LENGTH': 100,                         #Mezclas por iteración (train)

    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 10,                           #Número de mezclas procesadas antes de actualizar pesos
    'COHERENT_PROB': 0.5,

    'STFT_WINDOW_LENGTH': 512,
    'STFT_HOP_LENGTH': 128,
    'STFT_WINDOW_TYPE': 'sqrt_hann',

    'MODEL_NUM_CHANNELS': 1,
    'MODEL_HIDDEN_SIZE': 50,
    'MODEL_NUM_LAYERS': 1,
    'MODEL_BIDIRECTIONAL': True,
    'MODEL_DROPOUT': 0.0,
    'MODEL_NUM_SOURCES': 2,
    'MODEL_ACTIVATION': 'softmax',
    'MODEL_RNN_TYPE': 'lstm',

    'EVALUATOR_FRAMES': 5
}


