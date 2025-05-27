config = {
    'DEVICE': 'cuda',
    'MAX_MIXTURES': int(1e8),
    'MAX_EPOCHS':  25,
    'EPOCH_LENGTH': 10,

    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 10,
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

    'EVALUATOR_FRAMES': 5
}


