config = {
    'DEVICE': 'cuda',                   # Dispositivo donde se ejecuta el modelo ('cuda' para GPU, 'cpu' para procesador)
    'MAX_MIXTURES': int(1e8),           # Máximo número de mezclas que se generarán o procesarán (en get_data, para entrenar con tantas muestras como sea posible)
    'MAX_EPOCHS': 25,                   # Número máximo de épocas o iteraciones completas para entrenar el modelo
    'EPOCH_LENGTH': 100,                # Número de muestras (mezclas) que se procesan por cada época de entrenamiento
    'LEARNING_RATE': 1e-3,              # Tasa de aprendizaje del optimizador (qué tan rápido se ajustan los pesos)
    'BATCH_SIZE': 10,                   # Cantidad de mezclas procesadas antes de actualizar los pesos (una mini-batch)
    'COHERENT_PROB': 0.5,               # Probabilidad de aplicar algún tipo de coherencia o regularización (depende de implementación concreta)
    'WEIGHT_DECAY' : int(1e-5),         # Regularización de L2 para evitar overfitting.
    'GRADIENT_CLIP' : 1.0,              # Limitar la magnitud de los gradientes y evitar explosiones en el entrenamiento
    'LEARNING_RATE_SCHEDULE': None,     # Marca si se debe utilizar learning rate decay o algun scheduler ( 'step' , 'cosine ' none' )
    'EARLY_STOPPING_PATIENCE': 20,      # Número de iteraciones sin mejora tras las cuales detener el entrenamiento

    'STFT_WINDOW_LENGTH': 512,          # Tamaño de la ventana para STFT (número de muestras)
    'STFT_HOP_LENGTH': 128,             # Desplazamiento o salto entre ventanas de la STFT
    'STFT_WINDOW_TYPE': 'sqrt_hann',    # Tipo de ventana utilizada para la STFT (ventana sqrt hann en este caso)
    'STFT_PAD_MODE': None,              # Como se hace el padding para la STFT ('reflect' , 'constant' ... )


    'MODEL_NUM_CHANNELS': 1,            # Número de canales de entrada (1 para mono audio)
    'MODEL_HIDDEN_SIZE': 50,            # Tamaño del estado oculto en las capas RNN
    'MODEL_NUM_LAYERS': 1,              # Número de capas en la RNN
    'MODEL_BIDIRECTIONAL': True,        # Indica si la RNN es bidireccional (procesa secuencias en ambas direcciones)
    'MODEL_DROPOUT': 0.0,               # Porcentaje de dropout (regularización para evitar overfitting)
    'MODEL_NUM_SOURCES': 2,             # Número de fuentes o stems que se desean separar (por ejemplo, voz + acompañamiento)
    'MODEL_ACTIVATION': 'softmax',      # Función de activación usada en la capa de salida (softmax para probabilidades)
    'MODEL_RNN_TYPE': 'lstm',           # Tipo de RNN usada (LSTM, GRU, etc.)
    'MODEL_LOSS_FUNCTION': 'L1',        # Función de pérdida usada para el entrenamiento ('L1' para error absoluto medio)

    'EVALUATOR_FRAMES': 5,              # Número de frames o segmentos usados en la evaluación del modelo

    'AUGMENTATION_PROB': 0,
    'NOISE_LEVEL': 0,
    'SAMPLE_RATE': 0,
    'NORMALIZE_AUDIO': False
}



