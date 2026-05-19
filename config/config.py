from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


config = {

    'DEVICE': 'cuda',                   # Dispositivo donde se ejecuta el modelo ('cuda' para GPU, 'cpu' para procesador)                                                   --No se incluye
    'MAX_MIXTURES': int(1e8),           # Máximo número de mezclas que se generarán o procesarán (en get_data, para entrenar con tantas muestras como sea posible)          --No se incluye
    'MAX_EPOCHS': 100,                  # Número máximo de épocas o iteraciones completas para entrenar el modelo
    'EPOCH_LENGTH': 10,                 # Número de muestras (mezclas) que se procesan por cada época de entrenamiento
    'LEARNING_RATE': 1e-3,              # Tasa de aprendizaje del optimizador (qué tan rápido se ajustan los pesos)
    'BATCH_SIZE': 100,                  # Cantidad de mezclas procesadas antes de actualizar los pesos (una mini-batch)
    'COHERENT_PROB': 0.5,               # Probabilidad de aplicar algún tipo de coherencia o regularización (depende de implementación concreta)
    'WEIGHT_DECAY' : int(1e-5),         # Regularización de L2 para evitar overfitting.
    'GRADIENT_CLIP' : 1.0,              # Limitar la magnitud de los gradientes y evitar explosiones en el entrenamiento
    'LEARNING_RATE_SCHEDULE': 10,       # Marca si se debe utilizar learning rate decay o algún scheduler ( 'step' , 'cosine ' none' )
    'EARLY_STOPPING_PATIENCE': 20,      # Número de iteraciones sin mejora tras las cuales detener el entrenamiento

    'STFT_WINDOW_LENGTH': 512,          # Tamaño de la ventana para STFT (número de muestras)
    'STFT_HOP_LENGTH': 128,             # Desplazamiento o salto entre ventanas de la STFT
    'STFT_WINDOW_TYPE': 'sqrt_hann',    # Tipo de ventana utilizada para la STFT (ventana sqrt hann en este caso)
    'STFT_PAD_MODE': None,              # Como se hace el padding para la STFT ('reflect' , 'constant' ... )


    'MODEL_NUM_CHANNELS': 1,            # Número de canales de entrada (mono / stereo)
    'MODEL_HIDDEN_SIZE': 100,           # Tamaño del estado oculto en las capas RNN                                                                                         --No se incluye
    'MODEL_NUM_LAYERS': 1,              # Número de capas en la RNN                                                                                                         --No se incluye
    'MODEL_BIDIRECTIONAL': False,       # Indica si la RNN es bidireccional (procesa secuencias en ambas direcciones)                                                       --No se incluye
    'MODEL_DROPOUT': 0.0,               # Porcentaje de dropout (regularización para evitar overfitting)
    'MODEL_NUM_SOURCES': 2,             # Número de fuentes o stems que se desean separar (por ejemplo, voz + acompañamiento)
    'MODEL_ACTIVATION': 'sigmoid',      # Función de activación usada en la capa de salida (softmax para probabilidades)
    'MODEL_RNN_TYPE': 'lstm',           # Tipo de RNN usada (LSTM, GRU, etc.)                                                                                               --No se incluye
    'MODEL_LOSS_FUNCTION': 'l1',        # Función de pérdida usada para el entrenamiento ('L1' para error absoluto medio)
    'MODEL_INPUT_SIZE': 257,             # Número de características o bins de frecuencia que se alimentan al modelo (debe coincidir con F de la STFT)

    'EVALUATOR_FRAMES': 10,              # Número de frames o segmentos usados en la evaluación del modelo

    'AUGMENTATION_PROB': 0,
    'NOISE_LEVEL': 0,
    'SAMPLE_RATE': 44100,
    'REF_DB': -20,
    'NORMALIZE_AUDIO': False,

    "DEBUG_MODEL_FORWARD": False,

    'SOX_EFFECTS': [
        ('reverb',),
        ('pitch', '300'),  # 300 centésimas de semitono (3 semitonos)
        ('equalizer', '1000', '1.0q', '5')
    ],
    'USE_SOX_EFFECTS': False

    # Relative routing for data and model checkpoints, etc. (not absolute paths)




}

config = {
    "TRAIN_VERSION": "v2",

    "CHECKPOINTS_ROOT": "checkpoints",
    "CHECKPOINTS_GROUP": "Mis_modelos_v2",

    "RESULTS_ROOT": "resultados_modelos",
    "RESULTS_EVAL_GROUP": "evaluate_v2",

    "DEVICE": "cuda",

    "MAX_MIXTURES": int(1e8),
    "MAX_EPOCHS": 100,
    "EPOCH_LENGTH": 100,

    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-5,
    "GRADIENT_CLIP": 1.0,
    "EARLY_STOPPING_PATIENCE": 15,

    "EFFECTIVE_BATCH_SIZE_MAIN": 96,
    "EFFECTIVE_BATCH_SIZE_EXPERIMENTAL": 32,

    "BATCH_SIZE_2STEMS_MAIN": 32,
    "BATCH_SIZE_4STEMS_MAIN": 24,
    "BATCH_SIZE_EXPERIMENTAL": 4,

    "COHERENT_PROB": 0.5,

    "STFT_WINDOW_LENGTH": 512,
    "STFT_HOP_LENGTH": 128,
    "STFT_WINDOW_TYPE": "sqrt_hann",
    "STFT_PAD_MODE": None,

    "MODEL_NUM_CHANNELS": 1,
    "MODEL_HIDDEN_SIZE": 50,
    "MODEL_NUM_LAYERS": 1,
    "MODEL_BIDIRECTIONAL": True,
    "MODEL_DROPOUT": 0.0,
    "MODEL_NUM_SOURCES": 2,
    "MODEL_ACTIVATION": "sigmoid",
    "MODEL_RNN_TYPE": "lstm",
    "MODEL_LOSS_FUNCTION": "l1",
    "MODEL_INPUT_SIZE": 257,

    "EVALUATOR_FRAMES": 10,

    "AUGMENTATION_PROB": 0,
    "NOISE_LEVEL": 0,
    "SAMPLE_RATE": 44100,
    "REF_DB": -20,
    "NORMALIZE_AUDIO": False,

    "USE_SOX_EFFECTS": False,
}

