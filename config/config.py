from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

config = {
    "TRAIN_VERSION": "v2",                                      # Versión del entrenamiento

    "CHECKPOINTS_ROOT": str(PROJECT_ROOT / "checkpoints"),      # Nombre de la carpeta de checkpoints                              
    "CHECKPOINTS_GROUP": "Mis_modelos_v2",                      # Nombre de la carpeta de checkpoints de entrenamiento

    "RESULTS_ROOT": str(PROJECT_ROOT / "resultados_modelos"),   # Nombre de la carpeta de resultados
    "RESULTS_EVAL_GROUP": "evaluate_v2",                        # Nombre del grupo de evaluación

    "DEVICE": "cuda",                                           # Dispositivo donde se ejecuta el modelo ('cuda' para GPU, 'cpu' para procesador).

    "MAX_MIXTURES": int(1e8),                                   # Máximo número de mezclas que se generarán o procesarán (en get_data, para entrenar con tantas muestras como sea posible).
    "MAX_EPOCHS": 100,                                          # Número máximo de épocas o iteraciones completas para entrenar el modelo.
    "EPOCH_LENGTH": 100,                                        # Número de muestras (mezclas) que se procesan por cada época de entrenamiento.

    "LEARNING_RATE": 1e-3,                                      # Tasa de aprendizaje del optimizador (qué tan rápido se ajustan los pesos).
    "WEIGHT_DECAY": 1e-5,                                       # Regularización de L2 para evitar overfitting.
    "GRADIENT_CLIP": 1.0,                                       # Limitar la magnitud de los gradientes y evitar explosiones en el entrenamiento.  
    "EARLY_STOPPING_PATIENCE": 10,                              # Número de iteraciones sin mejora tras las cuales detener el entrenamiento.

    "BATCH SIZE":100,                                           # Cantidad de mezclas procesadas antes de actualizar los pesos (una mini-batch).
    "EFFECTIVE_BATCH_SIZE_MAIN": 96,                            # Cantidad de mezclas procesadas antes de actualizar los pesos en el conjunto de entrenamiento principal (una mini-batch).
    "EFFECTIVE_BATCH_SIZE_EXPERIMENTAL": 32,                    # Cantidad de mezclas procesadas antes de actualizar los pesos en el conjunto de entrenamiento experimental.

    "BATCH_SIZE_2STEMS_MAIN": 32,
    "BATCH_SIZE_4STEMS_MAIN": 24,
    "BATCH_SIZE_EXPERIMENTAL": 4,

    "COHERENT_PROB": 0.5,                                       # Probabilidad de aplicar algún tipo de coherencia o regularización (depende de implementación concreta)

    "STFT_WINDOW_LENGTH": 512,                                  # Tamaño de la ventana para STFT (número de muestras)
    "STFT_HOP_LENGTH": 128,                                     # Desplazamiento o salto entre ventanas de la STFT
    "STFT_WINDOW_TYPE": "sqrt_hann",                            # Tipo de ventana utilizada para la STFT (ventana sqrt hann en este caso)
    "STFT_PAD_MODE": None,                                      # Como se hace el padding para la STFT ('reflect' , 'constant' ... )

    "MODEL_NUM_CHANNELS": 1,                                    # Número de canales del modelo ( 1 = mono | 2 = estéreo).
    "MODEL_HIDDEN_SIZE": 50,                                    # Tamaño del estado oculto en las capas RNN                                                                       
    "MODEL_NUM_LAYERS": 1,                                      # Número de capas en la RNN                                                                                       
    "MODEL_BIDIRECTIONAL": True,                                # Indica si la RNN es bidireccional (procesa secuencias en ambas direcciones)                                     
    "MODEL_DROPOUT": 0.0,                                       # Porcentaje de dropout (regularización para evitar overfitting)
    "MODEL_NUM_SOURCES": 2,                                     # Número de fuentes o stems que se desean separar (por ejemplo, voz + acompañamiento)
    "MODEL_ACTIVATION": "sigmoid",                              # Función de activación usada en la capa de salida (softmax para probabilidades)
    "MODEL_RNN_TYPE": "lstm",                                   # Tipo de RNN usada (LSTM, GRU, etc.)                                                                             
    "MODEL_LOSS_FUNCTION": "l1",                                # Función de pérdida usada para el entrenamiento ('L1' para error absoluto medio)
    "MODEL_INPUT_SIZE": 257,                                    # Número de características o bins de frecuencia que se alimentan al modelo (debe coincidir con F de la STFT)

    "EVALUATOR_FRAMES": 10,                                     # Número de frames o segmentos usados en la evaluación del modelo

    "AUGMENTATION_PROB": 0,                                     # Probabilidad de añadir data augmentation en una pista
    "NOISE_LEVEL": 0,                                           # Nivel de ruido que añade data augmentation
    "SAMPLE_RATE": 44100,                                       # Frecuencia de sampleo
    "REF_DB": -20,                                              # DBs de referencia
    "NORMALIZE_AUDIO": False,                                   # Normalizar audio en data augmentation

    "USE_SOX_EFFECTS": False,                                   # Aplicar data augmentation o no

}

