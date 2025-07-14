from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from torch import nn
import nussl

#TODO
#Mirar funcionamiento (esquemático y algebráico )  del modelo en documentación

#TODO
#Mirar cómo se podría jugar con las funciones ( input normalization , amplitude to db ...)
#Se podría mirar el paper de musdb con los métodos que se utilizan aquí.
class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources,
                 activation):
        super().__init__()

        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features = num_features * num_audio_channels,       #Número de características que se mapean por cada frame
            hidden_size = hidden_size ,                             #Tamaño oculto del stack recurrente para cada capa
            num_layers=num_layers,                                  #Número de capas en el stack
            bidirectional = bool(bidirectional),                    #True -> BiLSTM. False ->BiGRU
            dropout = dropout,                                      #Para evitar overfitting, se ponen aleatoriamente una fracción de nodos a 0. Indica el número de nodos.

        )
        self.embedding = Embedding(

            num_features = num_features,                            #Número de características que se mapean por cada frame
            hidden_size = hidden_size * (int(bidirectional) + 1),                              #Tamaño del output de RecurrentStack hidden size
            embedding_size = num_sources,                           #Dimensionalidad del embedding
            activation = activation,                                #Función de activación que se va a aplicar
            num_audio_channels = num_audio_channels,                #Número de canales que tiene el input
            bias = True,                                            #Determina si se aplica un campo Bias en la capa linear
            reshape = True                                          #Determina si se hace reshape al output para que tenga formato de representación T-F (nb,nt,nf,nc, ... )
        )

    def forward(self, data):
        mix_magnitude = data  # save for masking

        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask

        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output

    # Added function
    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size,
              num_layers, bidirectional, dropout, num_sources,
              activation='sigmoid'):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(cls)

        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'MaskInference',
                'args': {
                    'num_features':         num_features,
                    'num_audio_channels':   num_audio_channels,
                    'hidden_size':          hidden_size,
                    'num_layers':           num_layers,
                    'bidirectional':        bidirectional,
                    'dropout':              dropout,
                    'num_sources':          num_sources,
                    'activation':           activation
                }
            }
        }

        # Step 2b: Define the connections between input and output.
        # Here, the mix_magnitude key is the only input to the model.
        connections = [
            ['model', ['mix_magnitude']]
        ]

        # Step 2c. The model outputs a dictionary, which SeparationModel will
        # change the keys to model:mask, model:estimates. The lines below
        # alias model:mask to just mask, and model:estimates to estimates.
        # This will be important later when we actually deploy our model.
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])

        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['estimates', 'mask', ]
        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)


#TODO
'''
o effectively adjust neural network parameters, focus on two key areas: hyperparameter tuning and weight initialization. Hyperparameters, like learning rate and batch size, are set before training, while weights are adjusted during the training process. Techniques like grid search, random search, and Bayesian optimization can help find optimal hyperparameter combinations. 
Proper weight initialization, along with learning algorithms like backpropagation, ensures efficient weight updates


Hyperparameter Tuning:

    Learning Rate:
    Controls the step size during weight updates. A higher learning rate can lead to faster convergence but may overshoot the optimal solution, while a smaller learning rate can be more stable but slower. 

Batch Size:
Determines how many training examples are processed before updating the model's weights. Larger batches offer more stable updates but can be computationally expensive. 
Number of Layers and Neurons:
More layers and neurons increase model capacity but can also lead to overfitting. Start with a reasonable architecture and experiment with adding or removing layers. 
Regularization:
Techniques like L1 or L2 regularization can prevent overfitting by penalizing large weights. Dropout regularization randomly disables neurons during training, further reducing overfitting. 
Activation Functions:
Choose appropriate activation functions (e.g., ReLU, sigmoid, tanh) for each layer to introduce non-linearity and improve learning. 

Weight Initialization:

    Random Initialization:
    Start with small random values for weights to break symmetry and allow neurons to learn different features. Common methods include Xavier/He initialization.
    Bias Initialization:
    Biases are often initialized to zero, but other strategies can be explored. 

Training and Optimization:

    Backpropagation:
    A core algorithm for updating weights based on the error signal from the output layer. 

Stochastic Gradient Descent (SGD):
A commons optimization algorithm that iteratively updates weights based on mini-batches of training data. 
Learning Rate Scheduling:
Adjust the learning rate during training, potentially decreasing it over time to fine-tune the model. 

Experimentation and Validation:

    Validation Set:
    Use a separate validation set to evaluate model performance during training and tune hyperparameters.
    Cross-Validation:
    Employ cross-validation techniques to get a more robust estimate of model performance.
    Early Stopping:
    Stop training when the validation loss starts to increase, preventing overfitting. 

By carefully adjusting these parameters and monitoring performance, you can optimize your neural network for better accuracy and generalization. 

'''