# Identificación y separación de voces e instrumentos
![Facultad Informática de Gipuzkoa](images_in_readme/Facultad_Informatica_Gipuzkoa_bilingue_negativo_alta.jpg)

Este es un trabajo de fin de grado que se ha realizado con la dirección de Basilio Sierra Araujo como director e Izaro Goienetxea Urkizu haciendo el rol de codirección.
En este Trabajo de Fin de Grado, con el objetivo de desarrollar una herramienta de identificación de elementos en una pista musical, se ha llevado a cabo un proceso de investigación y desarrollo que ha incluido la revisión de literatura sobre el estado del arte en separación de fuentes musicales, la implementación y entrenamiento de modelos de inferencia de máscaras utilizando diferentes funciones de pérdida, y la evaluación comparativa de su desempeño sobre las mismas piezas musicales.

Este TFG no busca superar modelos del estado del arte como Spleeter,Open-Unmix o Demucs si no que se limita a dar un acercamiento incial "sencillo" para aprender sobre la construcción de los sistemas que dan solución a este problema y dar un estudio experimental controlado dentro del alcance al estado del arte sobre diferentes funciones de pérdida.

## 1. Requisitos
- Python 3.10
- Conda
- CUDA opcional

## 2. Instalación
conda create -n tfg_py310 python=3.10
conda activate tfg_py310
pip install -r requirements.txt

## 3. Estructura de carpetas
- config/
- models/
- pipeline/
- commons/
- scripts/
- checkpoints/
- resultados_modelos/
- GUI/

## 4. Dataset
Para la tarea de separación de fuentes los datos de entrenamiento elegidos han sido tomados de la base de datos de **MUSDB18**. Se compone de 150 canciones completas de diferentes géneros junto con sus fuentes separadas: **drums**, **bass**, **vocals** y **others**.
El conjunto se separa en dos carpetas, train, de 100 canciones y test, de 50 canciones respectivamente. La documentación indica que los enfoques supervisados deben ser entrenados en el conjunto de entenamiento y probados en los conjuntos de test y de entrenamiento, y, además, que todas las pistas son estereofónicas y codificadas en 44.1kHz.

Por otro lado se ha definido un conjunto representativo de test para el protocolo de evaluación que se compone de 20 pistas extraidas del dataset original junto con sus respectivas fuentes llamado **representative_test**

Ubicaciones:

- Dataset de entrenamiento : datasets/mix_generation/foreground_train
- Dataset de validación : datasets/mix_generation/foreground_valid
- Dataset de test representativo : datasets/representative_test
- Pistas del test representativo : datasets/representative_test/track_00/ .. /track19/
- Estructura de cada pista de test : mixture.wav,vocals.wav,bass.wav,drums.wav,other.wav

MUSDB18 original : [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav)

## 5. Checkpoints
Los checkpoints del entrenamiento local se guardan en **checkpoints/Mis_modelos_v2** mientras que los checkpoints del entrenamiento por GUI se guardarán en **checkpoints/Modelos_Usuario**

## 6. Entrenamiento

python main.py --mode train

## 7. Evaluación

python -m scripts.evaluate_models

## 8. Baseline

python -m scripts.evaluate_mixture_baseline

## 9. Resultados y rankings

python -m scripts.generate_results_artifacts

## 10. Deployment
python main.py --mode deploy

## 11. GUI
streamlit run GUI/gui.py

## 12. Limitaciones conocidas
Modelo simple, mono, fase de mezcla, dataset limitado.
