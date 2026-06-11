# Identificación y separación de voces e instrumentos
![Facultad Informática de Gipuzkoa](images_in_readme/Facultad_Informatica_Gipuzkoa_bilingue_negativo_alta.jpg)
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
Explicar MUSDB18 y representative_test.

## 5. Checkpoints
Explicar dónde se colocan y formato esperado.

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
