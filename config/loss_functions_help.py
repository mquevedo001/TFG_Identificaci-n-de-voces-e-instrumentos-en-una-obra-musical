# help_quotes.py

import streamlit as st

def show_funcion_l1():
    st.markdown("**Función L1**")
    st.latex(r"\mathcal{L}_{L1} = \frac{1}{N} \sum_i | \hat{y}_i - y_i |")
    st.markdown("Minimiza el error absoluto promedio entre predicción y objetivo.")

def show_funcion_l2():
    st.markdown("**Función L2 (MSE)**")
    st.latex(r"\mathcal{L}_{L2} = \frac{1}{N} \sum_i (\hat{y}_i - y_i)^2")
    st.markdown("Minimiza el error cuadrático medio.")

def show_funcion_l1_freq():
    st.markdown("**Función L1 en frecuencia**")
    st.latex(r"\mathcal{L}_{L1}^{freq} = \frac{1}{N} \sum_{t,f} | \hat{Y}_{tf} - Y_{tf} |")
    st.markdown("Aplica L1 sobre espectrogramas reorganizados (frecuencia-tiempo).")

def show_funcion_l2_freq():
    st.markdown("**Función L2 en frecuencia**")
    st.latex(r"\mathcal{L}_{L2}^{freq} = \frac{1}{N} \sum_{t,f} (\hat{Y}_{tf} - Y_{tf})^2")
    st.markdown("Aplica L2 sobre espectrogramas reorganizados.")

def show_funcion_logl1():
    st.markdown("**Función LOG-L1**")
    st.latex(r"""
        \mathcal{L}_{\log L1} = 10 \cdot \mathbb{E} \left[ \log_{10} \left( \sum_{t,f} | |\hat{Y}_{tf}| - |Y_{tf}| | + \epsilon \right) \right]
    """)
    st.markdown("Reduce el impacto de grandes errores usando un logaritmo.")

def show_funcion_logl2():
    st.markdown("**Función LOG-L2**")
    st.latex(r"""
        \mathcal{L}_{\log L2} = 10 \cdot \mathbb{E} \left[ \log_{10} \left( \sum_{t,f} (|\hat{Y}_{tf}| - |Y_{tf}|)^2 + \epsilon \right) \right]
    """)
    st.markdown("Error cuadrático en escala logarítmica.")

def show_funcion_log_mag():
    st.markdown("**Pérdida de magnitud logarítmica**")
    st.latex(r"""
        \mathcal{L} = \frac{1}{N} \sum_i \left| \log_{10}(|\hat{y}_i| + \epsilon) - \log_{10}(|y_i| + \epsilon) \right|
    """)
    st.markdown("Compara las magnitudes en dB.")

def show_funcion_log_compressed_l2():
    st.markdown("**Función log comprimida**")
    st.latex(r"""
        \mathcal{L} = \frac{1}{N} \sum_i \left( \log(|\hat{y}_i - y_i| + \epsilon) \right)^2
    """)
    st.markdown("Comprime errores grandes con log y aplica L2.")

def show_funcion_lpsa():
    st.markdown("**LPSA**")
    st.latex(r"""
        \mathcal{L} = \text{MSE} \left( \log(|\hat{Y}|^2 + \epsilon), \log(|Y|^2 + \epsilon) \right)
    """)
    st.markdown("Comparación logarítmica de espectro de potencia.")

def show_funcion_lpsa_phase():
    st.markdown("**LPSA con fase**")
    st.latex(r"""
        Y^{PSA} = |Y| \cdot \cos(\angle X - \angle Y), \quad
        \mathcal{L} = \frac{1}{N} \sum_i \left( |\hat{Y}_i| - Y^{PSA}_i \right)^2
    """)
    st.markdown("Ajusta la magnitud según la fase de la mezcla.")

def show_funcion_lmrs():
    st.markdown("**LMRS (Log Mask Ratio Score)**")
    st.latex(r"""
        r = \log\left(\frac{y + \epsilon}{x + \epsilon} \right), \quad
        \hat{r} = \log\left(\frac{\hat{y} + \epsilon}{x + \epsilon} \right), \quad
        \mathcal{L} = \text{MSE}(r, \hat{r})
    """)
    st.markdown("Evalúa qué tan bien se estima la relación fuente/mezcla en log.")

def show_funcion_mask_l1():
    st.markdown("**Máscara ideal (L1 o L2)**")
    st.latex(r"""
        \mathcal{L} = \frac{1}{N} \sum_i \left| \hat{M}_i - M^\text{ideal}_i \right|, \quad
        M^\text{ideal} = \frac{y_i}{\sum_j y_j + \epsilon}
    """)
    st.markdown("Compara la máscara predicha con la máscara ideal.")

def show_funcion_l_mrs():
    st.markdown("**L-MRS (multi-resolution STFT loss)**")
    st.latex(r"""
        \mathcal{L} = \frac{1}{K} \sum_k \left( \text{SC}_k(\hat{Y}, Y) + \text{log\_mag}_k(\hat{Y}, Y) \right)
    """)
    st.markdown("Combinación de convergencia espectral y log-magnitud en múltiples resoluciones.")

def show_funcion_deep_feature():
    st.markdown("**Deep Feature Loss**")
    st.latex(r"""
        \mathcal{L} = \frac{1}{L} \sum_{l} \text{MSE}( \phi_l(\hat{Y}), \phi_l(Y) )
    """)
    st.markdown("Compara activaciones de capas convolucionales intermedias para capturar estructura perceptual.")

def show_funcion_deep_feature_emd():
    st.markdown("**Deep Feature Loss EMD**")
    st.latex(r"""Deep Feature Loss con Earth Mover Distance (EMD):
    \[
    \mathcal{L} = \frac{1}{L} \sum_{l} \text{EMD}\big( \phi_l(\hat{Y}), \phi_l(Y) \big)
    \])""")
    st.markdown("Compara características intermedias de capas convolucionales usando la distancia de transporte óptimo (EMD).")
# Mapeo para usar fácilmente desde keys
loss_key_map = {
    'L1': show_funcion_l1,
    'L2': show_funcion_l2,
    'L1_freq': show_funcion_l1_freq,
    'L2_freq': show_funcion_l2_freq,
    'LOGL1_freq': show_funcion_logl1,
    'LOGL2_freq': show_funcion_logl2,
    'LOG_mag': show_funcion_log_mag,
    'LOG_compressed_l2': show_funcion_log_compressed_l2,
    'MASK_L1': show_funcion_mask_l1,
    'LPSA': show_funcion_lpsa,
    'LPSA_phase': show_funcion_lpsa_phase,
    'LMRS': show_funcion_lmrs,
    'L_MRS': show_funcion_l_mrs,
    'Deep-feature': show_funcion_deep_feature,
    'MSE': show_funcion_l2,  # alias
    'SDR': show_funcion_l2,  # alias
    'Deep-feature-EMD': show_funcion_deep_feature_emd
}
