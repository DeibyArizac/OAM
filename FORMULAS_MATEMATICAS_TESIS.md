# FÓRMULAS MATEMÁTICAS DE LA TESIS OAM
## Codificación y Decodificación a partir del Momento Angular Orbital de la Luz

**Autor:** Deiby Fernando Ariza Cadena  
**Código:** 2195590  
**Director:** Dr. Omar Javier Tíjaro Rojas  
**Universidad Industrial de Santander - Escuela E3T**

---

## ÍNDICE DE FÓRMULAS

1. [Fundamentos de Óptica](#fundamentos-de-óptica)
2. [Momento Angular Orbital (OAM)](#momento-angular-orbital-oam)
3. [Haces Laguerre-Gauss](#haces-laguerre-gauss)
4. [Turbulencia Atmosférica](#turbulencia-atmosférica)
5. [Ruido AWGN](#ruido-awgn)
6. [Decodificación Modal](#decodificación-modal)
7. [Implementación en Python](#implementación-en-python)

---

## FUNDAMENTOS DE ÓPTICA

### 1. Ecuación de Helmholtz

**Label:** `eq:helmholtz`

```latex
\nabla^2 E + k^2 E = 0
```

**Descripción:**  
Ecuación de onda para campo eléctrico monocromático E(r,t) = E(r)e^(-iωt)

**Variables:**
- E: Campo eléctrico complejo
- k = 2π/λ = ω/c: Número de onda [m⁻¹]
- λ: Longitud de onda [m]
- ω: Frecuencia angular [rad/s]
- c: Velocidad de la luz [m/s]

---

### 2. Ecuación Paraxial

**Label:** `eq:paraxial`

```latex
2ik ∂u/∂z + ∇⊥² u = 0
```

**Descripción:**  
Ecuación de onda paraxial bajo aproximación de propagación predominante en z

**Variables:**
- u(x,y,z): Envolvente compleja del campo
- ∇⊥² = ∂²/∂x² + ∂²/∂y²: Laplaciano transversal
- z: Dirección de propagación [m]

**Condiciones:**
- Válida cuando la propagación es casi paralela al eje z
- Ángulos de difracción pequeños (< 10°)

---

### 3. Haz Gaussiano Fundamental

**Label:** `eq:gaussian_beam`

```latex
I(r,z) = I₀ (w₀/w(z))² exp(-2r²/w²(z))
```

**Descripción:**  
Distribución de intensidad de un haz Gaussiano (ℓ=0, p=0)

**Variables:**
- I(r,z): Intensidad en función de r y z [W/m²]
- I₀: Intensidad en el eje (r=0, z=0) [W/m²]
- w(z) = w₀√(1 + (z/zᴿ)²): Radio del haz [m]
- w₀: Cintura del haz (beam waist) [m]
- zᴿ = πw₀²/λ: Longitud de Rayleigh [m]
- r: Distancia radial desde el eje [m]

**Parámetros típicos (tesis):**
- λ = 630 nm (He-Ne rojo) o 1550 nm (telecom)
- w₀ = 6 mm
- zᴿ ≈ 180 m (para λ=630nm, w₀=6mm)

---

## MOMENTO ANGULAR ORBITAL (OAM)

### 4. Cuantización del OAM

**Label:** `eq:oam_quantum`

```latex
Lz = ℓℏ
```

**Descripción:**  
Momento angular orbital por fotón cuantizado

**Variables:**
- Lz: Componente z del momento angular [J·s]
- ℓ: Carga topológica (número entero: 0, ±1, ±2, ±3, ...)
- ℏ = h/2π ≈ 1.055×10⁻³⁴ J·s: Constante de Planck reducida

**Valores usados en la tesis:**
- ℓ ∈ {-4, -3, -2, -1, +1, +2, +3, +4} (8 modos OAM)
- Sistema de 4 bits por símbolo

---

### 5. Ortogonalidad de Modos OAM

**Label:** `eq:oam_orthogonality`

```latex
⟨Eℓ | Eℓ'⟩ = ∫₀²ᵖ e^(iℓφ) e^(-iℓ'φ) dφ = 2π δℓ,ℓ'
```

**Descripción:**  
Producto interno de modos con diferentes cargas topológicas

**Variables:**
- Eℓ: Campo con carga topológica ℓ
- φ: Ángulo azimutal [rad]
- δℓ,ℓ': Delta de Kronecker (1 si ℓ=ℓ', 0 si no)

**Importancia:**
- Base matemática para MDM (Mode Division Multiplexing)
- Permite multiplexar canales independientes sin interferencia
- Habilitador de sistemas de alta capacidad

---

## HACES LAGUERRE-GAUSS

### 6. Campo LG Completo

**Label:** `eq:lg_beam`

```latex
Eℓₚ(r,φ,z) = [Cℓₚ/w(z)] (r√2/w(z))^|ℓ| Lₚ^|ℓ|(2r²/w²(z))
             × exp(-r²/w²(z)) exp(-ikr²/2R(z))
             × exp(iℓφ) exp(i(2p+|ℓ|+1)ψ(z))
```

**Descripción:**  
Solución exacta de ecuación paraxial con OAM bien definido

**Variables:**
- Cℓₚ: Constante de normalización
- ℓ: Carga topológica (OAM)
- p: Índice radial (número de anillos radiales)
- Lₚ^|ℓ|(x): Polinomio de Laguerre generalizado
- w(z): Radio del haz en z
- R(z) = z(1 + (zᴿ/z)²): Radio de curvatura del frente de onda [m]
- ψ(z) = arctan(z/zᴿ): Fase de Gouy [rad]

**Configuración de la tesis:**
- p = 0 (modos radialmente fundamentales)
- ℓ ∈ {±1, ±2, ±3, ±4}

---

### 7. Intensidad LG (p=0)

**Label:** `eq:lg_intensity`

```latex
Iℓ₀(r,z) = I₀ (w₀/w(z))² (r√2/w(z))^(2|ℓ|) exp(-2r²/w²(z))
```

**Descripción:**  
Perfil de intensidad para modos Laguerre-Gauss radialmente fundamentales

**Características:**
- I(r=0) = 0 para ℓ ≠ 0 (vórtice óptico en el eje)
- Estructura anular con máximo en r = rₘₐₓ
- Simetría de rotación (independiente de φ)

---

### 8. Radio de Máxima Intensidad

**Label:** `eq:lg_radius_max`

```latex
rₘₐₓ = w(z) √(|ℓ|/2)
```

**Descripción:**  
Radio donde la intensidad alcanza su valor máximo

**Valores para la tesis (w₀=6mm, z=0):**
- ℓ = ±1: rₘₐₓ ≈ 4.24 mm
- ℓ = ±2: rₘₐₓ ≈ 6.00 mm
- ℓ = ±3: rₘₐₓ ≈ 7.35 mm
- ℓ = ±4: rₘₐₓ ≈ 8.49 mm

**Implicaciones:**
- Modos de mayor |ℓ| → anillos más grandes
- Mayor susceptibilidad a turbulencia
- Requieren aperturas receptoras más grandes

---

## TURBULENCIA ATMOSFÉRICA

### 9. Función de Estructura de Kolmogorov

**Label:** `eq:kolmogorov_structure`

```latex
Dₙ(r) = ⟨[n(r₁) - n(r₂)]²⟩ = Cₙ² |r₁ - r₂|^(2/3)
```

**Descripción:**  
Función de estructura del índice de refracción según Kolmogorov

**Variables:**
- Dₙ(r): Función de estructura
- n(r): Índice de refracción en posición r
- Cₙ²: Coeficiente de estructura [m⁻²/³]
- ⟨·⟩: Promedio estadístico (ensemble average)

**Valores típicos de Cₙ²:**
- 10⁻¹⁷ m⁻²/³: Laboratorio (sin turbulencia)
- 10⁻¹⁵ m⁻²/³: Exterior claro (turbulencia débil)
- 10⁻¹⁴ m⁻²/³: Turbulencia moderada
- 10⁻¹³ m⁻²/³: Turbulencia fuerte

---

### 10. Espectro de Potencia de Kolmogorov

**Label:** `eq:kolmogorov_spectrum`

```latex
Φₙ(κ) = 0.033 Cₙ² κ^(-11/3)
```

**Descripción:**  
Espectro de potencia del índice de refracción en el espacio de frecuencias espaciales

**Variables:**
- Φₙ(κ): Densidad espectral de potencia
- κ: Número de onda espacial [rad/m]
- Exponente -11/3: Ley de Kolmogorov en el rango inercial

**Rango de validez:**
- l₀ << 1/κ << L₀
- l₀ ≈ 2 mm (escala interna)
- L₀ ≈ 20 m (escala externa)

---

### 11. Parámetro de Fried (r₀)

**Label:** `eq:fried_parameter`

```latex
r₀ = (0.423 k² ∫₀ᴸ Cₙ²(z) dz)^(-3/5)
```

**Descripción:**  
Longitud de coherencia del frente de onda (diámetro efectivo sin distorsión)

**Variables:**
- r₀: Parámetro de Fried [m]
- k = 2π/λ: Número de onda [rad/m]
- L: Distancia de propagación [m]
- Cₙ²(z): Perfil de turbulencia a lo largo del camino

**Para turbulencia homogénea:**
```
r₀ ≈ (0.423 k² Cₙ² L)^(-3/5)
```

**Valores para λ=630nm, L=50m:**
- Cₙ² = 10⁻¹⁵: r₀ ≈ 0.64 m
- Cₙ² = 10⁻¹⁴: r₀ ≈ 0.29 m
- Cₙ² = 10⁻¹³: r₀ ≈ 0.13 m

**Criterio:**
- D >> r₀: Frente de onda severamente distorsionado
- D ≈ r₀: Distorsión moderada
- D << r₀: Propagación casi libre de turbulencia

---

### 12. Generación de Pantallas de Fase

**Label:** `eq:phase_screen_fft`

```latex
φ(r) = Re{F⁻¹[√Φφ(κ) · Z(κ)]}
```

**Descripción:**  
Síntesis de pantallas de fase turbulenta por método FFT

**Variables:**
- φ(r): Pantalla de fase [rad]
- F⁻¹: Transformada inversa de Fourier
- Φφ(κ) = 0.023 r₀^(-5/3) κ^(-11/3): Espectro de fase
- Z(κ): Ruido Gaussiano complejo (E[Z]=0, E[|Z|²]=1)

**Implementación en Python:**
```python
kappa = np.fft.fftfreq(N, d=dx) * 2*np.pi
Phi_phi = 0.023 * r0**(-5/3) * kappa**(-11/3)
Z = np.random.randn(N,N) + 1j*np.random.randn(N,N)
phi = np.real(np.fft.ifft2(np.sqrt(Phi_phi) * Z))
```

---

### 13. Propagación de Fresnel con Pantalla

**Label:** `eq:fresnel_prop`

```latex
E(r, z_s + Δz) = F⁻¹{F[E(r, z_s) · e^(iφ(r))] · H(κ, Δz)}
```

**Descripción:**  
Propagación del campo a través de pantalla de fase + distancia libre

**Variables:**
- E(r, z): Campo complejo en posición z
- φ(r): Pantalla de fase turbulenta
- H(κ, Δz) = exp(iκ²Δz/2k): Función de transferencia de Fresnel
- Δz: Distancia de propagación libre entre pantallas

**Algoritmo:**
1. Multiplicar campo por pantalla: E' = E · exp(iφ)
2. FFT del resultado: F[E']
3. Multiplicar por H(κ, Δz)
4. IFFT para obtener campo propagado

---

## RUIDO AWGN

### 14. Canal con Ruido AWGN

**Label:** `eq:awgn_channel`

```latex
E_rx(r) = E_signal(r) + n(r)
```

**Descripción:**  
Campo recibido como suma de señal + ruido Gaussiano aditivo

**Variables:**
- E_rx: Campo recibido
- E_signal: Campo de señal (después de turbulencia)
- n(r): Ruido complejo Gaussiano, n ~ CN(0, σₙ²)

**Modelo de ruido:**
- Parte real e imaginaria independientes
- n = nᴿₑ + i·nᵢₘ
- nᴿₑ, nᵢₘ ~ N(0, σₙ²/2)

---

### 15. Relación Señal-Ruido (SNR)

**Label:** `eq:snr_db`

```latex
SNR = 10 log₁₀(P_signal / σₙ²)  [dB]
```

**Descripción:**  
Relación entre potencia de señal y potencia de ruido

**Variables:**
- P_signal = ∫|E_signal(r)|² dr: Potencia de señal [W]
- σₙ²: Varianza del ruido (potencia de ruido) [W]

**Valores usados en la tesis:**
- SNR objetivo: 30 dB
- Rango de prueba: 20-40 dB

---

### 16. Cálculo de Varianza de Ruido

**Label:** `eq:noise_variance`

```latex
σₙ² = P_signal / 10^(SNR_target/10)
```

**Descripción:**  
Varianza de ruido requerida para alcanzar SNR objetivo

**Ejemplo de cálculo:**
- P_signal = 0.01 W (10 mW)
- SNR_target = 30 dB
- σₙ² = 0.01 / 10^(30/10) = 0.01 / 1000 = 10⁻⁵ W
- σₙ = √(10⁻⁵) ≈ 3.16×10⁻³ W^(1/2)

---

## DECODIFICACIÓN MODAL

### 17. Proyección Modal

**Label:** `eq:modal_projection`

```latex
cℓ = ⟨LGℓ | E_rx⟩ = ∫ E_rx(r) · LGℓ*(r) dr
```

**Descripción:**  
Producto interno del campo recibido con modo de referencia

**Variables:**
- cℓ: Coeficiente de proyección (número complejo)
- LGℓ(r): Modo Laguerre-Gauss de referencia con carga ℓ
- LGℓ*: Conjugado complejo del modo
- E_rx: Campo recibido

**Implementación numérica:**
```python
c_ell = np.sum(E_rx * np.conj(LG_ell)) * dx * dy
```

---

### 18. Coeficiente de Correlación Normalizado (NCC)

**Label:** `eq:ncc`

```latex
NCC_ℓ = |⟨LGℓ | E_rx⟩| / (√⟨E_rx|E_rx⟩ · √⟨LGℓ|LGℓ⟩)
```

**Descripción:**  
Correlación normalizada entre campo recibido y modo de referencia

**Variables:**
- NCC_ℓ ∈ [0, 1]: Coeficiente de correlación normalizado
- Numerador: Magnitud del producto interno
- Denominador: Producto de las normas (normalización)

**Interpretación:**
- NCC ≈ 1: Campo recibido es casi idéntico al modo de referencia
- NCC ≈ 0.5-0.8: Correlación moderada (con turbulencia/ruido)
- NCC < 0.3: Correlación baja (modo incorrecto)

**Decisión de símbolo:**
```
ℓ̂ = argmax_ℓ NCC_ℓ
```

**Para modulación por signo:**
1. Determinar magnitud |ℓ| por bits de magnitud
2. Evaluar NCC para +|ℓ| y -|ℓ|
3. Seleccionar signo con mayor NCC

---

## IMPLEMENTACIÓN EN PYTHON

### 19. Generación de Haz LG (código)

**Label:** `eq_19` (Metodología)

```latex
LGℓ₀(r,φ) = (Cℓ/w₀) (r√2/w₀)^|ℓ| exp(-r²/w₀²) exp(iℓφ)
```

**Implementación Python:**
```python
from scipy.special import genlaguerre

def generate_lg_beam(grid_size, wavelength, w0, ell, p=0):
    """
    Genera haz Laguerre-Gauss
    
    Parameters:
    - grid_size: Tamaño de la grilla (NxN)
    - wavelength: Longitud de onda [m]
    - w0: Cintura del haz [m]
    - ell: Carga topológica
    - p: Índice radial (default: 0)
    """
    # Grilla de coordenadas
    x = np.linspace(-3*w0, 3*w0, grid_size)
    y = np.linspace(-3*w0, 3*w0, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Coordenadas cilíndricas
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    
    # Polinomio de Laguerre
    L_p = genlaguerre(p, abs(ell))
    laguerre_term = L_p(2*r**2 / w0**2)
    
    # Término radial
    radial_term = (r * np.sqrt(2) / w0)**(abs(ell))
    
    # Envolvente Gaussiana
    gaussian = np.exp(-r**2 / w0**2)
    
    # Fase azimutal (OAM)
    phase_term = np.exp(1j * ell * phi)
    
    # Campo completo
    C_ell_p = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + abs(ell))))
    LG_field = (C_ell_p / w0) * radial_term * laguerre_term * gaussian * phase_term
    
    return LG_field
```

---

### 20. Síntesis de Pantalla de Fase (código)

**Label:** `eq_20` (Metodología)

```latex
φ(x,y) = FFT⁻¹{√Φₙ(κ) · W(κ)}
```

**Implementación Python:**
```python
def generate_phase_screen(grid_size, delta, r0, L0=20, l0=0.002):
    """
    Genera pantalla de fase turbulenta
    
    Parameters:
    - grid_size: Tamaño de grilla
    - delta: Espaciado de grilla [m]
    - r0: Parámetro de Fried [m]
    - L0: Escala externa de turbulencia [m]
    - l0: Escala interna de turbulencia [m]
    """
    # Grilla de frecuencias espaciales
    fx = np.fft.fftfreq(grid_size, d=delta)
    fy = np.fft.fftfreq(grid_size, d=delta)
    FX, FY = np.meshgrid(fx, fy)
    
    # Magnitud de frecuencia espacial
    kappa = 2*np.pi * np.sqrt(FX**2 + FY**2)
    kappa[0,0] = 1  # Evitar división por cero
    
    # Espectro de fase con escalas interna/externa
    # Modelo von Karman
    Phi_n = 0.023 * r0**(-5/3) * (kappa**2 + (1/L0)**2)**(-11/6) * \
            np.exp(-(kappa * l0 / 5.92)**2)
    
    # Ruido Gaussiano complejo
    cn = (np.random.randn(grid_size, grid_size) + \
          1j*np.random.randn(grid_size, grid_size)) / np.sqrt(2)
    
    # Síntesis por FFT
    phi = np.real(np.fft.ifft2(np.sqrt(Phi_n) * cn))
    
    return phi
```

---

## RESUMEN DE CONSTANTES FÍSICAS

| Constante | Símbolo | Valor | Unidades |
|-----------|---------|-------|----------|
| Velocidad de la luz | c | 2.998×10⁸ | m/s |
| Constante de Planck | h | 6.626×10⁻³⁴ | J·s |
| Constante de Planck reducida | ℏ | 1.055×10⁻³⁴ | J·s |
| Longitud de onda (He-Ne) | λ | 630×10⁻⁹ | m |
| Longitud de onda (telecom) | λ | 1550×10⁻⁹ | m |

## PARÁMETROS TÍPICOS DEL SISTEMA (TESIS)

| Parámetro | Símbolo | Valor | Descripción |
|-----------|---------|-------|-------------|
| Número de modos OAM | - | 8 | {-4,-3,-2,-1,+1,+2,+3,+4} |
| Longitud de onda | λ | 630 nm | He-Ne rojo |
| Cintura del haz TX | w₀ | 6 mm | Transmisor |
| Apertura TX | D_TX | 20 mm | Diámetro transmisor |
| Apertura RX | D_RX | 80 mm | Diámetro receptor |
| Distancia de propagación | L | 50 m | Enlace de prueba |
| Grilla computacional | N | 512×512 | Resolución |
| SNR objetivo | SNR | 30 dB | Relación señal-ruido |
| Turbulencia típica | Cₙ² | 10⁻¹⁵ | m⁻²/³ (ext. claro) |
| Pantallas de fase | Nₛ | 1 | Simulación |

---

**Documento generado:** $(date)  
**Archivo:** formulas_tesis_oam.md

