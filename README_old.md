# OAM Optical Communication System

**Orbital Angular Momentum Encoding and Decoding for GNU Radio**

A complete optical communication system based on Orbital Angular Momentum (OAM) of light, implemented as custom GNU Radio blocks for research and educational purposes.

---

## Overview

This project implements a sophisticated optical communication system that leverages the orbital angular momentum (OAM) of light beams to multiplex information across multiple spatial channels. The system provides a comprehensive simulation environment for studying OAM-based free-space optical communications under realistic atmospheric conditions.

### Key Features

- **Laguerre-Gaussian Beam Generation** with configurable topological charges
- **Direct Bit-to-Mode Encoding** for efficient data transmission
- **Atmospheric Channel Simulation** including turbulence and noise effects
- **Normalized Cross-Correlation Decoding** for robust symbol detection
- **Multi-Dashboard Visualization System** for real-time analysis and validation
- **GNU Radio Integration** with custom blocks for visual flowgraph design

---

## Project Information

**Author:** Deiby Fernando Ariza Cadena
**Email:** deibyarizac@gmail.com
**GitHub:** [@DeibyArizac](https://github.com/DeibyArizac)

**Thesis Director:** Dr. Omar Javier Tíjaro Rojas
**Email:** ojtijaro@uis.edu.co

**Institution:** School of Electrical, Electronic and Telecommunications Engineering (E³T)
Universidad Industrial de Santander, Colombia

**Repository:** [https://github.com/DeibyArizac/OAM](https://github.com/DeibyArizac/OAM)

---

## System Architecture

The system follows a modular design with four main processing stages:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │ -> │   Encoder   │ -> │   Channel   │ -> │   Decoder   │
│  (Message)  │    │ (OAM Modes) │    │ (Turbulence)│    │  (Detect)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                  |
                                                                  v
                                                          ┌─────────────┐
                                                          │ Visualizer  │
                                                          │ (Analysis)  │
                                                          └─────────────┘
```

### Processing Pipeline

1. **OAM Source** - Generates digital messages with STX/ETX framing
2. **OAM Encoder** - Maps bits to OAM modes and generates Laguerre-Gaussian beams
3. **OAM Channel** - Simulates atmospheric propagation with Kolmogorov turbulence
4. **OAM Decoder** - Recovers transmitted symbols using modal correlation
5. **OAM Visualizer** - Provides comprehensive analysis dashboards

---

## System Requirements

### Operating System
- **Ubuntu 22.04 LTS** or later (recommended)
- Other Linux distributions with GNU Radio 3.10+ support

### Core Dependencies
- **GNU Radio 3.10+** - Software-defined radio framework
- **Python 3.8+** - Core programming environment
- **NumPy 1.20.0+** - Numerical computations
- **SciPy 1.7.0+** - Scientific functions (Bessel, special functions)
- **Matplotlib 3.3.0+** - Plotting and visualization
- **PyQt5 5.15.0+** - GUI framework for dashboards

### Optional Tools
- **Git** - Version control (for cloning repository)
- **pdflatex** - Documentation generation (optional)

---

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/DeibyArizac/OAM.git
cd OAM

# Install system dependencies
sudo apt update
sudo apt install gnuradio python3-gnuradio python3-pip

# Install Python dependencies
pip3 install -r requirements.txt

# Install GNU Radio custom blocks
./install.sh
```

### Detailed Installation Steps

#### 1. Install GNU Radio

```bash
sudo apt update
sudo apt install gnuradio python3-gnuradio
```

Verify installation:
```bash
gnuradio-companion --version
```

Expected output: `GNU Radio Companion 3.10.x` or later

#### 2. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

This installs:
- `numpy` - Array operations and mathematical functions
- `scipy` - Laguerre polynomials and Bessel functions
- `matplotlib` - Plotting and visualization
- `PyQt5` - GUI framework

#### 3. Configure GNU Radio Custom Blocks

```bash
./install.sh
```

This script copies the custom block definitions (`.block.yml` files) to `~/.grc_gnuradio/blocks/`, making them available in GNU Radio Companion.

Verify installation:
```bash
ls ~/.grc_gnuradio/blocks/oam_*.block.yml
```

Expected output: 5 files (oam_source, oam_encoder, oam_channel, oam_decoder, oam_visualizer)

---

## Usage

### Method 1: GNU Radio Companion (Visual Interface)

1. Launch GNU Radio Companion:
```bash
gnuradio-companion
```

2. Open the example flowgraph:
```
File -> Open -> /path/to/OAM/oam_complete_flowgraph.grc
```

3. Locate the OAM blocks in the block library under category `[OAM]`

4. Configure parameters as needed (see Configuration section)

5. Execute the flowgraph:
```
Run -> Execute (F6)
```

### Method 2: Python Script (Direct Execution)

```bash
cd /path/to/OAM
python3 oam_complete_system.py
```

This method provides faster execution and is recommended for batch simulations or automated testing.

### Method 3: Headless Mode (No GUI)

For automated simulations without visualization:

```bash
python3 oam_complete_system.py --headless
```

Results are automatically saved to `current_run/` directory.

---

## GNU Radio Custom Blocks

The system provides 5 custom blocks under the `[OAM]` category:

### OAM Source

Generates digital message data with framing protocol.

**Parameters:**
- `message_text` - Text message to transmit (default: "UIS")
- `symbol_rate` - Symbol transmission rate in Hz (default: 32000)

**Output:** Trigger signal for pipeline execution

### OAM Encoder

Encodes bit streams into OAM spatial modes.

**Parameters:**
- `num_oam_modes` - Number of OAM channels: 2, 4, 6, or 8 (default: 6)
- `wavelength` - Optical wavelength in meters (default: 1550e-9)
- `tx_power` - Transmitter power in watts (default: 0.01)
- `tx_aperture_size` - Transmitter aperture diameter in meters (default: 0.01)
- `grid_size` - Computational grid resolution (default: 512)

**Output:** Complex field arrays (stored in pipeline)

**Modulation Scheme:**
- 2 modes: [-1, +1] -> 1 bit/symbol
- 4 modes: [-2, -1, +1, +2] -> 2 bits/symbol
- 6 modes: [-3, -2, -1, +1, +2, +3] -> 3 bits/symbol
- 8 modes: [-4, -3, -2, -1, +1, +2, +3, +4] -> 4 bits/symbol

### OAM Channel

Simulates atmospheric free-space propagation.

**Parameters:**
- `propagation_distance` - TX-RX distance in meters (default: 50)
- `cn2` - Refractive index structure constant in m^(-2/3) (default: 1e-15)
- `snr_target` - Target signal-to-noise ratio in dB (default: 30)
- `enable_turbulence` - Enable atmospheric turbulence simulation (default: True)
- `enable_noise` - Enable AWGN noise addition (default: True)

**Atmospheric Conditions:**
- `cn2 = 1e-17` - Laboratory (very weak turbulence)
- `cn2 = 1e-15` - Clear atmosphere (weak turbulence)
- `cn2 = 1e-14` - Moderate turbulence
- `cn2 = 1e-13` - Strong turbulence

### OAM Decoder

Recovers transmitted symbols from received optical fields.

**Parameters:**
- `rx_aperture_size` - Receiver aperture diameter in meters (default: 0.035)
- `detection_method` - Symbol detection algorithm (default: "ncc")
- `correlation_threshold` - Minimum correlation for valid detection (default: 0.5)

**Detection Methods:**
- `ncc` - Normalized Cross-Correlation (robust, recommended)
- `intensity` - Intensity pattern matching (faster, less robust)

### OAM Visualizer

Launches real-time analysis dashboards.

**Parameters:**
- `enable_dashboard_a` - Temporal field evolution (default: True)
- `enable_dashboard_b` - Quality metrics (SNR, BER, NCC) (default: True)
- `enable_dashboard_c` - Detailed symbol snapshots (default: True)
- `enable_dashboard_d` - Modal decomposition analysis (default: True)
- `enable_dashboard_e` - Integrated validation dashboard (default: True)
- `dashboard_step_delay` - Update interval in seconds (default: 3.0)

---

## Configuration

System configuration is centralized in `oam_system_config.py`. All parameters can be modified in a single location:

```python
SYSTEM_CONFIG = {
    # System configuration
    'num_oam_modes': 6,              # Number of OAM channels (2, 4, 6, 8)
    'wavelength': 1550e-9,           # Wavelength (m) - 1550nm telecom IR
    'grid_size': 512,                # Computational grid (512x512)
    'message_text': 'UIS',           # Message to transmit

    # Transmitter
    'tx_power': 0.01,                # TX power (W) - 10 mW
    'tx_aperture_size': 0.01,        # TX aperture (m) - 10 mm
    'symbol_rate': 32000,            # Symbol rate (Hz) - 32 kHz

    # Channel (atmospheric)
    'propagation_distance': 50,      # Distance (m)
    'cn2': 1e-15,                    # Turbulence strength (m^-2/3)
    'snr_target': 30,                # Target SNR (dB)

    # Receiver
    'rx_aperture_size': 0.035,       # RX aperture (m) - 35 mm

    # Dashboards
    'enable_dashboard_a': True,      # Enable temporal analysis
    'enable_dashboard_b': True,      # Enable QA metrics
    'enable_dashboard_c': True,      # Enable snapshot view
    'enable_dashboard_d': True,      # Enable modal stream
    'enable_dashboard_e': True,      # Enable validation dashboard
    'dashboard_step_delay': 3.0,     # Dashboard update interval (s)
}
```

### Common Configuration Scenarios

**Laboratory Simulation (Clean Conditions)**
```python
'propagation_distance': 10,
'cn2': 1e-17,
'snr_target': 35
```

**Outdoor Short-Range Link**
```python
'propagation_distance': 100,
'cn2': 1e-15,
'snr_target': 25
```

**Challenging Atmospheric Conditions**
```python
'propagation_distance': 500,
'cn2': 1e-14,
'snr_target': 20
```

---

## Dashboard System

The visualization system provides 5 specialized dashboards for comprehensive analysis:

### Dashboard A: Temporal Field Analysis
- Real-time visualization of field evolution
- Before/after channel comparison
- Symbol-by-symbol progression
- Intensity and phase distributions

### Dashboard B: Quality Metrics
- Signal-to-Noise Ratio (SNR) tracking
- Bit Error Rate (BER) estimation
- Normalized Cross-Correlation (NCC) per mode
- Modal mixing visualization

### Dashboard C: Detailed Snapshot
- In-depth analysis of individual symbols
- Modal decomposition visualization
- Intensity profiles and phase maps
- Correlation matrix display

### Dashboard D: Modal Stream Analysis
- Dynamic modal separation visualization
- Sign detection confidence metrics
- Real-time modal power distribution
- Dominant mode tracking

### Dashboard E: Integrated Validation
- System-level performance metrics
- Atmospheric condition monitoring
- Turbulence effect visualization
- Comprehensive quality assessment

---

## Mathematical Foundation

### Laguerre-Gaussian Modes

The system generates Laguerre-Gaussian (LG) beams with helical phase fronts:

```
LG_p^ℓ(r, θ, z) = A · (√2 r / w(z))^|ℓ| · L_p^|ℓ|(2r² / w(z)²) ·
                   exp(-r² / w(z)²) · exp(i ℓ θ) · exp(-i k z)
```

Where:
- `ℓ` = topological charge (orbital angular momentum quantum number)
- `p` = radial mode index (p = 0 for fundamental LG modes)
- `L_p^|ℓ|` = generalized Laguerre polynomial
- `w(z)` = beam waist at position z
- `k` = wave vector = 2π/λ

### OAM Modulation Scheme

Direct bit-to-mode mapping:
- Binary `0` -> negative OAM mode (ℓ < 0)
- Binary `1` -> positive OAM mode (ℓ > 0)

For 6 channels [-3, -2, -1, +1, +2, +3]:
- 3 bits encode magnitude: 001->ℓ₁, 010->ℓ₂, 011->ℓ₃
- Each bit position selects sign: 0->negative, 1->positive
- Total: 3 bits per symbol, 6 distinct modes

### Atmospheric Turbulence Model

Kolmogorov turbulence with phase screens:

```
Φ_n(f) = 0.033 · C_n² · L · exp(-(f·l₀/5.92)²) / (f² + (1/L₀)²)^(11/6)
```

Where:
- `C_n²` = refractive index structure constant (turbulence strength)
- `L` = propagation distance
- `L₀` = outer scale of turbulence (typically 20 m)
- `l₀` = inner scale of turbulence (typically 2 mm)
- `f` = spatial frequency

### Normalized Cross-Correlation (NCC) Detection

Symbol detection using template matching:

```
NCC(ℓ) = |⟨E_rx | LG_ℓ⟩| / (||E_rx|| · ||LG_ℓ||)
```

Where:
- `E_rx` = received optical field
- `LG_ℓ` = reference Laguerre-Gaussian template for mode ℓ
- `⟨·|·⟩` = inner product (spatial integral)
- `||·||` = field norm

---

## Project Structure

```
OAM/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── install.sh                     # Installation script
├── INSTALACION.txt                # Installation notes (Spanish)
│
├── oam_complete_system.py         # Main system executable
├── oam_complete_flowgraph.grc     # GNU Radio flowgraph example
│
├── Core Modules
│   ├── oam_source.py              # Message generation
│   ├── oam_encoder.py             # OAM encoding and LG beam generation
│   ├── oam_channel.py             # Atmospheric channel simulation
│   ├── oam_decoder.py             # Symbol decoding and detection
│   └── oam_visualizer.py          # Visualization dashboard system
│
├── GNU Radio Wrappers
│   ├── oam_source_wr.py           # Source wrapper for GNU Radio
│   ├── oam_encoder_wr.py          # Encoder wrapper
│   ├── oam_channel_wr.py          # Channel wrapper
│   ├── oam_decoder_wr.py          # Decoder wrapper
│   └── oam_visualizer_wr.py       # Visualizer wrapper
│
├── System Infrastructure
│   ├── oam_system_config.py       # Centralized configuration
│   ├── config_manager.py          # Parameter management
│   ├── pipeline.py                # Global data pipeline
│   ├── oam_logging.py             # Unified logging system
│   └── gnuradio_cache.py          # Cache management
│
├── GNU Radio Block Definitions
│   └── grc/
│       ├── oam_source.block.yml
│       ├── oam_encoder.block.yml
│       ├── oam_channel.block.yml
│       ├── oam_decoder.block.yml
│       └── oam_visualizer.block.yml
│
└── Runtime Data
    ├── current_run/               # Current simulation data
    ├── pipeline_logs/             # Execution logs
    └── __pycache__/               # Python bytecode cache
```

---

## Troubleshooting

### GNU Radio Companion Issues

**Problem:** Custom blocks do not appear in GNU Radio Companion

**Solution:**
```bash
# Verify block installation
ls ~/.grc_gnuradio/blocks/oam_*.block.yml

# Re-run installation script
./install.sh

# Restart GNU Radio Companion completely
pkill -9 gnuradio-companion
gnuradio-companion
```

---

**Problem:** Flowgraph fails to execute with "block not found" error

**Solution:**
```bash
# Set Python path to current directory
cd /path/to/OAM
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 oam_complete_system.py
```

---

### Python Import Errors

**Problem:** `ModuleNotFoundError: No module named 'oam_*'`

**Solution:**
```bash
# Ensure you're executing from the OAM directory
cd /path/to/OAM
python3 oam_complete_system.py

# Or set PYTHONPATH explicitly
export PYTHONPATH="/path/to/OAM:${PYTHONPATH}"
```

---

**Problem:** `ModuleNotFoundError: No module named 'scipy'`

**Solution:**
```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Verify installation
python3 -c "import numpy, scipy, matplotlib; print('Dependencies OK')"
```

---

### Visualization Issues

**Problem:** Dashboards do not launch or crash immediately

**Solution:**
```bash
# Install PyQt5
pip3 install PyQt5

# Verify Qt installation
python3 -c "from PyQt5 import QtWidgets; print('PyQt5 OK')"

# Check dashboard configuration in oam_system_config.py
# Set enable_dashboard_* = True for desired dashboards
```

---

**Problem:** Matplotlib backend error: "cannot connect to X server"

**Solution:**
```bash
# For headless servers, use Agg backend
export MPLBACKEND=Agg
python3 oam_complete_system.py

# Or install virtual framebuffer
sudo apt install xvfb
xvfb-run -a python3 oam_complete_system.py
```

---

### Performance Issues

**Problem:** System runs very slowly or crashes with memory error

**Solution:**
```python
# Reduce grid resolution in oam_system_config.py
'grid_size': 256,  # Instead of 512 or 1024

# Reduce number of OAM modes
'num_oam_modes': 4,  # Instead of 6 or 8

# Reduce message length
'message_text': 'TEST',  # Shorter message
```

---

**Problem:** Numerical instability or NaN values in results

**Solution:**
```python
# Check turbulence parameters - very high Cn² can cause instability
'cn2': 1e-15,  # Start with weak turbulence

# Ensure reasonable SNR values
'snr_target': 25,  # Not too low (< 10 dB may be unrealistic)

# Verify aperture sizes are physical
'tx_aperture_size': 0.01,  # 10 mm (not too small)
'rx_aperture_size': 0.035,  # 35 mm (larger than TX)
```

---

## Research Applications

This system is designed for research and education in:

- **Free-Space Optical Communications** - OAM multiplexing for high-capacity links
- **Atmospheric Optics** - Turbulence effects on structured light propagation
- **Quantum Communications** - OAM as high-dimensional quantum state space
- **Optical Information Processing** - Spatial mode manipulation and detection
- **Educational Demonstrations** - Visual understanding of OAM physics

---

## Performance Benchmarks

Typical performance on a modern workstation (Intel i7, 16GB RAM):

| Configuration | Grid Size | Symbols/sec | Memory Usage |
|---------------|-----------|-------------|--------------|
| 2 modes       | 256x256   | ~50         | ~500 MB      |
| 4 modes       | 512x512   | ~20         | ~2 GB        |
| 6 modes       | 512x512   | ~15         | ~3 GB        |
| 8 modes       | 1024x1024 | ~5          | ~8 GB        |

Note: Performance varies with CPU, turbulence settings, and dashboard configuration.

---

## Known Limitations

- **Computational Complexity**: Large grid sizes (1024+) require significant memory and CPU
- **Real-Time Constraints**: Not suitable for real-time signal processing (simulation only)
- **Turbulence Model**: Uses split-step method with phase screens (approximation)
- **Crosstalk**: Does not model non-ideal receiver optics or aperture diffraction effects
- **Polarization**: Currently assumes single linear polarization state

---

## Future Enhancements

Planned improvements for future versions:

- **GPU Acceleration** using CUDA or OpenCL for faster field propagation
- **Adaptive Optics** simulation with wavefront correction
- **Hybrid OAM-Polarization** multiplexing schemes
- **Machine Learning** decoders for improved turbulence resilience
- **Real Hardware Interface** for experimental validation with SDR platforms

---

## Citation

If you use this system in academic work, please cite:

```
Ariza Cadena, D.F. (2025). OAM Optical Communication System for GNU Radio.
Universidad Industrial de Santander, Colombia.
https://github.com/DeibyArizac/OAM
```

---

## License

This project is part of academic research at Universidad Industrial de Santander (UIS).

The code is provided for educational and research purposes. For commercial use or redistribution, please contact the authors.

---

## Acknowledgments

**Universidad Industrial de Santander**
School of Electrical, Electronic and Telecommunications Engineering (E³T)

**GNU Radio Project**
Open-source software-defined radio framework

**Research Community**
Contributors to OAM and free-space optical communications research

---

## Contact and Support

For questions, suggestions, or collaboration opportunities:

**Author:** Deiby Fernando Ariza Cadena
Email: deibyarizac@gmail.com
GitHub: [@DeibyArizac](https://github.com/DeibyArizac)

**Thesis Director:** Dr. Omar Javier Tíjaro Rojas
Email: ojtijaro@uis.edu.co

**Issues and Bug Reports:**
Please use the GitHub issue tracker: [https://github.com/DeibyArizac/OAM/issues](https://github.com/DeibyArizac/OAM/issues)

---

**Last Updated:** October 2024
**Version:** 1.0 (Production7)
