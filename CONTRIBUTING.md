# Contributing Guide

## Installation and Setup

### Prerequisites

- **Operating System**: Ubuntu 20.04 LTS or later
- **GNU Radio**: Version 3.8 or later
- **Python**: 3.8+
- **Required Packages**:
  - `gnuradio`
  - `python3-gnuradio`
  - `python3-pip`
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `PyQt5`

### Quick Installation

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

---

## Running the System

### Method 1: Python Standalone (Recommended)

```bash
cd GNU_Radio_Production7
python3 oam_complete_system.py
```

### Method 2: GNU Radio Companion (Visual Interface)

```bash
cd GNU_Radio_Production7
gnuradio-companion oam_system.grc
# Click Run button (▶) or press F6
```

---

## System Configuration

All parameters are centralized in `oam_system_config.py`:

```python
SYSTEM_CONFIG = {
    'num_oam_modes': 6,           # Number of OAM channels (2, 4, 6, 8)
    'wavelength': 630e-9,         # Wavelength in meters
    'grid_size': 512,             # Computational grid resolution
    'propagation_distance': 50,   # TX-RX distance [m]
    'cn2': 1e-15,                # Atmospheric turbulence [m^-2/3]
    'snr_target': 30,            # Target SNR [dB]
}
```

---

## Visualization Dashboards

```bash
cd GNU_Radio_Production7

# Dashboard A: Temporal analysis
python3 oam_visualizer.py --mode simple_dynamic --run current --gui qt

# Dashboard B: Quality metrics
python3 oam_visualizer.py --mode qa_dynamic --run current --gui qt --modalmix

# Dashboard C: Detailed snapshot
python3 oam_visualizer.py --mode snapshot_offline --run current --symbol 13 --gui qt

# Dashboard D: Modal stream analysis
python3 oam_visualizer.py --mode modal_stream --run current --gui qt
```

---

## Project Structure

```
OAM/
├── GNU_Radio_Production7/      # Current active version
│   ├── oam_complete_system.py  # Main system entry point
│   ├── oam_system_config.py    # Configuration file
│   ├── oam_encoder.py          # OAM encoder module
│   ├── oam_channel.py          # Atmospheric channel
│   ├── oam_decoder.py          # OAM decoder
│   ├── oam_visualizer.py       # Visualization system
│   └── grc/                    # GNU Radio block definitions
├── runs/                       # Historical simulation data
├── current_run/                # Active simulation output
└── Informefinal/              # Thesis LaTeX document
```

---

## Academic Context

**Author**: Deiby Fernando Ariza Cadena (Code: 2195590)
**Director**: Dr. Omar Javier Tíjaro Rojas
**Institution**: Universidad Industrial de Santander (UIS)
**School**: Electrical, Electronic and Telecommunications Engineering (E³T)

---

## Support

For issues or questions:
- Check [SUPPORT.md](SUPPORT.md) for troubleshooting
- Email: deibyarizac@gmail.com
- GitHub Issues: https://github.com/DeibyArizac/OAM/issues
