# Technical Support

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error

**Symptom:**
```
oom-kill: Killed process (python3) total-vm:4637628kB
```

**Solution:**
```bash
# Check memory usage
free -h

# Reduce grid size in oam_system_config.py
SYSTEM_CONFIG = {
    'grid_size': 256,  # Reduce from 512 or 1024
}

# Kill unnecessary processes
pkill -f oam_visualizer
```

#### 2. GNU Radio Blocks Not Visible

**Symptom:**
Custom OAM blocks don't appear in GNU Radio Companion

**Solution:**
```bash
# Reinstall GNU Radio blocks
cd GNU_Radio_Production7
./install_gnuradio_blocks.sh

# Restart GNU Radio Companion completely
pkill -9 gnuradio-companion
gnuradio-companion
```

#### 3. Module Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'oam_encoder_wr'
```

**Solution:**
```bash
# Set PYTHONPATH
cd GNU_Radio_Production7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 oam_complete_system.py
```

#### 4. Cache Corruption

**Symptom:**
Unexpected simulation behavior or inconsistent results

**Solution:**
```bash
# Clear cache
rm -rf GNU_Radio_Production7/current_run/
rm -rf /tmp/oam/

# Run fresh simulation
python3 GNU_Radio_Production7/oam_complete_system.py
```

#### 5. NPZ File Corruption

**Symptom:**
```
ValueError: cannot load file containing pickled data when allow_pickle=False
```

**Solution:**
The system automatically detects and cleans corrupted NPZ files on startup. If the issue persists:

```bash
# Manually remove corrupted files
find current_run/ -name "*.npz" -delete
```

#### 6. Dashboard Launch Issues

**Symptom:**
Dashboards fail to display or crash immediately

**Solution:**
```bash
# Install required packages
pip install PyQt5 matplotlib

# Test with simple mode
python3 oam_visualizer.py --mode simple_dynamic --run current --gui qt
```

#### 7. Git Authentication Failure (WSL)

**Symptom:**
```
fatal: could not read Username for 'https://github.com'
```

**Solution:**
```bash
# Configure Git credential helper for WSL
git config --global credential.helper "/mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager.exe"
```

---

## Performance Optimization

### Memory Usage Guidelines

| Grid Size | RAM Required | Recommended For |
|-----------|-------------|-----------------|
| 256       | ~500 MB     | Quick testing   |
| 512       | ~2 GB       | Standard runs   |
| 1024      | ~8 GB       | High accuracy   |
| 2048      | ~32 GB      | Publication quality |

### Computation Time Estimates

| Configuration | Grid 256 | Grid 512 | Grid 1024 |
|---------------|----------|----------|-----------|
| 2 OAM modes   | ~5 s     | ~20 s    | ~90 s     |
| 4 OAM modes   | ~10 s    | ~40 s    | ~180 s    |
| 6 OAM modes   | ~15 s    | ~60 s    | ~270 s    |
| 8 OAM modes   | ~20 s    | ~80 s    | ~360 s    |

---

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04 LTS
- **RAM**: 4 GB
- **CPU**: Dual-core 2.0 GHz
- **Disk**: 2 GB free space
- **Python**: 3.8+
- **GNU Radio**: 3.8+

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **RAM**: 16 GB
- **CPU**: Quad-core 3.0 GHz
- **Disk**: 10 GB free space (for historical runs)
- **Python**: 3.10+
- **GNU Radio**: 3.10+

---

## Logging System

### Log Levels

The system uses centralized logging via `oam_logging.py`:

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages (system continues)
- **ERROR**: Error messages (component may fail)

### Log File Locations

```
GNU_Radio_Production7/
├── laboratorio.log              # Laboratory scenario
├── exterior_moderado.log        # Moderate outdoor scenario
└── turbulencia_fuerte.log       # Strong turbulence scenario
```

### Viewing Logs

```bash
# View real-time logs
tail -f GNU_Radio_Production7/laboratorio.log

# Search for errors
grep ERROR GNU_Radio_Production7/*.log

# Filter by module
grep "oam_channel" GNU_Radio_Production7/*.log
```

---

## Dashboard Modes Reference

### Dashboard A: Simple Dynamic (Temporal Analysis)

**Purpose**: Real-time symbol-by-symbol visualization

**Command**:
```bash
python3 oam_visualizer.py --mode simple_dynamic --run current --gui qt --step 3.0
```

**Parameters**:
- `--step`: Time delay between symbols [seconds]
- `--run`: Run directory (`current` or timestamp)

### Dashboard B: QA Dynamic (Quality Metrics)

**Purpose**: Performance metrics with modal mixing analysis

**Command**:
```bash
python3 oam_visualizer.py --mode qa_dynamic --run current --gui qt --modalmix
```

**Parameters**:
- `--modalmix`: Enable modal mixing analysis
- `--step`: Update interval [seconds]

### Dashboard C: Snapshot Offline (Detailed Analysis)

**Purpose**: In-depth analysis of specific symbols

**Command**:
```bash
python3 oam_visualizer.py --mode snapshot_offline --run current --symbol 13 --gui qt
```

**Parameters**:
- `--symbol`: Symbol index to analyze

### Dashboard D: Modal Stream (Modal Separation)

**Purpose**: Dynamic modal component visualization

**Command**:
```bash
python3 oam_visualizer.py --mode modal_stream --run current --gui qt
```

---

## Data Export and Analysis

### Exporting Results

```python
import numpy as np
import json

# Load complex field data
data = np.load('current_run/fields_after_channel.npz')
fields = data['fields']

# Load metadata
with open('current_run/meta.json', 'r') as f:
    meta = json.load(f)

# Load metrics
import jsonlines
metrics = []
with jsonlines.open('current_run/metrics.jsonl') as reader:
    for obj in reader:
        metrics.append(obj)
```

### Custom Analysis Scripts

Example script to extract SNR values:

```python
import jsonlines
import numpy as np

# Read metrics
snr_values = []
with jsonlines.open('current_run/metrics.jsonl') as reader:
    for obj in reader:
        snr_values.append(obj['snr_db'])

# Calculate statistics
print(f"Mean SNR: {np.mean(snr_values):.2f} dB")
print(f"Std Dev: {np.std(snr_values):.2f} dB")
print(f"Min SNR: {np.min(snr_values):.2f} dB")
print(f"Max SNR: {np.max(snr_values):.2f} dB")
```

---

## Mathematical Foundation

For detailed mathematical documentation, see:
- `FORMULAS_MATEMATICAS_TESIS.md`: All thesis equations with explanations
- `Informefinal/tesis_latex/`: Complete LaTeX thesis document

Key equations:
- Helmholtz equation: ∇²E + k²E = 0
- OAM quantization: Lz = ℓℏ
- Laguerre-Gaussian field generation
- Kolmogorov turbulence structure function
- Normalized Cross-Correlation (NCC) detection

---

## Getting Help

### Documentation Resources
- **README.md**: Project overview and quick start
- **CONTRIBUTING.md**: Installation and usage guide
- **SECURITY.md**: Configuration reference
- **SUPPORT.md**: This troubleshooting guide (you are here)

### Contact Information
- **Email**: deibyarizac@gmail.com
- **GitHub Issues**: https://github.com/DeibyArizac/OAM/issues
- **Institution**: Universidad Industrial de Santander (UIS)
- **School**: Electrical, Electronic and Telecommunications Engineering (E³T)

### Academic Supervision
- **Director**: Dr. Omar Javier Tíjaro Rojas
- **Author**: Deiby Fernando Ariza Cadena (Code: 2195590)

---

## FAQ

**Q: Can I use this system for commercial applications?**
A: This is an academic research project. Check the LICENSE file for usage terms.

**Q: What Python version is required?**
A: Python 3.8 or later. Recommended: Python 3.10+

**Q: Does this work on Windows?**
A: The system is designed for Linux (Ubuntu). Use WSL2 for Windows.

**Q: How do I cite this work?**
A: Citation information will be provided once the thesis is published.

**Q: Can I modify the OAM mode mapping?**
A: Yes, modify `oam_encoder.py` and `oam_decoder.py`, but ensure consistency.

**Q: What's the maximum number of OAM modes supported?**
A: Currently 8 modes (±1, ±2, ±3, ±4). Can be extended by modifying the encoder/decoder.
