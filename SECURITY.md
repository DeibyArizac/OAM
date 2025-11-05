# System Configuration

## Configuration Management

### Primary Configuration File

All system parameters are managed through `GNU_Radio_Production7/oam_system_config.py`. This file serves as the single source of truth for all simulation parameters.

```python
SYSTEM_CONFIG = {
    # System parameters
    'num_oam_modes': 6,              # Number of OAM channels (2, 4, 6, 8)
    'wavelength': 630e-9,            # Wavelength [m] (630 nm red laser)
    'grid_size': 512,                # Computational grid resolution

    # Transmitter parameters
    'tx_power': 1.0,                 # Transmit power [W]
    'tx_aperture_size': 0.1,         # Transmitter aperture diameter [m]
    'symbol_rate': 1.0,              # Symbol transmission rate [symbols/s]

    # Channel parameters
    'propagation_distance': 50,      # TX-RX distance [m]
    'cn2': 1e-15,                   # Atmospheric turbulence strength [m^-2/3]
    'snr_target': 30,               # Target SNR [dB]
    'atmospheric_conditions': 'clear', # Preset: 'laboratory', 'clear', 'moderate', 'strong'

    # Receiver parameters
    'rx_aperture_size': 0.1,         # Receiver aperture diameter [m]

    # Dashboard configuration
    'enable_dashboard_a': True,      # Temporal analysis dashboard
    'enable_dashboard_b': True,      # Quality metrics dashboard
    'enable_dashboard_c': False,     # Detailed snapshot (on-demand)
    'enable_dashboard_d': True,      # Modal stream analysis
    'dashboard_step_delay': 3.0,     # Time between updates [s]
}
```

---

## Parameter Descriptions

### OAM Mode Configuration

**`num_oam_modes`**: Number of OAM channels to use

Valid values: 2, 4, 6, 8

Example configurations:
- `2`: Uses modes ±1 (2 channels, 1 bit/symbol)
- `4`: Uses modes ±1, ±2 (4 channels, 2 bits/symbol)
- `6`: Uses modes ±1, ±2, ±3 (6 channels, ~2.58 bits/symbol)
- `8`: Uses modes ±1, ±2, ±3, ±4 (8 channels, 3 bits/symbol)

### Wavelength Configuration

**`wavelength`**: Optical carrier wavelength

Common values:
- `630e-9`: Red laser (630 nm)
- `1550e-9`: Telecom wavelength (1550 nm)
- `850e-9`: Near-infrared (850 nm)

### Grid Resolution

**`grid_size`**: Computational grid size (pixels)

Recommended values:
- `256`: Fast computation, lower accuracy
- `512`: Balanced (default)
- `1024`: High accuracy, slower computation
- `2048`: Maximum accuracy, high memory usage

### Atmospheric Turbulence (Cₙ²)

**`cn2`**: Refractive index structure constant

Typical values:
- `1e-17`: Laboratory/indoor (negligible turbulence)
- `1e-15`: Clear outdoor day (weak turbulence)
- `1e-14`: Moderate turbulence (daytime, moderate wind)
- `1e-13`: Strong turbulence (high thermal activity)

### Atmospheric Condition Presets

**`atmospheric_conditions`**: Quick configuration presets

Available presets:
- `'laboratory'`: Cₙ² = 10⁻¹⁷, SNR = 35 dB
- `'clear'`: Cₙ² = 10⁻¹⁵, SNR = 30 dB
- `'moderate'`: Cₙ² = 10⁻¹⁴, SNR = 25 dB
- `'strong'`: Cₙ² = 10⁻¹³, SNR = 20 dB

---

## Configuration Scenarios

### Laboratory Testing (Ideal Conditions)

```python
SYSTEM_CONFIG = {
    'num_oam_modes': 8,
    'wavelength': 630e-9,
    'grid_size': 512,
    'propagation_distance': 10,
    'cn2': 1e-17,
    'snr_target': 35,
}
```

### Outdoor Free-Space Communication

```python
SYSTEM_CONFIG = {
    'num_oam_modes': 6,
    'wavelength': 1550e-9,
    'grid_size': 1024,
    'propagation_distance': 100,
    'cn2': 1e-15,
    'snr_target': 30,
}
```

### Turbulent Atmospheric Channel

```python
SYSTEM_CONFIG = {
    'num_oam_modes': 4,
    'wavelength': 630e-9,
    'grid_size': 512,
    'propagation_distance': 50,
    'cn2': 1e-14,
    'snr_target': 25,
}
```

---

## Cache Management

The system implements intelligent caching based on configuration hash:

```bash
# Cache location
GNU_Radio_Production7/current_run/

# Clear cache if needed
rm -rf GNU_Radio_Production7/current_run/
```

The system automatically detects parameter changes and regenerates data when configuration is modified.

---

## Data Storage Structure

### Current Run Data

```
current_run/
├── fields_before_channel.npz    # Complex fields before channel
├── fields_after_channel.npz     # Complex fields after propagation
├── fields_at_decoder.npz        # Complex fields at decoder
├── meta.json                    # Run metadata and configuration
├── metrics.jsonl                # Per-symbol performance metrics
└── config_hash.txt              # Configuration hash for validation
```

### Historical Runs

```
runs/
└── <timestamp>/                 # Timestamped run directory
    ├── fields_*.npz
    ├── meta.json
    └── metrics.jsonl
```

---

## Configuration Best Practices

1. **Always modify** `oam_system_config.py`, never hardcode parameters
2. **Test changes** with small `grid_size` (256) before high-resolution runs
3. **Document scenarios** by saving different config files
4. **Monitor memory** usage with large grid sizes (≥1024)
5. **Use presets** for standard atmospheric conditions

---

## Contact

For configuration support or questions:
- Email: deibyarizac@gmail.com
- GitHub Issues: https://github.com/DeibyArizac/OAM/issues
