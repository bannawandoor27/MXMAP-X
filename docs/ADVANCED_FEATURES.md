# MXMAP-X Advanced Features

Complete guide to advanced endpoints for optimization, exploration, comparison, and real-time predictions.

## 📋 Table of Contents

1. [Multi-Objective Optimization](#multi-objective-optimization)
2. [Chemistry Space Exploration](#chemistry-space-exploration)
3. [Candidate Comparison](#candidate-comparison)
4. [Recipe Card Export](#recipe-card-export)
5. [WebSocket Predictions](#websocket-predictions)

---

## Multi-Objective Optimization

Find Pareto-optimal device designs that balance competing objectives.

### Endpoint

```
POST /api/v1/optimize
```

### Use Cases

- **High Performance + Low Cost**: Maximize capacitance while minimizing thickness
- **Balanced Design**: Optimize capacitance, ESR, and cycle life simultaneously
- **Constrained Optimization**: Find best design within manufacturing constraints

### Request Format

```json
{
  "objectives": [
    {
      "metric": "capacitance",
      "target": "maximize",
      "weight": 1.0,
      "constraint_min": 200.0,
      "constraint_max": null
    },
    {
      "metric": "esr",
      "target": "minimize",
      "weight": 0.8,
      "constraint_min": null,
      "constraint_max": 5.0
    },
    {
      "metric": "cycle_life",
      "target": "maximize",
      "weight": 0.6,
      "constraint_min": 8000,
      "constraint_max": null
    }
  ],
  "constraints": {
    "thickness_min": 2.0,
    "thickness_max": 15.0
  },
  "population_size": 100,
  "generations": 50
}
```

### Parameters

**Objectives:**
- `metric`: Target metric (capacitance, esr, rate_capability, cycle_life)
- `target`: "maximize" or "minimize"
- `weight`: Relative importance (0.0-1.0)
- `constraint_min`: Minimum acceptable value (optional)
- `constraint_max`: Maximum acceptable value (optional)

**Constraints:**
- `thickness_min`: Minimum film thickness (μm)
- `thickness_max`: Maximum film thickness (μm)

**Algorithm Parameters:**
- `population_size`: Number of candidates per generation (20-500)
- `generations`: Number of optimization iterations (10-200)

### Response Format

```json
{
  "pareto_optimal": [
    {
      "rank": 0,
      "composition": {
        "mxene_type": "Ti3C2Tx",
        "terminations": "O",
        "electrolyte": "H2SO4",
        "thickness_um": 5.2,
        "deposition_method": "vacuum_filtration",
        "annealing_temp_c": 135.0
      },
      "predictions": {
        "capacitance": 385.2,
        "esr": 2.1,
        "rate_capability": 87.5,
        "cycle_life": 11250
      },
      "objectives": {
        "capacitance": -385.2,
        "esr": 1.68,
        "cycle_life": -6750.0
      },
      "crowding_distance": 1.234
    }
  ],
  "total_evaluated": 100,
  "pareto_size": 15,
  "objectives": ["capacitance", "esr", "cycle_life"]
}
```

### Understanding Results

**Pareto-Optimal Solutions:**
- Solutions where improving one objective worsens another
- Represent best trade-offs between competing goals
- Sorted by crowding distance (diversity metric)

**Crowding Distance:**
- Measures solution diversity in objective space
- Higher values = more unique solutions
- Helps select diverse set of candidates

**Rank:**
- 0 = Pareto-optimal (first front)
- Higher ranks = dominated by other solutions

### Example: Maximize Capacitance + Minimize ESR

```bash
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "objectives": [
      {"metric": "capacitance", "target": "maximize", "weight": 1.0},
      {"metric": "esr", "target": "minimize", "weight": 1.0}
    ],
    "population_size": 100,
    "generations": 50
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/optimize",
    json={
        "objectives": [
            {"metric": "capacitance", "target": "maximize", "weight": 1.0},
            {"metric": "esr", "target": "minimize", "weight": 0.8},
            {"metric": "cycle_life", "target": "maximize", "weight": 0.6}
        ],
        "constraints": {"thickness_min": 2.0, "thickness_max": 15.0},
        "population_size": 100,
        "generations": 50
    }
)

pareto_solutions = response.json()["pareto_optimal"]

# Get top 5 diverse solutions
top_5 = pareto_solutions[:5]

for i, sol in enumerate(top_5, 1):
    print(f"\nSolution {i}:")
    print(f"  Capacitance: {sol['predictions']['capacitance']:.1f} mF/cm²")
    print(f"  ESR: {sol['predictions']['esr']:.2f} Ω")
    print(f"  Cycle Life: {sol['predictions']['cycle_life']} cycles")
```

---

## Chemistry Space Exploration

Visualize the design space using UMAP dimensionality reduction.

### Endpoint

```
GET /api/v1/explore?n_samples=200&n_neighbors=15&min_dist=0.1
```

### Use Cases

- **Visualize Design Space**: See relationships between materials
- **Identify Clusters**: Find groups of similar devices
- **Discover Gaps**: Identify unexplored regions
- **Guide Experiments**: Select diverse candidates for testing

### Query Parameters

- `n_samples`: Number of samples to visualize (50-1000)
- `n_neighbors`: UMAP n_neighbors parameter (5-50)
  - Lower = focus on local structure
  - Higher = preserve global structure
- `min_dist`: UMAP min_dist parameter (0.0-1.0)
  - Lower = tighter clusters
  - Higher = more spread out

### Response Format

```json
{
  "embeddings": [
    {"x": 2.34, "y": -1.56},
    {"x": 1.89, "y": 0.42}
  ],
  "devices": [
    {
      "composition": {
        "mxene_type": "Ti3C2Tx",
        "terminations": "O",
        "electrolyte": "H2SO4",
        "thickness_um": 5.0
      },
      "predictions": {
        "capacitance": 350.5,
        "esr": 2.5,
        "rate_capability": 85.0,
        "cycle_life": 10000
      },
      "embedding": {"x": 2.34, "y": -1.56}
    }
  ],
  "parameters": {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "euclidean"
  }
}
```

### Visualization Example

```python
import requests
import matplotlib.pyplot as plt
import numpy as np

# Get chemistry map
response = requests.get(
    "http://localhost:8000/api/v1/explore",
    params={"n_samples": 200, "n_neighbors": 15, "min_dist": 0.1}
)

data = response.json()

# Extract embeddings and predictions
x = [d["embedding"]["x"] for d in data["devices"]]
y = [d["embedding"]["y"] for d in data["devices"]]
capacitance = [d["predictions"]["capacitance"] for d in data["devices"]]

# Create scatter plot colored by capacitance
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=capacitance, cmap="viridis", s=50, alpha=0.6)
plt.colorbar(scatter, label="Capacitance (mF/cm²)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("MXene Supercapacitor Chemistry Space")
plt.tight_layout()
plt.savefig("chemistry_map.png", dpi=300)
```

### Interpreting the Map

- **Proximity**: Nearby points have similar compositions
- **Clusters**: Groups of related materials
- **Outliers**: Unique or unusual compositions
- **Color Gradients**: Performance trends across space

---

## Candidate Comparison

Compare 2-10 device candidates side-by-side.

### Endpoint

```
POST /api/v1/compare
```

### Use Cases

- **Design Selection**: Choose between candidate designs
- **Trade-off Analysis**: Understand performance differences
- **Sensitivity Analysis**: See impact of composition changes
- **Benchmarking**: Compare against literature values

### Request Format

```json
{
  "candidates": [
    {
      "mxene_type": "Ti3C2Tx",
      "terminations": "O",
      "electrolyte": "H2SO4",
      "thickness_um": 5.0,
      "deposition_method": "vacuum_filtration"
    },
    {
      "mxene_type": "Mo2CTx",
      "terminations": "F",
      "electrolyte": "KOH",
      "thickness_um": 10.0,
      "deposition_method": "spray_coating"
    },
    {
      "mxene_type": "V2CTx",
      "terminations": "OH",
      "electrolyte": "ionic_liquid",
      "thickness_um": 7.5,
      "deposition_method": "blade_coating"
    }
  ],
  "metrics": ["capacitance", "esr", "rate_capability", "cycle_life"]
}
```

### Response Format

```json
{
  "comparison": [
    {
      "composition": {...},
      "capacitance": 350.5,
      "esr": 2.5,
      "rate_capability": 85.0,
      "cycle_life": 10000,
      "confidence": "high",
      "confidence_score": 0.92
    }
  ],
  "rankings": {
    "capacitance": [0, 2, 1],
    "esr": [1, 0, 2],
    "rate_capability": [0, 1, 2],
    "cycle_life": [2, 0, 1]
  },
  "best_overall": 0,
  "summary": {
    "num_candidates": 3,
    "metrics_compared": ["capacitance", "esr", "rate_capability", "cycle_life"],
    "best_capacitance_idx": 0,
    "best_esr_idx": 1,
    "best_rate_capability_idx": 0,
    "best_cycle_life_idx": 2
  }
}
```

### Understanding Rankings

- **Rankings**: List of candidate indices sorted by performance
- **Best Overall**: Candidate with best average rank across all metrics
- **Summary**: Quick reference for best performer in each metric

### Example

```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": [
      {
        "mxene_type": "Ti3C2Tx",
        "terminations": "O",
        "electrolyte": "H2SO4",
        "thickness_um": 5.0,
        "deposition_method": "vacuum_filtration"
      },
      {
        "mxene_type": "Mo2CTx",
        "terminations": "F",
        "electrolyte": "KOH",
        "thickness_um": 10.0,
        "deposition_method": "spray_coating"
      }
    ]
  }'
```

---

## Recipe Card Export

Generate fabrication instructions for a device.

### Endpoint

```
GET /api/v1/recipes/{device_id}
```

### Use Cases

- **Lab Protocols**: Generate step-by-step instructions
- **Documentation**: Record fabrication procedures
- **Reproducibility**: Share detailed recipes
- **Training**: Teach fabrication techniques

### Response Format

```json
{
  "recipe_id": "MXMAP-000042",
  "device_composition": {
    "mxene_type": "Ti3C2Tx",
    "terminations": "O",
    "electrolyte": "H2SO4",
    "thickness_um": 5.0,
    "deposition_method": "vacuum_filtration"
  },
  "processing_steps": [
    {
      "step": 1,
      "title": "MXene Preparation",
      "description": "Prepare Ti3C2Tx MXene with O terminations",
      "duration": "2-4 hours",
      "temperature": "Room temperature",
      "equipment": ["Sonicator", "Centrifuge", "Vacuum filtration setup"]
    },
    {
      "step": 2,
      "title": "Electrolyte Preparation",
      "description": "Prepare 1.0M H2SO4 solution",
      "duration": "30 minutes",
      "temperature": "Room temperature",
      "equipment": ["Volumetric flask", "Magnetic stirrer"]
    }
  ],
  "predicted_performance": {
    "areal_capacitance": "350.5 mF/cm²",
    "esr": "2.5 Ω",
    "rate_capability": "85.0%",
    "cycle_life": "10000 cycles",
    "confidence": "high"
  },
  "materials_list": [
    {"name": "Ti3C2Tx MXene", "quantity": "~100 mg", "purity": "Research grade"},
    {"name": "H2SO4", "quantity": "50-100 mL", "purity": "ACS grade"}
  ],
  "safety_notes": [
    "Wear appropriate PPE (lab coat, gloves, safety glasses)",
    "Handle H2SO4 with care - corrosive material",
    "Work in well-ventilated area or fume hood"
  ],
  "estimated_time": "6.5 hours",
  "difficulty": "Intermediate"
}
```

### Example

```bash
curl http://localhost:8000/api/v1/recipes/42
```

### Export to PDF (Python)

```python
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Get recipe
response = requests.get("http://localhost:8000/api/v1/recipes/42")
recipe = response.json()

# Generate PDF
pdf = canvas.Canvas(f"recipe_{recipe['recipe_id']}.pdf", pagesize=letter)
pdf.setTitle(f"Recipe {recipe['recipe_id']}")

# Add content
y = 750
pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(50, y, f"Recipe {recipe['recipe_id']}")

y -= 30
pdf.setFont("Helvetica", 12)
pdf.drawString(50, y, f"Difficulty: {recipe['difficulty']}")
pdf.drawString(200, y, f"Time: {recipe['estimated_time']}")

# Add processing steps
y -= 40
pdf.setFont("Helvetica-Bold", 14)
pdf.drawString(50, y, "Processing Steps:")

for step in recipe["processing_steps"]:
    y -= 25
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, f"Step {step['step']}: {step['title']}")
    y -= 15
    pdf.setFont("Helvetica", 10)
    pdf.drawString(70, y, step['description'])

pdf.save()
```

---

## WebSocket Predictions

Real-time predictions with progress updates.

### Endpoint

```
WS /api/v1/ws/predict
```

### Use Cases

- **Batch Processing**: Monitor progress of large batches
- **Interactive Tools**: Build responsive UIs
- **Long-Running Tasks**: Track optimization progress
- **Real-Time Feedback**: Update visualizations live

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/predict');

ws.onopen = () => {
  console.log('Connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Message Format

**Send (Batch Prediction):**
```json
{
  "type": "predict_batch",
  "data": {
    "devices": [
      {
        "mxene_type": "Ti3C2Tx",
        "terminations": "O",
        "electrolyte": "H2SO4",
        "thickness_um": 5.0,
        "deposition_method": "vacuum_filtration"
      }
    ]
  }
}
```

**Receive (Progress Update):**
```json
{
  "type": "progress",
  "progress": 0.5,
  "message": "Processing device 50/100"
}
```

**Receive (Complete):**
```json
{
  "type": "complete",
  "results": [
    {
      "capacitance": 350.5,
      "esr": 2.5,
      "rate_capability": 85.0,
      "cycle_life": 10000,
      "confidence": "high"
    }
  ]
}
```

### Python Example

```python
import asyncio
import websockets
import json

async def predict_batch():
    uri = "ws://localhost:8000/api/v1/ws/predict"
    
    async with websockets.connect(uri) as websocket:
        # Send batch prediction request
        await websocket.send(json.dumps({
            "type": "predict_batch",
            "data": {
                "devices": [
                    {
                        "mxene_type": "Ti3C2Tx",
                        "terminations": "O",
                        "electrolyte": "H2SO4",
                        "thickness_um": 5.0,
                        "deposition_method": "vacuum_filtration"
                    }
                    # ... more devices
                ]
            }
        }))
        
        # Receive updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "progress":
                print(f"Progress: {data['progress']*100:.1f}% - {data['message']}")
            
            elif data["type"] == "complete":
                print("Complete!")
                print(f"Results: {len(data['results'])} predictions")
                break
            
            elif data["type"] == "error":
                print(f"Error: {data['message']}")
                break

asyncio.run(predict_batch())
```

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/predict');

ws.onopen = () => {
  // Send batch prediction
  ws.send(JSON.stringify({
    type: 'predict_batch',
    data: {
      devices: [
        {
          mxene_type: 'Ti3C2Tx',
          terminations: 'O',
          electrolyte: 'H2SO4',
          thickness_um: 5.0,
          deposition_method: 'vacuum_filtration'
        }
      ]
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'progress') {
    updateProgressBar(data.progress);
    console.log(data.message);
  } else if (data.type === 'complete') {
    displayResults(data.results);
  }
};
```

---

## Performance Considerations

### Optimization
- **Population Size**: Larger = better coverage, slower
- **Generations**: More = better convergence, slower
- **Typical Runtime**: 10-30 seconds for 100 population × 50 generations

### Chemistry Exploration
- **Sample Size**: More = better map, slower
- **UMAP Parameters**: Affect map quality and speed
- **Typical Runtime**: 5-15 seconds for 200 samples

### Comparison
- **Candidates**: 2-10 supported
- **Typical Runtime**: <1 second for 5 candidates

### WebSocket
- **Batch Size**: Recommended <100 devices
- **Update Frequency**: Every device (can be throttled)
- **Connection Timeout**: 5 minutes default

---

## Best Practices

1. **Start Small**: Test with small populations/samples first
2. **Use Constraints**: Narrow search space for faster optimization
3. **Cache Results**: Store Pareto fronts for reuse
4. **Monitor Progress**: Use WebSocket for long-running tasks
5. **Validate Results**: Check predictions against known data

---

**Last Updated**: 2024-01-15  
**API Version**: v1  
**Status**: Production Ready ✅


## AC-Line Filtering Mode

### Overview

The AC-Line Filtering Mode enables design and optimization of on-chip MXene microsupercapacitors (MSCs) for AC-line filtering applications. This mode predicts electrochemical impedance spectroscopy (EIS) behavior, phase angles, and ripple attenuation performance for interdigitated electrode geometries.

### Key Features

- **EIS Surrogate Model**: Rs + (CPE || Rleak) circuit model for accurate impedance prediction
- **Geometry-to-Parameters Mapping**: Converts electrode geometry to circuit parameters
- **Layout Optimization**: NSGA-II inspired multi-objective optimization for minimal footprint
- **Bode/Nyquist Visualization**: Interactive plots with frequency-specific readouts
- **CSV Data Fitting**: Fit circuit parameters from measured impedance data
- **Recipe Export**: JSON export of optimized geometries and fabrication parameters

### Circuit Model

The filtering mode uses a constant phase element (CPE) based circuit:

```
Z(jω) = Rs + (Z_CPE || R_leak)

where:
  Z_CPE(jω) = 1 / (Q * (jω)^α)
  0 < α ≤ 1
```

**Parameters:**
- **Rs**: Series resistance (Ω) - includes sheet resistance and contact resistance
- **Q**: CPE coefficient (F·s^(α-1)) - scales with active area and porosity
- **α**: CPE exponent - depends on porosity and surface roughness (0.75-0.95)
- **R_leak**: Leakage resistance (Ω) - typically very high for solid/gel electrolytes

### Key Performance Indicators (KPIs)

1. **Phase @ 120 Hz**: Target ≤ -80° for good filtering (ideal capacitor = -90°)
2. **Capacitance @ 120 Hz**: Areal capacitance in mF/cm²
3. **Impedance @ 120 Hz**: |Z| in Ω (lower is better for filtering)
4. **Attenuation @ 50/60 Hz**: Ripple attenuation in dB (negative = filtering)
5. **f(φ=-60°)**: Frequency where phase reaches -60° (kHz response)
6. **Device Area**: Total footprint in mm²

### API Endpoints

#### POST /api/v1/filtering/predict

Predict filtering performance for a given geometry.

**Request:**
```json
{
  "frequency_range_hz": [1, 100000],
  "load_resistance_ohm": 33,
  "geometry": {
    "finger_width_um": 8,
    "finger_spacing_um": 8,
    "finger_length_um": 2000,
    "num_fingers_per_electrode": 50,
    "overlap_length_um": 1800,
    "thickness_nm": 200,
    "substrate": "Si/SiO2"
  },
  "mxene_film": {
    "flake_size_um": 2.0,
    "porosity_pct": 30,
    "sheet_res_ohm_sq": 10,
    "terminations": "-O,-OH",
    "electrolyte": "PVA/H2SO4",
    "process": "photolithography"
  },
  "fit_from_data": {
    "enable": false,
    "digitized_bode_csv_path": null
  }
}
```

**Response:**
```json
{
  "kpis": {
    "phase_deg_120hz": -86.4,
    "capacitance_mf_cm2_120hz": 12.1,
    "impedance_ohm_120hz": 1.4,
    "attenuation_db_50hz": -18.2,
    "attenuation_db_60hz": -18.7,
    "f_phi_minus60_deg_hz": 18500,
    "device_area_mm2": 3.20
  },
  "bode": {
    "freq_hz": [...],
    "mag_ohm": [...],
    "phase_deg": [...]
  },
  "nyquist": {
    "re_ohm": [...],
    "im_ohm": [...]
  },
  "params": {
    "Rs": 0.8,
    "Q": 0.45,
    "alpha": 0.86,
    "Rleak": 2000
  },
  "recipe": {
    "geometry": {...},
    "mxene_film": {...},
    "assumptions": "..."
  }
}
```

#### POST /api/v1/filtering/optimize

Optimize electrode layout to meet filtering constraints with minimal footprint.

**Request:**
```json
{
  "frequency_range_hz": [1, 100000],
  "load_resistance_ohm": 33,
  "mxene_film": {...},
  "constraints": {
    "max_area_mm2": 4.0,
    "min_spacing_um": 5,
    "targets": {
      "phase_deg_120hz": -85,
      "C_min_mf_cm2_120hz": 10,
      "Z_max_ohm_120hz": 2.0
    }
  }
}
```

**Response:**
```json
{
  "solutions": [
    {
      "geometry": {...},
      "kpis": {...},
      "params": {...},
      "bode": {...},
      "nyquist": {...}
    }
  ],
  "num_feasible": 15,
  "computation_time_s": 2.3
}
```

#### POST /api/v1/filtering/fit

Fit circuit parameters from measured Bode data (CSV format).

**CSV Format:**
```csv
freq_hz,mag_ohm,phase_deg
1,125.3,-82.1
10,42.7,-84.5
100,8.2,-86.3
...
```

**Request:**
```json
{
  "digitized_bode_csv_path": "/path/to/data.csv",
  "electrolyte": "PVA/H2SO4",
  "process": "photolithography",
  "fit_rleak": false
}
```

**Response:**
```json
{
  "status": "success",
  "params": {
    "Rs": 0.85,
    "Q": 0.42,
    "alpha": 0.87,
    "Rleak": 1e9
  },
  "calibration_key": "PVA/H2SO4_photolithography"
}
```

#### GET /api/v1/filtering/presets

Get preset configurations (load resistances, geometry bounds, process templates).

### Web Interface

Access the filtering interface at `/filtering`.

**Features:**
- Left panel: Geometry and film property inputs with presets
- KPI header: Real-time display of all 7 KPIs
- Bode plot: |Z| and phase vs frequency with markers at 50/60/120 Hz
- Nyquist plot: Complex impedance plane
- Layout optimizer: Multi-objective solver with Pareto table
- CSV import: Fit parameters from measured data
- Recipe export: JSON download of optimized design

### Design Guidelines

**For AC-Line Filtering (50/60 Hz):**
1. Target phase @ 120 Hz: -85° to -88° (closer to -90° is better)
2. Minimize impedance @ 120 Hz: < 2 Ω preferred
3. Maximize areal capacitance: > 10 mF/cm² for compact designs
4. Attenuation @ 60 Hz: < -10 dB for effective filtering

**Geometry Optimization:**
- Increase overlap length → higher capacitance, larger area
- Decrease finger spacing → higher capacitance, harder fabrication
- Increase number of fingers → higher capacitance, larger area
- Optimize thickness: 150-300 nm balances performance and transparency

**Material Selection:**
- Lower porosity (15-25%) → higher α, better capacitive behavior
- Lower sheet resistance (5-15 Ω/sq) → lower Rs, better filtering
- PVA/H₂SO₄ gel electrolyte → good performance, easy processing
- Photolithography → finest features (2 μm), best performance

### Performance Targets

- **Prediction latency**: < 150 ms for 300 frequency points
- **Optimization**: 60 population × 40 generations, returns top-5 solutions
- **Plot decimation**: ≤ 512 points per curve for fast rendering
- **Accuracy**: Phase within ±5° at 120 Hz, |Z| within ±10% (10-1000 Hz)

### Acceptance Criteria

✓ **Numerics**: Default geometry achieves phase ≤ -80° @ 120 Hz and attenuation ≤ -6 dB @ 60 Hz  
✓ **Solver**: Returns ≥3 feasible Pareto designs under constraints  
✓ **Fit**: Reproduces φ within ±5° @ 120 Hz and |Z| within ±10% (10-1k Hz)  
✓ **UI**: KPIs visible, Bode/Nyquist interactive, CSV import works, recipe export works

### Example Use Cases

**1. Design for 33 Ω Load (Typical LED Driver)**
```python
# Target: Phase ≤ -85° @ 120 Hz, Area < 4 mm²
geometry = {
    "finger_width_um": 8,
    "finger_spacing_um": 8,
    "finger_length_um": 2000,
    "num_fingers_per_electrode": 50,
    "overlap_length_um": 1800,
    "thickness_nm": 200
}
# Expected: Phase ~ -86°, C ~ 12 mF/cm², Area ~ 3.2 mm²
```

**2. Optimize for Minimal Footprint**
```python
# Constraints: Phase ≤ -80°, C ≥ 10 mF/cm², Area ≤ 4 mm²
# Solver will explore finger width, spacing, length, count
# Returns 5 Pareto-optimal solutions sorted by area
```

**3. Calibrate from Measured Data**
```python
# Upload CSV with measured Bode data
# Fit Rs, Q, α to match experimental curves
# Store calibration factors for future predictions
```

### Database Schema

**filtering_models** table:
- Stores fitted circuit parameters and calibration factors
- Keyed by (electrolyte, process) for reuse

**filtering_runs** table:
- Logs all prediction/optimization runs
- Stores input parameters, KPIs, and results for analysis

### References

- CPE model: Brug et al., J. Electroanal. Chem. 176, 275 (1984)
- MXene MSCs: Kurra et al., Nano Energy 13, 500 (2015)
- AC-line filtering: Shao et al., Nat. Commun. 4, 2381 (2013)
