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
