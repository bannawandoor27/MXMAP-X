# Electrochromic Visualization System

## Overview

The Electrochromic Visualization System provides real-time color prediction and electrochemical behavior visualization for MXene-based electrochromic devices. It includes voltage control, 3D rendering, CV/GCD curves, and animated voltage sweeps.

## Features

### 1. Voltage Slider Component

**Interactive Voltage Control**
- Range: -1.0 V to +1.0 V
- Step: 0.01 V precision
- Real-time color updates
- Visual gradient background (blue → green → yellow)
- Large, easy-to-use thumb control

**Implementation**
```html
<input type="range" x-model.number="voltage" @input="updateVisualization()"
       min="-1.0" max="1.0" step="0.01"
       class="voltage-slider">
```

**Styling**
- Custom gradient background representing voltage states
- White thumb with purple border
- Box shadow for depth
- Smooth transitions

### 2. Color Prediction Model Integration

**Physics-Based Color Model**

The system uses a physics-informed model to predict electrochromic color changes based on:
- Applied voltage
- MXene type (Ti₃C₂Tₓ, Mo₂CTₓ, V₂CTₓ)
- Electrolyte type
- Device thickness

**Color States**

1. **Reduced State** (-1.0 to -0.5 V)
   - Color: Deep blue (#1e3a8a)
   - Transmittance: 30-50%
   - Mechanism: Electron injection, intercalation

2. **Neutral State** (-0.5 to +0.5 V)
   - Color: Gray (#6b7280)
   - Transmittance: ~50%
   - Mechanism: Balanced state

3. **Oxidized State** (+0.5 to +1.0 V)
   - Color: Yellow/Orange (#fbbf24)
   - Transmittance: 50-80%
   - Mechanism: Electron extraction, de-intercalation


**Color Prediction Algorithm**

```javascript
predictColor(voltage) {
    // Normalize voltage to 0-1 range
    const normalized = (voltage + 1) / 2;
    
    let r, g, b, transmittance, state;
    
    if (voltage < -0.5) {
        // Reduced state (blue)
        state = 'Reduced';
        r = 30 + (107 - 30) * ((voltage + 1) / 0.5);
        g = 58 + (114 - 58) * ((voltage + 1) / 0.5);
        b = 138 + (128 - 138) * ((voltage + 1) / 0.5);
        transmittance = 0.3 + 0.2 * ((voltage + 1) / 0.5);
    } else if (voltage > 0.5) {
        // Oxidized state (yellow/orange)
        state = 'Oxidized';
        const factor = (voltage - 0.5) / 0.5;
        r = 107 + (251 - 107) * factor;
        g = 114 + (191 - 114) * factor;
        b = 128 + (36 - 128) * factor;
        transmittance = 0.5 + 0.3 * factor;
    } else {
        // Neutral state (gray)
        state = 'Neutral';
        r = 107; g = 114; b = 128;
        transmittance = 0.5;
    }
    
    // Material-specific adjustments
    if (mxene_type === 'Mo2CTx') {
        r *= 0.9; g *= 1.1;
    } else if (mxene_type === 'V2CTx') {
        r *= 1.1; b *= 0.9;
    }
    
    return { rgb, hex, transmittance, state };
}
```

### 3. Real-Time Color Rendering (Canvas)

**3D Device Visualization**

The system renders a 3D perspective view of the electrochromic device using HTML5 Canvas:

**Layers (bottom to top)**
1. **Substrate**: Gray base layer
2. **MXene Film**: Active layer with current color
3. **Electrolyte**: Semi-transparent top layer

**Rendering Features**
- 3D perspective with depth
- Dynamic shadows
- Shine/reflection effects
- Real-time color updates
- Transmittance indicator bar

**Canvas Implementation**

```javascript
renderDevice() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, '#f9fafb');
    gradient.addColorStop(1, '#e5e7eb');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw device layers with 3D perspective
    this.drawLayer(ctx, centerX, centerY + 40, width, height, 20); // Substrate
    this.drawLayer(ctx, centerX, centerY, width, height, 15);      // MXene
    this.drawLayer(ctx, centerX, centerY - 40, width, height, 10); // Electrolyte
    
    // Add voltage and state labels
    ctx.fillText(`${voltage.toFixed(2)} V`, centerX, 50);
    ctx.fillText(state, centerX, 80);
    
    // Draw transmittance bar
    ctx.fillRect(barX, barY, barWidth * transmittance, barHeight);
}
```

**3D Layer Drawing**

```javascript
drawLayer(ctx, x, y, width, height, depth) {
    // Top face
    ctx.moveTo(x - width/2, y - height/2);
    ctx.lineTo(x + width/2, y - height/2);
    ctx.lineTo(x + width/2 + depth, y - height/2 - depth);
    ctx.lineTo(x - width/2 + depth, y - height/2 - depth);
    
    // Front face (darker)
    // Right face (darkest)
}
```


### 4. CV/GCD Curve Overlay

**Cyclic Voltammetry (CV) Curve**

Real-time CV curve showing:
- Forward and reverse voltage sweeps
- Capacitive current (rectangular component)
- Faradaic current (redox peaks)
- Current voltage state marker

**CV Model**

```javascript
calculateCurrent(voltage, direction) {
    const scanRate = 50; // mV/s
    const capacitance = 300; // mF/cm²
    
    // Capacitive current
    const capacitiveCurrent = capacitance * scanRate / 1000;
    
    // Faradaic peaks (Gaussian)
    const peak1 = peakHeight * exp(-((v - peakV1) / width)²);
    const peak2 = -peakHeight * exp(-((v - peakV2) / width)²);
    
    return directionFactor * capacitiveCurrent + peak1 + peak2;
}
```

**CV Features**
- Hysteresis loop visualization
- Redox peak identification
- Real-time voltage marker
- Interactive Plotly chart
- Zero-line reference

**Galvanostatic Charge-Discharge (GCD) Curve**

Shows constant current charge/discharge behavior:
- Linear voltage change during charge
- Linear voltage change during discharge
- Symmetric triangular waveform
- Time-based x-axis

**GCD Model**

```javascript
// Charge phase (0 to chargeDuration)
voltage = -1.0 + (2.0 * t / chargeDuration);

// Discharge phase
voltage = 1.0 - (2.0 * t / dischargeDuration);
```

**Toggle Between CV and GCD**
- Button controls to switch views
- Maintains same chart area
- Smooth transitions

### 5. Animation (Voltage Sweep)

**Automated Voltage Sweep**

Features:
- Configurable voltage range (min/max)
- Adjustable sweep duration (1-10 seconds)
- Triangle wave pattern (forward + reverse)
- Real-time color updates
- Start/Stop control

**Sweep Implementation**

```javascript
startSweep() {
    this.isSweeping = true;
    const startTime = Date.now();
    const duration = this.sweepDuration * 1000;
    const range = this.sweepMax - this.sweepMin;
    
    this.sweepInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = (elapsed % duration) / duration;
        
        // Triangle wave
        let sweepProgress;
        if (progress < 0.5) {
            sweepProgress = progress * 2; // Forward
        } else {
            sweepProgress = 2 - progress * 2; // Reverse
        }
        
        this.voltage = this.sweepMin + range * sweepProgress;
        this.updateVisualization();
    }, 50); // 20 FPS
}
```

**Sweep Controls**
- Min voltage input (-1.0 to 1.0 V)
- Max voltage input (-1.0 to 1.0 V)
- Duration slider (1-10 seconds)
- Play/Pause button
- Automatic loop


## Additional Features

### Voltage & Color History

**Real-Time History Tracking**
- Records voltage over time
- Tracks RGB color values
- Dual y-axis chart (voltage + RGB)
- Rolling window (last 100 points)

**History Chart**

```javascript
renderHistoryChart() {
    const voltageTrace = {
        x: times,
        y: voltages,
        name: 'Voltage',
        yaxis: 'y'
    };
    
    const rTrace = { x: times, y: rValues, name: 'Red', yaxis: 'y2' };
    const gTrace = { x: times, y: gValues, name: 'Green', yaxis: 'y2' };
    const bTrace = { x: times, y: bValues, name: 'Blue', yaxis: 'y2' };
    
    const layout = {
        yaxis: { title: 'Voltage (V)', side: 'left' },
        yaxis2: { title: 'RGB Value', side: 'right', overlaying: 'y' }
    };
}
```

### Device Configuration

**Adjustable Parameters**
- MXene type selection
- Electrolyte type selection
- Thickness (affects response time)

**Material Effects**
- **Ti₃C₂Tₓ**: Baseline electrochromic behavior
- **Mo₂CTₓ**: Enhanced green component
- **V₂CTₓ**: Enhanced red component

### Color Information Display

**Real-Time Metrics**
- RGB values (0-255)
- Hex color code
- Transmittance percentage
- Current state (Reduced/Neutral/Oxidized)

### Color Preview

**Large Color Display**
- Full-width color preview box
- Smooth color transitions (0.3s)
- Reference state indicators
- Side-by-side comparison

## Technical Implementation

### Dependencies

```html
<!-- Plotly.js for charts -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

<!-- Alpine.js for reactivity -->
<script src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
```

### Performance Optimization

**Canvas Rendering**
- Efficient redraw on voltage change only
- Hardware-accelerated 2D context
- Minimal state changes

**Animation**
- 50ms update interval (20 FPS)
- RequestAnimationFrame alternative available
- Automatic cleanup on stop

**History Management**
- Rolling buffer (max 100 points)
- Efficient array operations
- Debounced chart updates


## Usage Guide

### Basic Usage

1. **Navigate to Electrochromic Page**
   ```
   http://localhost:8000/electrochromic
   ```

2. **Adjust Voltage**
   - Drag the voltage slider
   - Watch color change in real-time
   - Observe 3D device rendering update

3. **Run Voltage Sweep**
   - Set sweep range (e.g., -1.0 to 1.0 V)
   - Set duration (e.g., 3 seconds)
   - Click "Start Sweep"
   - Watch animated color transitions

4. **View Electrochemical Curves**
   - Toggle between CV and GCD views
   - Observe current voltage marker on CV
   - Analyze charge/discharge behavior

5. **Monitor History**
   - Track voltage changes over time
   - Correlate voltage with RGB values
   - Identify color transition patterns

### Advanced Usage

**Custom Voltage Profiles**
```javascript
// Implement custom voltage patterns
const customProfile = [
    { time: 0, voltage: 0 },
    { time: 1, voltage: 1.0 },
    { time: 2, voltage: -1.0 },
    { time: 3, voltage: 0 }
];
```

**Export Capabilities**
- Screenshot canvas rendering
- Export CV/GCD data
- Save color history as CSV

**Integration with Predictions**
- Use predicted device parameters
- Correlate with capacitance values
- Validate electrochromic performance

## Scientific Background

### Electrochromic Mechanism in MXenes

**Charge Storage**
- Intercalation/de-intercalation of ions
- Electron transfer at electrode interface
- Reversible redox reactions

**Color Change Mechanism**
1. **Reduction** (negative voltage)
   - Electron injection into MXene
   - Ion intercalation
   - Increased optical absorption
   - Blue/dark coloration

2. **Oxidation** (positive voltage)
   - Electron extraction from MXene
   - Ion de-intercalation
   - Decreased optical absorption
   - Yellow/light coloration

**Optical Properties**
- Transmittance: 30-80% range
- Response time: milliseconds to seconds
- Coloration efficiency: 20-60 cm²/C
- Contrast ratio: 2:1 to 10:1

### CV Curve Interpretation

**Key Features**
- **Capacitive current**: Rectangular shape
- **Redox peaks**: Faradaic reactions
- **Hysteresis**: Kinetic limitations
- **Peak separation**: Reversibility indicator

**Performance Metrics**
- Specific capacitance from CV area
- Rate capability from scan rate dependence
- Reversibility from peak symmetry

### GCD Curve Interpretation

**Key Features**
- **Linear regions**: Ideal capacitive behavior
- **IR drop**: Equivalent series resistance
- **Coulombic efficiency**: Charge/discharge ratio
- **Voltage window**: Stability range

## Future Enhancements

### Planned Features

1. **Advanced Color Models**
   - Machine learning-based prediction
   - Spectroscopic data integration
   - Multi-wavelength analysis

2. **Enhanced Rendering**
   - WebGL for 3D visualization
   - Particle effects for ion movement
   - Realistic material textures

3. **Experimental Data Integration**
   - Upload CV/GCD data
   - Compare with predictions
   - Parameter fitting

4. **Multi-Device Comparison**
   - Side-by-side visualization
   - Performance benchmarking
   - Optimization guidance

5. **Export & Sharing**
   - Video recording of sweeps
   - Interactive 3D exports
   - Presentation mode

## API Integration (Future)

### Color Prediction Endpoint

```python
@router.post("/api/v1/electrochromic/predict")
async def predict_color(
    voltage: float,
    mxene_type: str,
    electrolyte: str
) -> ColorPrediction:
    """Predict electrochromic color at given voltage."""
    pass
```

### CV Simulation Endpoint

```python
@router.post("/api/v1/electrochromic/cv")
async def simulate_cv(
    scan_rate: float,
    voltage_range: tuple[float, float],
    device_config: DeviceConfig
) -> CVData:
    """Simulate CV curve for device."""
    pass
```

## Troubleshooting

**Issue: Canvas not rendering**
- Check browser compatibility (Chrome, Firefox, Safari)
- Verify canvas element exists
- Check console for JavaScript errors

**Issue: Slow animation**
- Reduce update frequency (increase interval)
- Simplify canvas rendering
- Close other browser tabs

**Issue: Incorrect colors**
- Verify voltage range (-1.0 to 1.0 V)
- Check device configuration
- Reset to default settings

## References

1. Electrochromic behavior of MXenes
2. Cyclic voltammetry principles
3. Canvas 2D rendering optimization
4. Color theory and RGB models

