# MXMAP-X Web Interface

Modern, dynamic web interface built with Jinja2, Alpine.js, and TailwindCSS.

## 🎨 Features

- **Single-Page Application**: Fast, responsive interface
- **Real-time Predictions**: Instant feedback with loading states
- **Interactive Visualizations**: Chart.js for data visualization
- **Modern UI**: TailwindCSS for beautiful, responsive design
- **Reactive Components**: Alpine.js for dynamic interactions
- **No Build Step**: Pure HTML/CSS/JS served by FastAPI

## 📁 Structure

```
app/
├── templates/
│   ├── base.html           # Base template with navigation
│   ├── index.html          # Main prediction interface
│   ├── optimize.html       # Multi-objective optimization
│   └── explore.html        # Chemistry space exploration
└── web/
    ├── __init__.py
    └── routes.py           # Web interface routes
```

## 🚀 Pages

### 1. Prediction Interface (`/`)

**Features:**
- Material composition form
- Real-time validation
- Collapsible advanced parameters
- Live prediction results
- Confidence visualization
- Example data loader

**Usage:**
1. Select MXene type, terminations, electrolyte
2. Enter thickness and deposition method
3. Optionally add advanced parameters
4. Click "Predict Performance"
5. View results with uncertainty intervals

### 2. Optimization Interface (`/optimize`)

**Features:**
- Multi-objective configuration
- Dynamic objective management
- Algorithm parameter tuning
- Pareto-optimal solutions display
- Solution comparison

**Usage:**
1. Add optimization objectives
2. Set targets (maximize/minimize)
3. Configure weights and constraints
4. Run optimization
5. Browse Pareto-optimal solutions

### 3. Exploration Interface (`/explore`)

**Features:**
- UMAP dimensionality reduction
- Interactive scatter plot
- Color-coded by performance
- Device selection and details
- Statistics dashboard

**Usage:**
1. Set number of samples
2. Choose color metric
3. Generate chemistry map
4. Click points to see details
5. Discover material relationships

## 🛠️ Technology Stack

### Frontend
- **Alpine.js 3.x**: Reactive components
- **TailwindCSS 3.x**: Utility-first CSS
- **Chart.js 4.x**: Data visualization
- **Axios 1.x**: HTTP client

### Backend
- **FastAPI**: Web server
- **Jinja2**: Template engine
- **Python 3.10+**: Backend logic

## 🎯 Key Components

### Base Template (`base.html`)

**Features:**
- Gradient navigation bar
- Health status indicator
- Toast notifications
- Global Alpine.js state
- Responsive layout

**Global Functions:**
```javascript
appData() {
  health: {},           // API health status
  toast: {},            // Toast notifications
  checkHealth(),        // Check API health
  showToast(),          // Show notification
  apiCall()             // Make API request
}
```

### Prediction Page (`index.html`)

**Alpine.js Component:**
```javascript
predictionData() {
  form: {},             // Form data
  result: null,         // Prediction result
  loading: false,       // Loading state
  showAdvanced: false,  // Advanced params toggle
  predict(),            // Make prediction
  loadExample()         // Load example data
}
```

### Optimization Page (`optimize.html`)

**Alpine.js Component:**
```javascript
optimizeData() {
  objectives: [],       // Optimization objectives
  results: null,        // Pareto solutions
  selectedSolution: null, // Selected solution
  addObjective(),       // Add objective
  removeObjective(),    // Remove objective
  optimize()            // Run optimization
}
```

### Exploration Page (`explore.html`)

**Alpine.js Component:**
```javascript
exploreData() {
  mapData: null,        // Chemistry map data
  chart: null,          // Chart.js instance
  selectedPoint: null,  // Selected device
  colorBy: 'capacitance', // Color metric
  loadData(),           // Load map data
  renderChart(),        // Render visualization
  updateVisualization() // Update colors
}
```

## 🎨 Styling

### Color Scheme

```css
Primary: #667eea (Purple)
Secondary: #764ba2 (Dark Purple)
Success: #10b981 (Green)
Warning: #f59e0b (Yellow)
Error: #ef4444 (Red)
```

### Confidence Levels

- **High**: Green badge, >90% confidence
- **Medium**: Yellow badge, 70-90% confidence
- **Low**: Red badge, <70% confidence

### Animations

```css
.fade-in {
  animation: fadeIn 0.3s ease-out;
}

.gradient-bg {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

## 📱 Responsive Design

### Breakpoints

- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

### Grid Layouts

```html
<!-- Prediction Page -->
<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <div class="lg:col-span-2"><!-- Form --></div>
  <div class="lg:col-span-1"><!-- Results --></div>
</div>

<!-- Optimization Page -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
  <div><!-- Config --></div>
  <div><!-- Results --></div>
</div>

<!-- Exploration Page -->
<div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
  <div class="lg:col-span-1"><!-- Controls --></div>
  <div class="lg:col-span-3"><!-- Visualization --></div>
</div>
```

## 🔧 Customization

### Adding New Pages

1. Create template in `app/templates/`:
```html
{% extends "base.html" %}
{% block content %}
  <!-- Your content -->
{% endblock %}
```

2. Add route in `app/web/routes.py`:
```python
@router.get("/mypage", response_class=HTMLResponse)
async def mypage(request: Request):
    return templates.TemplateResponse("mypage.html", {"request": request})
```

3. Add navigation link in `base.html`:
```html
<a href="/mypage" class="text-white hover:text-gray-200">My Page</a>
```

### Custom Alpine.js Components

```html
<div x-data="myComponent()">
  <!-- Component template -->
</div>

<script>
function myComponent() {
  return {
    // Component state
    data: null,
    
    // Component methods
    async loadData() {
      this.data = await this.apiCall('GET', '/endpoint');
    }
  }
}
</script>
```

### Custom Styling

Add to `<style>` block in `base.html`:
```css
.my-custom-class {
  /* Custom styles */
}
```

Or use Tailwind utility classes:
```html
<div class="bg-blue-500 text-white p-4 rounded-lg shadow-md">
  Custom styled element
</div>
```

## 🚀 Deployment

### Development

```bash
# Start FastAPI server
uvicorn app.main:app --reload

# Access web interface
open http://localhost:8000
```

### Production

```bash
# Start with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Or with Docker
docker-compose up -d
```

### Environment Variables

```bash
# .env
DEBUG=false
API_V1_PREFIX=/api/v1
BACKEND_CORS_ORIGINS=["https://yourdomain.com"]
```

## 📊 Performance

### Optimization

- **CDN Resources**: TailwindCSS, Alpine.js, Chart.js from CDN
- **Lazy Loading**: Charts rendered on demand
- **Caching**: Browser caching for static assets
- **Compression**: Gzip compression enabled

### Best Practices

1. **Minimize API Calls**: Cache results when possible
2. **Debounce Inputs**: Prevent excessive API calls
3. **Loading States**: Show feedback during operations
4. **Error Handling**: Graceful error messages
5. **Responsive Design**: Mobile-first approach

## 🐛 Troubleshooting

### Common Issues

**Issue**: Alpine.js not working
```html
<!-- Ensure Alpine.js is loaded -->
<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
```

**Issue**: Chart not rendering
```javascript
// Wait for DOM to be ready
this.$nextTick(() => this.renderChart());
```

**Issue**: API calls failing
```javascript
// Check CORS settings in main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📚 Resources

- **Alpine.js**: https://alpinejs.dev/
- **TailwindCSS**: https://tailwindcss.com/
- **Chart.js**: https://www.chartjs.org/
- **Jinja2**: https://jinja.palletsprojects.com/
- **FastAPI Templates**: https://fastapi.tiangolo.com/advanced/templates/

## 🎓 Examples

### Making API Calls

```javascript
// GET request
const data = await this.apiCall('GET', '/devices?page=1');

// POST request
const result = await this.apiCall('POST', '/predict', formData);

// With error handling
try {
  const data = await this.apiCall('GET', '/endpoint');
  this.showToast('Success!', 'success');
} catch (error) {
  // Error already shown by apiCall
  console.error(error);
}
```

### Showing Notifications

```javascript
// Success notification
this.showToast('Operation completed!', 'success');

// Error notification
this.showToast('Something went wrong', 'error');
```

### Form Validation

```html
<form @submit.prevent="handleSubmit()">
  <input type="number" 
         x-model="value" 
         required 
         min="0" 
         max="100"
         class="w-full px-3 py-2 border rounded-md">
  <button type="submit">Submit</button>
</form>
```

---

**Last Updated**: 2024-01-15  
**Version**: 1.0.0  
**Status**: Production Ready ✅
