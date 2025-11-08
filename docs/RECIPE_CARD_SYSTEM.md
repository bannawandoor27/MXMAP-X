# Recipe Card System Documentation

## Overview

The Recipe Card System provides a comprehensive solution for generating, managing, and sharing fabrication recipes for MXene supercapacitors. It includes print-friendly cards, PDF export, favorites management, and batch download capabilities.

## Features

### 1. Recipe Card Component

**Print-Friendly Design**
- Optimized A4 layout (210mm width)
- Clean, professional styling
- Automatic page breaks for printing
- Hides navigation and action buttons when printing

**Card Sections**
- **Header**: Recipe ID, device type, favorite button
- **Device Composition**: Material specifications
- **Processing Steps**: Step-by-step fabrication instructions
- **Predicted Performance**: Expected device metrics
- **Materials List**: Required materials with quantities
- **Safety Notes**: Important safety considerations
- **Footer**: Estimated time and generation date

### 2. PDF Export (jsPDF)

**Single Recipe Export**
- Click "Export PDF" button on any recipe card
- Uses html2canvas to capture the visual layout
- Generates high-quality PDF (A4 format)
- Automatic filename: `MXMAP-XXXXXX.pdf`

**Technical Implementation**
```javascript
async exportToPDF() {
    const { jsPDF } = window.jspdf;
    const element = document.getElementById('recipeCardContent');
    
    const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        logging: false
    });
    
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    
    const imgWidth = 210;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    
    pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
    pdf.save(`${this.currentRecipe.recipe_id}.pdf`);
}
```


### 3. Bookmarking/Favorites System

**Features**
- Add/remove recipes from favorites with star button
- Persistent storage using localStorage
- Favorites counter in header
- Toggle view between current recipe and favorites list
- Click any favorite to load it

**Data Storage**
```javascript
// Save to localStorage
localStorage.setItem('mxmap_favorites', JSON.stringify(favorites));

// Load from localStorage
const saved = localStorage.getItem('mxmap_favorites');
if (saved) {
    this.favorites = JSON.parse(saved);
}
```

**Favorite Card Display**
- Recipe ID and composition summary
- Estimated time and difficulty level
- Quick access to full recipe
- Remove from favorites button

### 4. Share Link Generation

**Functionality**
- Generate shareable URL with recipe ID parameter
- Copy link to clipboard automatically
- Fallback to prompt dialog if clipboard API unavailable
- URL format: `https://your-domain/recipes?id=MXMAP-XXXXXX`

**Implementation**
```javascript
shareRecipe() {
    const baseUrl = window.location.origin + window.location.pathname;
    const shareUrl = `${baseUrl}?id=${this.currentRecipe.recipe_id}`;
    
    navigator.clipboard.writeText(shareUrl).then(() => {
        this.showToast('Share link copied to clipboard!', 'success');
    }).catch(() => {
        prompt('Share this link:', shareUrl);
    });
}
```

**URL Parameter Handling**
- Automatically loads recipe if `?id=` parameter present
- Checks favorites first, then API (if implemented)
- Shows error toast if recipe not found

### 5. Batch Download (ZIP)

**Features**
- Download all favorite recipes as a single ZIP file
- Each recipe exported as individual PDF
- Automatic filename: `mxmap-recipes-YYYY-MM-DD.zip`
- Progress indication during generation

**Technical Implementation**
```javascript
async batchDownload() {
    const zip = new JSZip();
    const { jsPDF } = window.jspdf;
    
    // Generate PDF for each favorite
    for (let i = 0; i < this.favorites.length; i++) {
        const recipe = this.favorites[i];
        
        // Temporarily load recipe
        this.currentRecipe = recipe;
        await this.$nextTick();
        
        // Generate PDF
        const element = document.getElementById('recipeCardContent');
        const canvas = await html2canvas(element, { scale: 2 });
        const imgData = canvas.toDataURL('image/png');
        
        const pdf = new jsPDF('p', 'mm', 'a4');
        const imgWidth = 210;
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
        
        // Add to ZIP
        const pdfBlob = pdf.output('blob');
        zip.file(`${recipe.recipe_id}.pdf`, pdfBlob);
    }
    
    // Download ZIP
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    saveAs(zipBlob, `mxmap-recipes-${new Date().toISOString().split('T')[0]}.zip`);
}
```


## Recipe Generation

### From Current Prediction

1. Make a prediction on the main page
2. Click "Create Recipe Card" button
3. Automatically redirects to recipe page
4. Recipe generated from session storage data

### From Templates

**Available Templates**
- **High Capacitance**: Optimized for maximum capacitance
- **Long Cycle Life**: Optimized for durability
- **Low ESR**: Optimized for low resistance

**Template Configurations**
```javascript
const templates = {
    high_capacitance: {
        mxene_type: 'Ti3C2Tx',
        terminations: 'O',
        electrolyte: 'H2SO4',
        thickness_um: 10.0,
        deposition_method: 'vacuum_filtration',
        electrolyte_concentration: 1.0,
        annealing_temp_c: 120,
        annealing_time_min: 60
    },
    long_cycle_life: {
        mxene_type: 'Ti3C2Tx',
        terminations: 'F',
        electrolyte: 'ionic_liquid',
        thickness_um: 5.0,
        deposition_method: 'spray_coating',
        annealing_temp_c: 150,
        annealing_time_min: 90
    },
    low_esr: {
        mxene_type: 'Ti3C2Tx',
        terminations: 'O',
        electrolyte: 'KOH',
        thickness_um: 3.0,
        deposition_method: 'spray_coating',
        electrolyte_concentration: 2.0
    }
};
```

## Processing Steps Generation

The system automatically generates detailed processing steps based on device configuration:

### Step 1: MXene Preparation
- Duration: 2-4 hours
- Temperature: Room temperature
- Equipment: Sonicator, Centrifuge, Vacuum filtration setup

### Step 2: Electrolyte Preparation (if applicable)
- Duration: 30 minutes
- Temperature: Room temperature
- Equipment: Volumetric flask, Magnetic stirrer

### Step 3: Film Deposition
- Duration: Varies by method (30 min - 4 hours)
- Temperature: Room temperature
- Equipment: Method-specific

**Deposition Methods**
- **Vacuum Filtration**: 1-2 hours, requires vacuum pump
- **Spray Coating**: 30-60 minutes, requires spray gun
- **Drop Casting**: 2-4 hours, requires drying oven

### Step 4: Thermal Annealing (if applicable)
- Duration: User-specified
- Temperature: User-specified
- Equipment: Tube furnace, Inert gas supply

### Step 5: Device Assembly
- Duration: 1-2 hours
- Temperature: Room temperature
- Equipment: Current collectors, Separator, Cell housing

## Materials List

Automatically generated based on device composition:

1. **MXene Material**: ~100 mg, Research grade
2. **Electrolyte**: 50-100 mL, ACS grade
3. **Deionized Water**: 500 mL, 18.2 MΩ·cm
4. **Current Collectors**: 2 pieces
5. **Separator Membrane**: 1 piece

## Safety Notes

Automatically generated based on device configuration:

**Standard Notes**
- Wear appropriate PPE (lab coat, gloves, safety glasses)
- Handle electrolyte with care - corrosive material
- Work in well-ventilated area or fume hood
- Follow institutional chemical safety protocols

**Conditional Notes**
- High temperature warning (if annealing > 150°C)
- Specific electrolyte hazards
- Equipment-specific safety considerations

