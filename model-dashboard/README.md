# Model Comparison Dashboard

Beautiful React dashboard for comparing model performance with interactive visualizations.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd model-dashboard
npm install
```

### 2. Setup Logo Images

Create a `public/logos` folder and place your logo files:

```bash
mkdir -p public/logos
```

Copy your logo files:
- `openai.png`
- `claude.png`
- `deepseek.png`
- `qwen.png`
- `gemini.png`
- `trained.png`

```bash
cp ../models_results/openai.png public/logos/
cp ../models_results/claude.png public/logos/
cp ../models_results/deepseek.png public/logos/
cp ../models_results/qwen.png public/logos/
cp ../models_results/gemini.png public/logos/
cp ../models_results/trained.png public/logos/
```

### 3. Copy JSON Data (Optional - for real data)

Copy your evaluation JSON files to `public/data`:

```bash
mkdir -p public/data
cp ../models_results/eval_results_*.json public/data/
```

### 4. Start Development Server

```bash
npm start
```

The app will open at `http://localhost:3000`

## 📊 Features

### Dashboard View
- Beautiful card-based interface
- Click any card to view visualization
- Gradient header with modern design
- Hover effects and smooth transitions

### Chart View
- Interactive bar charts with Chart.js
- Model logos displayed above bars
- Values shown on top of bars
- Professional styling matching your reference
- Back button to return to dashboard

### Visualizations
- **F1 Micro/Macro Comparison**
- **Precision & Recall Comparison**
- **AUC Comparison**
- **@k Metrics** (k=5,8,15)
- **Stratified Performance** (by code frequency and document length)
- **Cross-tabulation Heatmaps**
- **Summary Table**

## 🎨 Styling

The dashboard uses:
- Light blue (`#A8BFEA`) for AI models
- Darker blue (`#6B7FD7`) for trained models
- Gradient purple header
- Clean card-based layout
- Smooth animations and transitions

## 🔧 Customization

### Update Model Names

Edit `src/components/ChartView.js`:

```javascript
const modelNames = [
  'OpenAI GPT-5 mini',
  'Claude Haiku 4.5',
  // ... add your models
];
```

### Load Real Data

To load real data from your JSON files, update the `loadChartData()` function in `src/components/ChartView.js` to read from the public folder:

```javascript
const loadChartData = async () => {
  try {
    const response = await fetch('/data/eval_results_your_model.json');
    const data = await response.json();
    // Process and set chart data
  } catch (error) {
    console.error('Error loading data:', error);
  }
};
```

### Add New Visualizations

1. Add a new card in `src/components/Dashboard.js`:

```javascript
{
  id: 'my-custom-view',
  title: 'My Custom View',
  description: 'Description here',
  icon: '📊'
}
```

2. Handle the view in `src/components/ChartView.js` by adding data for that view ID.

## 📦 Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` folder.

### Deploy

You can deploy to:
- **Netlify**: Drag and drop the `build` folder
- **Vercel**: Connect GitHub repo
- **GitHub Pages**: Use `gh-pages` package

## 🖼️ Logo Requirements

- **Format**: PNG with transparent background (recommended)
- **Size**: 200x200px minimum
- **Quality**: High resolution for crisp display
- **Naming**: Must match the logo map in `ChartView.js`

## 🎯 Model Name Mapping

The app automatically maps folder names to display names:

| Folder Name | Display Name |
|------------|-------------|
| `openai_gpt-5-mini_*` | OpenAI GPT-5 mini |
| `openai_gpt-4o_*` | OpenAI GPT-4o |
| `anthropic_claude-haiku_*` | Claude Haiku 4.5 |
| `anthropic_claude-sonnet_*` | Claude 3.7 Sonnet |
| `deepseek_*` | DeepSeek |
| `qwen_*` | Qwen3-30B-A3B |
| `gemini_*` | Google Gemini 2.0 |
| `conv_attn_*` or `laat_*` | Trained Att ConvNet |

## 💡 Tips

1. **Responsive Design**: Works on desktop, tablet, and mobile
2. **Fast Loading**: Optimized for quick chart rendering
3. **Interactive**: Hover over bars for detailed tooltips
4. **Professional**: Publication-ready visualizations

## 🐛 Troubleshooting

### Charts not displaying
- Check browser console for errors
- Ensure Chart.js is properly installed
- Verify data format matches expected structure

### Logos not showing
- Verify logo files are in `public/logos/`
- Check file names match exactly
- Ensure images are not too large (< 1MB recommended)

### Data not loading
- Check JSON files are in correct location
- Verify JSON format is valid
- Check browser console for fetch errors

## 📚 Technologies Used

- **React 18** - UI framework
- **Chart.js 4** - Charting library
- **react-chartjs-2** - React wrapper for Chart.js
- **chartjs-plugin-datalabels** - Data labels on charts

## 🎉 Enjoy!

Your beautiful model comparison dashboard is ready to use. Click around, explore the visualizations, and share with your team!
