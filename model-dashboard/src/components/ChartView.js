import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import './ChartView.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartDataLabels
);

const ChartView = ({ view, modelsData, onBack }) => {
  const [chartData, setChartData] = useState(null);
  const [logoImages, setLogoImages] = useState({});
  const [modelsConfig, setModelsConfig] = useState(null);
  const [modelNames, setModelNames] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [logoMap, setLogoMap] = useState({});
  const [fileMap, setFileMap] = useState({});
  const [configLoaded, setConfigLoaded] = useState(false);

  useEffect(() => {
    loadModelsConfig();
  }, []);

  useEffect(() => {
    if (configLoaded) {
      loadLogos();
    }
  }, [configLoaded]);

  useEffect(() => {
    if (configLoaded) {
      loadChartData();
    }
  }, [view, configLoaded, selectedModels]);

  const loadModelsConfig = async () => {
    try {
      const response = await fetch('/data/900/models_config.json');
      const config = await response.json();
      setModelsConfig(config);
      
      // Build modelNames, logoMap, and fileMap from config
      const names = config.models.map(m => m.name);
      const logos = {};
      const files = {};
      
      config.models.forEach(model => {
        logos[model.name] = model.logo;
        files[model.name] = `/data/900/${model.file}`;
      });
      
      setModelNames(names);
      setSelectedModels(names); // Select all models by default
      setLogoMap(logos);
      setFileMap(files);
      setConfigLoaded(true);
      
      console.log(`✅ Loaded ${names.length} models:`, names);
    } catch (error) {
      console.error('Error loading models config:', error);
      // Fallback to default models
      const defaultNames = ['Trained ConvNet', 'Gemini 2.0 Flash', 'Qwen 30B'];
      const defaultLogos = {
        'Trained ConvNet': '/logos/trained.png',
        'Gemini 2.0 Flash': '/logos/gemini.png',
        'Qwen 30B': '/logos/qwen.png'
      };
      const defaultFiles = {
        'Trained ConvNet': '/data/900/eval_results_trained_convnet.json',
        'Gemini 2.0 Flash': '/data/900/eval_results_gemini_2_0_flash.json',
        'Qwen 30B': '/data/900/eval_results_qwen_30b.json'
      };
      setModelNames(defaultNames);
      setLogoMap(defaultLogos);
      setFileMap(defaultFiles);
      setConfigLoaded(true);
    }
  };

  const loadLogos = () => {
    const images = {};
    modelNames.forEach((name) => {
      const img = new Image();
      img.src = logoMap[name];
      img.onload = () => {
        images[name] = img;
        setLogoImages({...images});
      };
      img.onerror = () => {
        // Use a default logo if the specific one fails to load
        console.warn(`Logo not found for ${name}, using default`);
      };
    });
  };

  const loadChartData = async () => {
    try {
      console.log('Loading data for view:', view.id);
      // Load real data from JSON files
      const realData = await loadRealMetrics();
      console.log('Loaded data:', realData);
      setChartData(realData);
    } catch (error) {
      console.error('Error loading data:', error);
      // Fallback to sample data if loading fails
      const sampleData = generateSampleData();
      setChartData(sampleData);
    }
  };

  const loadRealMetrics = async () => {
    // Map model names to their JSON files
    // Only load data for selected models
    const values = [];
    
    for (const modelName of selectedModels) {
      const filePath = fileMap[modelName];
      if (filePath) {
        try {
          const response = await fetch(filePath);
          const data = await response.json();
          const overall = data.overall || {};
          
          // Extract the metric based on view.id
          let value = 0;
          
          // Handle stratified metrics
          if (view.id === 'stratified-frequency') {
            // Average across common, medium, rare
            const stratified = data.stratified || {};
            const byFreq = stratified.code_freq || {};
            const common = (byFreq.common?.f1_micro || 0) * 100;
            const medium = (byFreq.medium?.f1_micro || 0) * 100;
            const rare = (byFreq.rare?.f1_micro || 0) * 100;
            value = (common + medium + rare) / 3;
            console.log(`${modelName} frequency stratified: common=${common.toFixed(2)}, medium=${medium.toFixed(2)}, rare=${rare.toFixed(2)}, avg=${value.toFixed(2)}`);
          } else if (view.id === 'stratified-length') {
            // Average across short, medium, long
            const stratified = data.stratified || {};
            const byLength = stratified.length || {};
            const short = (byLength.short?.f1_micro || 0) * 100;
            const mediumLen = (byLength.medium?.f1_micro || 0) * 100;
            const long = (byLength.long?.f1_micro || 0) * 100;
            value = (short + mediumLen + long) / 3;
            console.log(`${modelName} length stratified: short=${short.toFixed(2)}, medium=${mediumLen.toFixed(2)}, long=${long.toFixed(2)}, avg=${value.toFixed(2)}`);
          } else if (view.id === 'stratified-comorbidity') {
            // Average across low, medium, high comorbidity
            const stratified = data.stratified || {};
            const byComorb = stratified.comorbidity || {};
            const low = (byComorb.low?.f1_micro || 0) * 100;
            const mediumComorb = (byComorb.medium?.f1_micro || 0) * 100;
            const high = (byComorb.high?.f1_micro || 0) * 100;
            value = (low + mediumComorb + high) / 3;
            console.log(`${modelName} comorbidity stratified: low=${low.toFixed(2)}, medium=${mediumComorb.toFixed(2)}, high=${high.toFixed(2)}, avg=${value.toFixed(2)}`);
          } else if (view.id === 'stratified-race') {
            // Average across racial groups
            const stratified = data.stratified || {};
            const byRace = stratified.race || {};
            const groups = ['White', 'Black', 'Asian', 'Hispanic', 'Other'];
            let sum = 0;
            let count = 0;
            groups.forEach(group => {
              if (byRace[group]?.f1_micro) {
                sum += (byRace[group].f1_micro * 100);
                count++;
              }
            });
            value = count > 0 ? sum / count : 0;
            console.log(`${modelName} race stratified: avg=${value.toFixed(2)}`);
          } else if (view.id === 'calibration') {
            // Expected Calibration Error
            const calibration = data.calibration || {};
            value = (calibration.ece || 0) * 100;
            console.log(`${modelName} ECE: ${value.toFixed(2)}`);
          } else if (view.id === 'f1-micro') {
            value = (overall.f1_micro || 0) * 100;
          } else if (view.id === 'f1-macro') {
            value = (overall.f1_macro || 0) * 100;
          } else if (view.id === 'precision-micro') {
            value = (overall.prec_micro || 0) * 100;
          } else if (view.id === 'recall-micro') {
            value = (overall.rec_micro || 0) * 100;
          } else if (view.id === 'auc-micro') {
            value = (overall.auc_micro || 0) * 100;
          } else if (view.id === 'precision-at-5') {
            value = (overall.prec_at_5 || 0) * 100;
          } else if (view.id === 'recall-at-5') {
            value = (overall.rec_at_5 || 0) * 100;
          } else if (view.id === 'f1-at-5') {
            value = (overall.f1_at_5 || 0) * 100;
          }
          
          values.push(value);
        } catch (err) {
          console.error(`Error loading ${modelName}:`, err);
          values.push(0);
        }
      } else {
        values.push(0);
      }
    }

    // Sort by performance (descending)
    // Use selectedModels instead of modelNames to match the values array
    const combined = selectedModels.map((name, i) => ({ name, value: values[i] }));
    combined.sort((a, b) => b.value - a.value);
    
    const sortedLabels = combined.map(item => item.name);
    const sortedValues = combined.map(item => item.value);
    
    // Color code by performance tiers - blue/purple palette with darker top performers
    const getColor = (value, name, rank) => {
      if (name.includes('Trained')) {
        // AI Models - Blue shades (darker for top performers)
        if (rank === 0) return '#5B7FB8'; // Best - darkest blue
        if (rank === 1) return '#6B8FC8'; // Second - dark blue
        return value >= 70 ? '#7B9FD4' : value >= 50 ? '#93B3E0' : '#A8C8E8'; // Others - lighter blues
      }
      // Human Readers - Purple shades (darker for top performers)
      if (rank === 0) return '#6B5FA0'; // Best - darkest purple
      if (rank === 1) return '#7B6FB0'; // Second - dark purple
      return value >= 70 ? '#8B7FB8' : value >= 50 ? '#9B8FC7' : '#AB9FD4'; // Others - lighter purples
    };
    
    const convnetValue = combined.find(item => item.name.includes('Trained'))?.value || 0;

    return {
      labels: sortedLabels,
      datasets: [{
        label: view.title,
        data: sortedValues,
        backgroundColor: sortedLabels.map((name, i) => getColor(sortedValues[i], name, i)),
        borderColor: sortedLabels.map((name, i) => 
          i === 0 ? '#FFD700' : '#ffffff' // Gold border for best performer
        ),
        borderWidth: sortedLabels.map((name, i) => i === 0 ? 3 : 2),
        borderRadius: 4
      }],
      convnetBaseline: convnetValue // Store for threshold line
    };
  };

  const generateSampleData = () => {
    // Fallback sample data
    const values = modelNames.map(() => Math.random() * 50);
    return {
      labels: modelNames,
      datasets: [{
        label: view.title,
        data: values,
        backgroundColor: modelNames.map(name => 
          name.includes('Trained') ? '#6B7FD7' : '#A8BFEA'
        ),
        borderColor: '#ffffff',
        borderWidth: 2,
        borderRadius: 4
      }]
    };
  };

  // Custom plugin to draw logos at bottom next to model names
  const logoPlugin = {
    id: 'logoPlugin',
    afterDraw: (chart) => {
      const ctx = chart.ctx;
      const meta = chart.getDatasetMeta(0);
      const xAxis = chart.scales.x;
      
      if (!meta || !meta.data) return;
      
      meta.data.forEach((bar, index) => {
        const label = chart.data.labels[index];
        const logo = logoImages[label];
        
        if (!logo || !bar) return;
        
        // Get the x position of the bar
        const x = bar.x;
        // Position at the bottom, below the chart
        const y = chart.chartArea.bottom + 70; // Below x-axis labels
        
        const logoSize = 35;
        
        ctx.save();
        ctx.drawImage(
          logo,
          x - logoSize / 2,
          y - logoSize / 2,
          logoSize,
          logoSize
        );
        ctx.restore();
      });
    }
  };

  // Threshold line plugin
  const thresholdPlugin = {
    id: 'thresholdLines',
    afterDatasetsDraw: (chart) => {
      const ctx = chart.ctx;
      const yAxis = chart.scales.y;
      const xAxis = chart.scales.x;
      
      ctx.save();
      
      // Draw ConvNet baseline if available
      if (chartData?.convnetBaseline && chartData.convnetBaseline > 0) {
        const yConvnet = yAxis.getPixelForValue(chartData.convnetBaseline);
        ctx.strokeStyle = '#7B9FD4';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(xAxis.left, yConvnet);
        ctx.lineTo(xAxis.right, yConvnet);
        ctx.stroke();
        
        ctx.fillStyle = '#7B9FD4';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`ConvNet Baseline (${chartData.convnetBaseline.toFixed(1)}%)`, xAxis.right - 5, yConvnet - 5);
      }
      
      ctx.restore();
    }
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: {
        top: 20,
        bottom: 80 // Make room for logos at bottom
      }
    },
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: false // Hide title, we'll use the header instead
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        titleFont: {
          size: 14,
          weight: 'bold'
        },
        bodyFont: {
          size: 13
        },
        callbacks: {
          label: function(context) {
            const rank = context.dataIndex + 1;
            return [
              `#${rank} ${context.label}`,
              `Score: ${context.parsed.y.toFixed(2)}%`
            ];
          }
        }
      },
      datalabels: {
        anchor: 'end',
        align: 'top',
        offset: 4,
        font: {
          size: 13,
          weight: 'bold'
        },
        formatter: (value, context) => {
          if (value === undefined || value === null) return '';
          const rank = context.dataIndex + 1;
          return `#${rank}\n${value.toFixed(1)}%`;
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          font: {
            size: 12
          },
          callback: function(value) {
            return value + '%';
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        ticks: {
          font: {
            size: 11
          },
          maxRotation: 45,
          minRotation: 45
        },
        grid: {
          display: false
        }
      }
    }
  };

  // Model selection toggle functions
  const toggleModel = (modelName) => {
    setSelectedModels(prev => {
      if (prev.includes(modelName)) {
        // Deselect (but keep at least one selected)
        return prev.length > 1 ? prev.filter(m => m !== modelName) : prev;
      } else {
        // Select
        return [...prev, modelName];
      }
    });
  };

  const selectAll = () => {
    setSelectedModels(modelNames);
  };

  const selectNone = () => {
    // Keep at least one model selected
    if (modelNames.length > 0) {
      setSelectedModels([modelNames[0]]);
    }
  };

  // Show loading state while models config is loading
  if (!configLoaded || modelNames.length === 0) {
    return (
      <div className="chart-view">
        <div className="chart-header">
          <button className="back-button" onClick={onBack}>
            ← Back to Dashboard
          </button>
          <h2>{view.title}</h2>
        </div>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '400px',
          flexDirection: 'column'
        }}>
          <div className="spinner"></div>
          <p>Loading models configuration...</p>
        </div>
      </div>
    );
  }

  // Render stratified views with special components
  if (view.id === 'stratified-frequency') {
    return <StratifiedView view={view} onBack={onBack} modelNames={selectedModels} fileMap={fileMap} stratifyBy="code_freq" allModels={modelNames} selectedModels={selectedModels} toggleModel={toggleModel} selectAll={selectAll} selectNone={selectNone} />;
  }
  
  if (view.id === 'stratified-length') {
    return <StratifiedView view={view} onBack={onBack} modelNames={selectedModels} fileMap={fileMap} stratifyBy="length" allModels={modelNames} selectedModels={selectedModels} toggleModel={toggleModel} selectAll={selectAll} selectNone={selectNone} />;
  }
  
  if (view.id === 'stratified-comorbidity') {
    return <StratifiedView view={view} onBack={onBack} modelNames={selectedModels} fileMap={fileMap} stratifyBy="comorbidity" allModels={modelNames} selectedModels={selectedModels} toggleModel={toggleModel} selectAll={selectAll} selectNone={selectNone} />;
  }
  
  if (view.id === 'stratified-race') {
    return <StratifiedView view={view} onBack={onBack} modelNames={selectedModels} fileMap={fileMap} stratifyBy="race" allModels={modelNames} selectedModels={selectedModels} toggleModel={toggleModel} selectAll={selectAll} selectNone={selectNone} />;
  }

  // Render calibration view
  if (view.id === 'calibration') {
    return <CalibrationView view={view} onBack={onBack} modelNames={selectedModels} fileMap={fileMap} allModels={modelNames} selectedModels={selectedModels} toggleModel={toggleModel} selectAll={selectAll} selectNone={selectNone} />;
  }

  // Render performance heatmap
  if (view.id === 'performance-heatmap') {
    return <PerformanceHeatmap view={view} onBack={onBack} modelNames={selectedModels} fileMap={fileMap} allModels={modelNames} selectedModels={selectedModels} toggleModel={toggleModel} selectAll={selectAll} selectNone={selectNone} />;
  }

  // Render summary table
  if (view.id === 'summary-table') {
    return <SummaryTable view={view} onBack={onBack} modelNames={selectedModels} fileMap={fileMap} allModels={modelNames} selectedModels={selectedModels} toggleModel={toggleModel} selectAll={selectAll} selectNone={selectNone} />;
  }

  return (
    <div className="chart-view">
      <div className="chart-header">
        <button className="back-button" onClick={onBack}>
          ← Back to Dashboard
        </button>
      </div>

      <div className="chart-container">
        <div className="chart-main">
          {chartData && Object.keys(logoImages).length > 0 ? (
            <>
              <h3 className="chart-title">{view.title}</h3>
              <div className="chart-wrapper">
                <Bar data={chartData} options={options} plugins={[logoPlugin]} />
              </div>
              <div className="legend-info" style={{
                fontSize: '12px',
                marginTop: '10px',
                padding: '10px',
                backgroundColor: '#f5f5f5',
                borderRadius: '4px'
              }}>
                <strong>Color Legend:</strong>
                <span style={{marginLeft: '15px'}}>🔵 AI Models (Light Blue)</span>
                <span style={{marginLeft: '15px'}}>🟣 Human Readers (Purple)</span>
                <span style={{marginLeft: '15px'}}>🥇 Gold border = Top performer</span>
              </div>
            </>
          ) : (
            <div className="loading-chart">
              <div className="spinner"></div>
              <p>Loading chart...</p>
            </div>
          )}
        </div>

        {/* Model Selector */}
        <div className="model-selector">
          <h3>⚙️ Select Models</h3>
          <div className="model-checkbox-list">
            {modelNames.map(modelName => (
              <div key={modelName} className="model-checkbox-item">
                <input
                  type="checkbox"
                  id={`model-${modelName}`}
                  checked={selectedModels.includes(modelName)}
                  onChange={() => toggleModel(modelName)}
                />
                <label htmlFor={`model-${modelName}`}>{modelName}</label>
              </div>
            ))}
          </div>
          <div className="model-selector-actions">
            <button className="selector-btn selector-btn-all" onClick={selectAll}>
              All
            </button>
            <button className="selector-btn selector-btn-none" onClick={selectNone}>
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// CrossTabHeatmap Component
const CrossTabHeatmap = ({ view, onBack, modelNames, fileMap }) => {
  const [heatmapData, setHeatmapData] = useState(null);

  useEffect(() => {
    loadHeatmapData();
  }, []);

  const loadHeatmapData = async () => {
    const data = [];
    
    for (const modelName of modelNames) {
      const filePath = fileMap[modelName];
      if (filePath) {
        try {
          const response = await fetch(filePath);
          const json = await response.json();
          const stratified = json.stratified || {};
          const crossTab = stratified.cross_tab || {};
          
          const modelData = {
            name: modelName,
            common_short: (crossTab.common_short?.f1_micro || 0) * 100,
            common_medium: (crossTab.common_medium?.f1_micro || 0) * 100,
            common_long: (crossTab.common_long?.f1_micro || 0) * 100,
            medium_short: (crossTab.medium_short?.f1_micro || 0) * 100,
            medium_medium: (crossTab.medium_medium?.f1_micro || 0) * 100,
            medium_long: (crossTab.medium_long?.f1_micro || 0) * 100,
            rare_short: (crossTab.rare_short?.f1_micro || 0) * 100,
            rare_medium: (crossTab.rare_medium?.f1_micro || 0) * 100,
            rare_long: (crossTab.rare_long?.f1_micro || 0) * 100
          };
          
          data.push(modelData);
        } catch (err) {
          console.error(`Error loading ${modelName}:`, err);
        }
      }
    }
    
    setHeatmapData(data);
  };

  const getColorForValue = (value) => {
    if (value === 0) return '#f0f0f0';
    if (value < 10) return '#fee5d9';
    if (value < 20) return '#fcbba1';
    if (value < 30) return '#fc9272';
    if (value < 40) return '#fb6a4a';
    return '#de2d26';
  };

  if (!heatmapData) {
    return (
      <div className="chart-view">
        <div className="chart-header">
          <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
        </div>
        <div className="chart-container">
          <div className="loading-chart">
            <div className="spinner"></div>
            <p>Loading heatmap...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-view">
      <div className="chart-header">
        <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
      </div>
      <div className="chart-container">
        <h3 className="chart-title">{view.title}</h3>
        <div style={{overflowX: 'auto', padding: '20px 0'}}>
          <table style={{width: '100%', borderCollapse: 'collapse', minWidth: '800px'}}>
            <thead>
              <tr>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#f5f7fa', fontWeight: '600'}}>Model</th>
                <th colSpan="3" style={{padding: '12px', border: '1px solid #e0e0e0', background: '#667eea', color: 'white', fontWeight: '600'}}>Common Codes</th>
                <th colSpan="3" style={{padding: '12px', border: '1px solid #e0e0e0', background: '#667eea', color: 'white', fontWeight: '600'}}>Medium Codes</th>
                <th colSpan="3" style={{padding: '12px', border: '1px solid #e0e0e0', background: '#667eea', color: 'white', fontWeight: '600'}}>Rare Codes</th>
              </tr>
              <tr>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#f5f7fa'}}></th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Short</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Medium</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Long</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Short</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Medium</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Long</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Short</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Medium</th>
                <th style={{padding: '12px', border: '1px solid #e0e0e0', background: '#e6e9f5', fontSize: '13px'}}>Long</th>
              </tr>
            </thead>
            <tbody>
              {heatmapData.map((model, idx) => (
                <tr key={idx}>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', fontWeight: '500', background: '#f9fafb'}}>{model.name}</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.common_short), fontWeight: '500'}}>{model.common_short.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.common_medium), fontWeight: '500'}}>{model.common_medium.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.common_long), fontWeight: '500'}}>{model.common_long.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.medium_short), fontWeight: '500'}}>{model.medium_short.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.medium_medium), fontWeight: '500'}}>{model.medium_medium.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.medium_long), fontWeight: '500'}}>{model.medium_long.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.rare_short), fontWeight: '500'}}>{model.rare_short.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.rare_medium), fontWeight: '500'}}>{model.rare_medium.toFixed(1)}%</td>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', background: getColorForValue(model.rare_long), fontWeight: '500'}}>{model.rare_long.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{marginTop: '20px', textAlign: 'center', color: '#718096', fontSize: '14px'}}>
            <p>📊 Heatmap shows F1 Micro scores across Code Frequency (columns) × Document Length (sub-columns)</p>
            <p>🎨 Color scale: Light (0-10%) → Dark Red (40%+)</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// SummaryTable Component
const SummaryTable = ({ view, onBack, modelNames, fileMap, allModels, selectedModels, toggleModel, selectAll, selectNone }) => {
  const [tableData, setTableData] = useState(null);

  useEffect(() => {
    loadTableData();
  }, [modelNames]); // Reload when selected models change

  const loadTableData = async () => {
    const data = [];
    
    for (const modelName of modelNames) {
      const filePath = fileMap[modelName];
      if (filePath) {
        try {
          const response = await fetch(filePath);
          const json = await response.json();
          const overall = json.overall || {};
          
          data.push({
            model: modelName,
            f1_micro: (overall.f1_micro || 0) * 100,
            f1_macro: (overall.f1_macro || 0) * 100,
            prec_micro: (overall.prec_micro || 0) * 100,
            prec_macro: (overall.prec_macro || 0) * 100,
            rec_micro: (overall.rec_micro || 0) * 100,
            rec_macro: (overall.rec_macro || 0) * 100,
            auc_micro: (overall.auc_micro || 0) * 100,
            auc_macro: (overall.auc_macro || 0) * 100,
            prec_at_5: (overall.prec_at_5 || 0) * 100,
            rec_at_5: (overall.rec_at_5 || 0) * 100,
            f1_at_5: (overall.f1_at_5 || 0) * 100
          });
        } catch (err) {
          console.error(`Error loading ${modelName}:`, err);
        }
      }
    }
    
    setTableData(data);
  };

  // Find the max value for each metric column
  const getMaxValues = () => {
    if (!tableData) return {};
    
    const metrics = ['f1_micro', 'f1_macro', 'prec_micro', 'prec_macro', 'rec_micro', 'rec_macro', 'auc_micro', 'auc_macro', 'prec_at_5', 'rec_at_5', 'f1_at_5'];
    const maxValues = {};
    
    metrics.forEach(metric => {
      maxValues[metric] = Math.max(...tableData.map(row => row[metric]));
    });
    
    return maxValues;
  };

  // Check if this is the max value for the column
  const isMaxValue = (value, metric, maxValues) => {
    return Math.abs(value - maxValues[metric]) < 0.01; // Account for floating point precision
  };

  // Get cell style based on whether it's the max value
  const getCellStyle = (value, metric, maxValues, isEven) => {
    const baseStyle = {
      padding: '12px',
      border: '1px solid #e0e0e0',
      textAlign: 'center'
    };

    if (isMaxValue(value, metric, maxValues)) {
      return {
        ...baseStyle,
        background: '#d4edda',
        fontWeight: '700',
        color: '#155724',
        border: '2px solid #28a745'
      };
    }

    return baseStyle;
  };

  if (!tableData) {
    return (
      <div className="chart-view">
        <div className="chart-header">
          <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
        </div>
        <div className="chart-container">
          <div className="loading-chart">
            <div className="spinner"></div>
            <p>Loading table...</p>
          </div>
        </div>
      </div>
    );
  }

  const maxValues = getMaxValues();

  return (
    <div className="chart-view">
      <div className="chart-header">
        <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
      </div>
      <div className="chart-container">
        <div className="chart-main">
          <h3 className="chart-title">{view.title}</h3>
          <div style={{overflowX: 'auto', padding: '20px 0'}}>
            <table style={{width: '100%', borderCollapse: 'collapse', minWidth: '1000px'}}>
              <thead>
                <tr style={{background: '#667eea', color: 'white'}}>
                  <th style={{padding: '14px', textAlign: 'left', fontWeight: '600', position: 'sticky', left: 0, background: '#667eea', zIndex: 1}}>Model</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>F1 Micro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>F1 Macro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>Prec Micro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>Prec Macro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>Rec Micro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>Rec Macro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>AUC Micro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>AUC Macro</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>Prec@5</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>Rec@5</th>
                <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>F1@5</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, idx) => (
                <tr key={idx} style={{background: idx % 2 === 0 ? '#f9fafb' : 'white'}}>
                  <td style={{padding: '12px', border: '1px solid #e0e0e0', fontWeight: '500', position: 'sticky', left: 0, background: idx % 2 === 0 ? '#f9fafb' : 'white', zIndex: 1}}>{row.model}</td>
                  <td style={getCellStyle(row.f1_micro, 'f1_micro', maxValues, idx % 2 === 0)}>{row.f1_micro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.f1_macro, 'f1_macro', maxValues, idx % 2 === 0)}>{row.f1_macro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.prec_micro, 'prec_micro', maxValues, idx % 2 === 0)}>{row.prec_micro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.prec_macro, 'prec_macro', maxValues, idx % 2 === 0)}>{row.prec_macro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.rec_micro, 'rec_micro', maxValues, idx % 2 === 0)}>{row.rec_micro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.rec_macro, 'rec_macro', maxValues, idx % 2 === 0)}>{row.rec_macro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.auc_micro, 'auc_micro', maxValues, idx % 2 === 0)}>{row.auc_micro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.auc_macro, 'auc_macro', maxValues, idx % 2 === 0)}>{row.auc_macro.toFixed(2)}%</td>
                  <td style={getCellStyle(row.prec_at_5, 'prec_at_5', maxValues, idx % 2 === 0)}>{row.prec_at_5.toFixed(2)}%</td>
                  <td style={getCellStyle(row.rec_at_5, 'rec_at_5', maxValues, idx % 2 === 0)}>{row.rec_at_5.toFixed(2)}%</td>
                  <td style={getCellStyle(row.f1_at_5, 'f1_at_5', maxValues, idx % 2 === 0)}>{row.f1_at_5.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{marginTop: '20px', textAlign: 'center', color: '#718096', fontSize: '14px'}}>
            <p>📋 Comprehensive comparison of all models across key metrics</p>
            <p>🌟 <strong>Highlighted cells</strong> show the best (highest) value in each column</p>
          </div>
        </div>
        </div>

        {/* Model Selector */}
        <div className="model-selector">
          <h3>⚙️ Select Models</h3>
          <div className="model-checkbox-list">
            {allModels.map(modelName => (
              <div key={modelName} className="model-checkbox-item">
                <input
                  type="checkbox"
                  id={`summary-model-${modelName}`}
                  checked={selectedModels.includes(modelName)}
                  onChange={() => toggleModel(modelName)}
                />
                <label htmlFor={`summary-model-${modelName}`}>{modelName}</label>
              </div>
            ))}
          </div>
          <div className="model-selector-actions">
            <button className="selector-btn selector-btn-all" onClick={selectAll}>
              All
            </button>
            <button className="selector-btn selector-btn-none" onClick={selectNone}>
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Stratified View Component (Frequency, Length, Comorbidity, Race)
const StratifiedView = ({ view, onBack, modelNames, fileMap, stratifyBy, allModels, selectedModels, toggleModel, selectAll, selectNone }) => {
  const [stratData, setStratData] = useState(null);

  useEffect(() => {
    loadStratifiedData();
  }, [modelNames]); // Reload when selected models change

  const loadStratifiedData = async () => {
    const data = [];
    
    const stratifyMap = {
      'code_freq': { key: 'code_freq', groups: ['common', 'medium', 'rare'] },
      'length': { key: 'length', groups: ['short', 'medium', 'long'] },
      'comorbidity': { key: 'comorbidity', groups: ['low', 'medium', 'high'] },
      'race': { key: 'race', groups: ['White', 'Black', 'Asian', 'Hispanic', 'Other'] }
    };
    
    const config = stratifyMap[stratifyBy];
    
    for (const modelName of modelNames) {
      const filePath = fileMap[modelName];
      if (filePath) {
        try {
          const response = await fetch(filePath);
          const json = await response.json();
          const stratified = json.stratified || {};
          const byGroup = stratified[config.key] || {};
          
          const modelData = { model: modelName };
          config.groups.forEach(group => {
            modelData[group] = byGroup[group] || {};
          });
          
          data.push(modelData);
        } catch (err) {
          console.error(`Error loading ${modelName}:`, err);
        }
      }
    }
    
    // Load rare-only models (like RAG) when viewing code frequency
    if (stratifyBy === 'code_freq') {
      try {
        const configResponse = await fetch('/data/900/models_config.json');
        const configJson = await configResponse.json();
        const rareOnlyModels = configJson.rare_only_models || [];
        
        for (const model of rareOnlyModels) {
          try {
            const response = await fetch(`/data/900/${model.file}`);
            const json = await response.json();
            const stratified = json.stratified || {};
            const byGroup = stratified.code_freq || {};
            
            const modelData = { 
              model: model.name,
              common: {}, // Empty for rare-only models
              medium: {}, // Empty for rare-only models
              rare: byGroup.rare || {} // Only has rare data
            };
            
            data.push(modelData);
            console.log(`✅ Loaded rare-only model: ${model.name}`);
          } catch (err) {
            console.error(`Error loading rare-only model ${model.name}:`, err);
          }
        }
      } catch (err) {
        console.error('Error loading rare-only models:', err);
      }
    }
    
    setStratData({ data, config });
  };

  if (!stratData) {
    return (
      <div className="chart-view">
        <div className="chart-header">
          <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
        </div>
        <div className="loading-chart">
          <div className="spinner"></div>
          <p>Loading data...</p>
        </div>
      </div>
    );
  }

  // Calculate max values for highlighting
  const getMaxValuesForGroup = (group) => {
    const maxVals = {};
    const metrics = [
      'f1_micro', 'f1_macro', 
      'prec_micro', 'prec_macro',
      'rec_micro', 'rec_macro',
      'auc_micro', 'auc_macro',
      'prec_at_5', 'rec_at_5', 'f1_at_5',
      'prec_at_8', 'rec_at_8', 'f1_at_8',
      'prec_at_15', 'rec_at_15', 'f1_at_15'
    ];
    
    metrics.forEach(metric => {
      let max = -Infinity;
      stratData.data.forEach(modelData => {
        const groupData = modelData[group] || {};
        const val = groupData[metric] || 0;
        if (val > max) max = val;
      });
      maxVals[metric] = max;
    });
    
    return maxVals;
  };
  
  const getCellStyleStrat = (value, metric, maxValues, isEven) => {
    const baseStyle = {
      padding: '10px',
      border: '1px solid #e0e0e0',
      textAlign: 'center'
    };
    
    if (value && maxValues[metric] && value === maxValues[metric]) {
      return {
        ...baseStyle,
        background: '#fef3c7',
        fontWeight: '700',
        color: '#92400e',
        boxShadow: 'inset 0 0 0 2px #fbbf24'
      };
    }
    
    return baseStyle;
  };
  
  // Get insights text with expert-level analysis
  const getInsights = () => {
    const insights = {
      'code_freq': (
        <div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>📊 Why This Matters</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>ICD codes follow power-law distribution: common codes (hypertension, diabetes) appear frequently; rare codes (exotic diseases) may have {'<'}10 training examples.</p>
          </div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>🤖 Model Comparison</strong>
            <ul style={{margin: '8px 0 0 20px', lineHeight: '1.8', paddingLeft: '0'}}>
              <li><strong>ConvNet</strong>: Trained on 50K+ MIMIC notes. Excels on common codes, may struggle with rare (limited corpus).</li>
              <li><strong>Gemini (Google)</strong>: Web-scale pre-training includes medical literature. Zero-shot knowledge of rare diseases, but lacks clinical note structure.</li>
              <li><strong>Qwen (Alibaba)</strong>: Chinese medical data access. Poor performance (F1=1.6%) suggests prompt issues, not architectural advantage.</li>
            </ul>
          </div>
          <div>
            <strong style={{color: '#059669', fontSize: '16px'}}>💡 Clinical Impact</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Rare code prediction critical for billing and detecting emerging diseases. Models predicting only common codes offer limited value.</p>
          </div>
        </div>
      ),
      
      'length': (
        <div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>📊 Why This Matters</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Note length correlates with complexity: short notes (ED visits) vs long notes (ICU, complex surgery).</p>
          </div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>🏗️ Architecture Impact</strong>
            <ul style={{margin: '8px 0 0 20px', lineHeight: '1.8', paddingLeft: '0'}}>
              <li><strong>ConvNet (CNN)</strong>: Fixed receptive fields (2500 tokens). Excellent at local patterns, may miss long-range dependencies.</li>
              <li><strong>Gemini/Qwen (Transformers)</strong>: Attention handles 32K+ tokens, better for scattered information. Cost: O(n²) complexity.</li>
            </ul>
          </div>
          <div>
            <strong style={{color: '#059669', fontSize: '16px'}}>💡 Clinical Impact</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Consistent performance across lengths = more deployable. Note length varies by physician style and patient complexity.</p>
          </div>
        </div>
      ),
      
      'comorbidity': (
        <div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>📊 Why This Matters</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Comorbidity = disease interaction complexity. High comorbidity (6+ diagnoses) tests true clinical reasoning capability.</p>
          </div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>🎯 Why ConvNet Excels (P@8=68.1% on high comorbidity)</strong>
            <ul style={{margin: '8px 0 0 20px', lineHeight: '1.8', paddingLeft: '0'}}>
              <li><strong>Specialized Training</strong>: MIMIC-IV has median 8+ diagnoses. Learned real disease co-occurrence patterns.</li>
              <li><strong>Pattern Recognition</strong>: CNN filters detect comorbidity-specific language ("multiple medical problems").</li>
            </ul>
          </div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#dc2626', fontSize: '16px'}}>❌ Why LLMs Struggle</strong>
            <ul style={{margin: '8px 0 0 20px', lineHeight: '1.8', paddingLeft: '0'}}>
              <li>Web pre-training lacks dense clinical comorbidity exposure (textbooks show isolated diseases).</li>
              <li>Multi-hop reasoning fails in zero-shot without chain-of-thought prompting.</li>
            </ul>
          </div>
          <div>
            <strong style={{color: '#dc2626', fontSize: '16px'}}>⚠️ Clinical Risk</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>High comorbidity patients most vulnerable—coding errors lead to inadequate care planning.</p>
          </div>
        </div>
      ),
      
      'race': (
        <div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#dc2626', fontSize: '16px'}}>⚖️ Critical Context</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Performance disparities indicate potential bias. Healthcare AI fairness essential for equitable deployment.</p>
          </div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>📊 Data Demographics (MIMIC-IV)</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Boston hospital: ~60% White, 10% Black, {'<'}5% Asian/Hispanic. Underrepresentation → less training signal for minorities.</p>
          </div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#1e40af', fontSize: '16px'}}>🤖 Model-Specific Factors</strong>
            <ul style={{margin: '8px 0 0 20px', lineHeight: '1.8', paddingLeft: '0'}}>
              <li><strong>ConvNet</strong>: Inherits MIMIC demographics. Best on White (53.4%), declines for Hispanic (41.9%).</li>
              <li><strong>Gemini</strong>: Western medical literature bias. US/UK healthcare overrepresented in training.</li>
              <li><strong>Qwen</strong>: Chinese medical data access could help Asian patients, but catastrophic failure (F1=1.6%) negates advantage.</li>
            </ul>
          </div>
          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#7c3aed', fontSize: '16px'}}>🔬 Contributing Factors</strong>
            <ul style={{margin: '8px 0 0 20px', lineHeight: '1.8', paddingLeft: '0'}}>
              <li><strong>Documentation Bias</strong>: Physicians document differently by race (JAMA/BMJ evidence).</li>
              <li><strong>Disease Prevalence</strong>: Genetic factors (e.g., sickle cell in Black, Tay-Sachs in Ashkenazi Jewish).</li>
            </ul>
          </div>
          <div>
            <strong style={{color: '#dc2626', fontSize: '16px'}}>⚠️ Ethical Imperative</strong>
            <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}><strong>{'>'} 10% performance gap requires investigation before deployment.</strong> Unchecked disparities amplify healthcare inequities. Require diverse training data, fairness constraints, and human-in-loop for underrepresented groups.</p>
          </div>
        </div>
      )
    };
    return insights[stratifyBy] || '';
  };

  return (
    <div className="chart-view">
      <div className="chart-header">
        <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
      </div>
      <div className="chart-container">
        <div className="chart-main">
          <h3 className="chart-title">{view.title}</h3>
          
          <div style={{
            background: '#e0e7ff',
            padding: '20px',
            borderRadius: '8px',
            marginBottom: '30px',
            borderLeft: '4px solid #667eea'
          }}>
            <div style={{color: '#3730a3', fontSize: '15px'}}>
              {getInsights()}
            </div>
          </div>
          
          {stratData.config.groups.map((group, idx) => {
          const maxValues = getMaxValuesForGroup(group);
          
          return (
            <div key={group} style={{marginBottom: '40px'}}>
              <h4 style={{
                color: '#2d3748',
                fontSize: '20px',
                marginBottom: '20px',
                textTransform: 'capitalize',
                borderBottom: '2px solid #667eea',
                paddingBottom: '10px'
              }}>
                {group} Group ({stratData.data.reduce((sum, m) => sum + (m[group]?.n_samples || 0), 0)} samples)
              </h4>
              
              <div style={{overflowX: 'auto'}}>
                <table style={{width: '100%', borderCollapse: 'collapse', minWidth: '1800px'}}>
                  <thead>
                    <tr style={{background: '#667eea', color: 'white'}}>
                      <th style={{padding: '12px', textAlign: 'left', fontWeight: '600', position: 'sticky', left: 0, background: '#667eea', zIndex: 10}}>Model</th>
                      <th colSpan={2} style={{padding: '12px', textAlign: 'center', fontWeight: '600', borderBottom: '1px solid white'}}>F1 Score</th>
                      <th colSpan={2} style={{padding: '12px', textAlign: 'center', fontWeight: '600', borderBottom: '1px solid white'}}>Precision</th>
                      <th colSpan={2} style={{padding: '12px', textAlign: 'center', fontWeight: '600', borderBottom: '1px solid white'}}>Recall</th>
                      <th colSpan={2} style={{padding: '12px', textAlign: 'center', fontWeight: '600', borderBottom: '1px solid white'}}>AUC</th>
                      <th colSpan={3} style={{padding: '12px', textAlign: 'center', fontWeight: '600', borderBottom: '1px solid white'}}>P@K</th>
                      <th colSpan={3} style={{padding: '12px', textAlign: 'center', fontWeight: '600', borderBottom: '1px solid white'}}>R@K</th>
                      <th colSpan={3} style={{padding: '12px', textAlign: 'center', fontWeight: '600', borderBottom: '1px solid white'}}>F1@K</th>
                      <th style={{padding: '12px', textAlign: 'center', fontWeight: '600'}}>N</th>
                    </tr>
                    <tr style={{background: '#7c3aed', color: 'white', fontSize: '13px'}}>
                      <th style={{padding: '8px', textAlign: 'left', position: 'sticky', left: 0, background: '#7c3aed', zIndex: 10}}></th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Micro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Macro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Micro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Macro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Micro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Macro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Micro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>Macro</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@5</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@8</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@15</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@5</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@8</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@15</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@5</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@8</th>
                      <th style={{padding: '8px', textAlign: 'center'}}>@15</th>
                      <th style={{padding: '8px', textAlign: 'center'}}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {stratData.data.map((modelData, modelIdx) => {
                      const groupData = modelData[group] || {};
                      return (
                        <tr key={modelIdx} style={{background: modelIdx % 2 === 0 ? '#f9fafb' : 'white'}}>
                          <td style={{padding: '10px', border: '1px solid #e0e0e0', fontWeight: '500', position: 'sticky', left: 0, background: modelIdx % 2 === 0 ? '#f9fafb' : 'white', zIndex: 1}}>{modelData.model}</td>
                          <td style={getCellStyleStrat(groupData.f1_micro, 'f1_micro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.f1_micro ? (groupData.f1_micro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.f1_macro, 'f1_macro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.f1_macro ? (groupData.f1_macro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.prec_micro, 'prec_micro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.prec_micro ? (groupData.prec_micro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.prec_macro, 'prec_macro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.prec_macro ? (groupData.prec_macro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.rec_micro, 'rec_micro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.rec_micro ? (groupData.rec_micro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.rec_macro, 'rec_macro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.rec_macro ? (groupData.rec_macro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.auc_micro, 'auc_micro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.auc_micro ? (groupData.auc_micro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.auc_macro, 'auc_macro', maxValues, modelIdx % 2 === 0)}>
                            {groupData.auc_macro ? (groupData.auc_macro * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.prec_at_5, 'prec_at_5', maxValues, modelIdx % 2 === 0)}>
                            {groupData.prec_at_5 ? (groupData.prec_at_5 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.prec_at_8, 'prec_at_8', maxValues, modelIdx % 2 === 0)}>
                            {groupData.prec_at_8 ? (groupData.prec_at_8 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.prec_at_15, 'prec_at_15', maxValues, modelIdx % 2 === 0)}>
                            {groupData.prec_at_15 ? (groupData.prec_at_15 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.rec_at_5, 'rec_at_5', maxValues, modelIdx % 2 === 0)}>
                            {groupData.rec_at_5 ? (groupData.rec_at_5 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.rec_at_8, 'rec_at_8', maxValues, modelIdx % 2 === 0)}>
                            {groupData.rec_at_8 ? (groupData.rec_at_8 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.rec_at_15, 'rec_at_15', maxValues, modelIdx % 2 === 0)}>
                            {groupData.rec_at_15 ? (groupData.rec_at_15 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.f1_at_5, 'f1_at_5', maxValues, modelIdx % 2 === 0)}>
                            {groupData.f1_at_5 ? (groupData.f1_at_5 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.f1_at_8, 'f1_at_8', maxValues, modelIdx % 2 === 0)}>
                            {groupData.f1_at_8 ? (groupData.f1_at_8 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={getCellStyleStrat(groupData.f1_at_15, 'f1_at_15', maxValues, modelIdx % 2 === 0)}>
                            {groupData.f1_at_15 ? (groupData.f1_at_15 * 100).toFixed(1) + '%' : 'N/A'}
                          </td>
                          <td style={{padding: '10px', border: '1px solid #e0e0e0', textAlign: 'center', color: '#718096', fontSize: '13px'}}>
                            {groupData.n_samples || 'N/A'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
        
          <div style={{marginTop: '30px', textAlign: 'center', color: '#718096', fontSize: '14px', padding: '20px', background: '#f9fafb', borderRadius: '8px'}}>
            <p style={{margin: 0}}>🌟 <strong>Highlighted cells</strong> show the best (highest) value for each metric within each group</p>
          </div>
        </div>

        {/* Model Selector */}
        <div className="model-selector">
          <h3>⚙️ Select Models</h3>
          <div className="model-checkbox-list">
            {allModels.map(modelName => (
              <div key={modelName} className="model-checkbox-item">
                <input
                  type="checkbox"
                  id={`strat-model-${modelName}`}
                  checked={selectedModels.includes(modelName)}
                  onChange={() => toggleModel(modelName)}
                />
                <label htmlFor={`strat-model-${modelName}`}>{modelName}</label>
              </div>
            ))}
          </div>
          <div className="model-selector-actions">
            <button className="selector-btn selector-btn-all" onClick={selectAll}>
              All
            </button>
            <button className="selector-btn selector-btn-none" onClick={selectNone}>
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Calibration View Component
const CalibrationView = ({ view, onBack, modelNames, fileMap, allModels, selectedModels, toggleModel, selectAll, selectNone }) => {
  const [calibData, setCalibData] = useState(null);

  useEffect(() => {
    loadCalibrationData();
  }, [modelNames]); // Reload when selected models change

  const loadCalibrationData = async () => {
    const data = [];
    
    for (const modelName of modelNames) {
      const filePath = fileMap[modelName];
      if (filePath) {
        try {
          const response = await fetch(filePath);
          const json = await response.json();
          const calibration = json.calibration || {};
          
          data.push({
            model: modelName,
            ece: calibration.ece || 0,
            curve: calibration.calibration_curve || calibration.curve || []
          });
        } catch (err) {
          console.error(`Error loading ${modelName}:`, err);
        }
      }
    }
    
    setCalibData(data);
  };

  if (!calibData) {
    return (
      <div className="chart-view">
        <div className="chart-header">
          <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
        </div>
        <div className="loading-chart">
          <div className="spinner"></div>
          <p>Loading data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-view">
      <div className="chart-header">
        <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
      </div>
      <div className="chart-container">
        <div className="chart-main">
          <h3 className="chart-title">{view.title}</h3>
          
          <div style={{
            background: '#e0e7ff',
            padding: '20px',
            borderRadius: '8px',
            marginBottom: '30px',
            borderLeft: '4px solid #667eea'
          }}>
            <div style={{color: '#3730a3', fontSize: '15px'}}>
              <div style={{marginBottom: '12px'}}>
                <strong style={{color: '#1e40af', fontSize: '16px'}}>📊 What is Calibration?</strong>
              <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Well-calibrated models produce confidence scores matching actual accuracy. Example: 80% confidence = correct ~80% of time.</p>
            </div>
            <div style={{marginBottom: '12px'}}>
              <strong style={{color: '#1e40af', fontSize: '16px'}}>📈 ECE (Expected Calibration Error)</strong>
              <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Lower ECE = better calibration. Measures gap between predicted confidence and actual accuracy.</p>
            </div>
            <div>
              <strong style={{color: '#dc2626', fontSize: '16px'}}>⚠️ Clinical Risk</strong>
              <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>Poor calibration = unreliable confidence scores = dangerous in clinical settings. Physicians cannot trust model's uncertainty estimates.</p>
            </div>
          </div>
        </div>
        
        <div style={{marginBottom: '40px'}}>
          <h4 style={{
            color: '#2d3748',
            fontSize: '20px',
            marginBottom: '20px',
            borderBottom: '2px solid #667eea',
            paddingBottom: '10px'
          }}>
            Expected Calibration Error (ECE)
          </h4>
          <p style={{color: '#718096', marginBottom: '20px'}}>
            Lower ECE indicates better calibration (confidence aligns with accuracy)
          </p>
          
          <div style={{overflowX: 'auto'}}>
            <table style={{width: '100%', borderCollapse: 'collapse'}}>
              <thead>
                <tr style={{background: '#667eea', color: 'white'}}>
                  <th style={{padding: '14px', textAlign: 'left', fontWeight: '600'}}>Model</th>
                  <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>ECE</th>
                  <th style={{padding: '14px', textAlign: 'center', fontWeight: '600'}}>Calibration Quality</th>
                </tr>
              </thead>
              <tbody>
                {calibData.map((modelData, idx) => {
                  const ece = modelData.ece;
                  const quality = ece < 0.05 ? 'Excellent' : ece < 0.10 ? 'Good' : ece < 0.15 ? 'Fair' : 'Poor';
                  const color = ece < 0.05 ? '#10b981' : ece < 0.10 ? '#3b82f6' : ece < 0.15 ? '#f59e0b' : '#ef4444';
                  
                  // Find minimum ECE (best calibration)
                  const minECE = Math.min(...calibData.map(d => d.ece));
                  const isBest = ece === minECE;
                  
                  return (
                    <tr key={idx} style={{
                      background: isBest ? '#fef3c7' : (idx % 2 === 0 ? '#f9fafb' : 'white'),
                      boxShadow: isBest ? 'inset 0 0 0 2px #fbbf24' : 'none'
                    }}>
                      <td style={{padding: '12px', border: '1px solid #e0e0e0', fontWeight: isBest ? '700' : '500', color: isBest ? '#92400e' : 'inherit'}}>
                        {modelData.model} {isBest && '⭐'}
                      </td>
                      <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center', fontSize: '18px', fontWeight: '600', color: isBest ? '#92400e' : 'inherit'}}>
                        {(ece * 100).toFixed(2)}%
                      </td>
                      <td style={{padding: '12px', border: '1px solid #e0e0e0', textAlign: 'center'}}>
                        <span style={{
                          padding: '6px 12px',
                          borderRadius: '12px',
                          background: color + '20',
                          color: color,
                          fontWeight: '600'
                        }}>
                          {quality}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
        
        <div style={{marginTop: '40px'}}>
          <h4 style={{
            color: '#2d3748',
            fontSize: '20px',
            marginBottom: '20px',
            borderBottom: '2px solid #667eea',
            paddingBottom: '10px'
          }}>
            Calibration Curves
          </h4>
          <p style={{color: '#718096', marginBottom: '20px'}}>
            Showing confidence vs accuracy for each model
          </p>
          
          {calibData.map((modelData, idx) => (
            <div key={idx} style={{marginBottom: '30px'}}>
              <h5 style={{color: '#4a5568', marginBottom: '10px'}}>{modelData.model}</h5>
              {modelData.curve && modelData.curve.length > 0 ? (
                <div style={{overflowX: 'auto'}}>
                  <table style={{width: '100%', borderCollapse: 'collapse', fontSize: '13px'}}>
                    <thead>
                      <tr style={{background: '#e0e7ff'}}>
                        <th style={{padding: '8px', textAlign: 'center'}}>Bin</th>
                        {modelData.curve.map((bin, binIdx) => (
                          <th key={binIdx} style={{padding: '8px', textAlign: 'center'}}>{binIdx + 1}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{background: '#f9fafb'}}>
                        <td style={{padding: '8px', border: '1px solid #e0e0e0', fontWeight: '500'}}>Confidence</td>
                        {modelData.curve.map((bin, binIdx) => (
                          <td key={binIdx} style={{padding: '8px', border: '1px solid #e0e0e0', textAlign: 'center'}}>
                            {(bin.confidence * 100).toFixed(1)}%
                          </td>
                        ))}
                      </tr>
                      <tr style={{background: 'white'}}>
                        <td style={{padding: '8px', border: '1px solid #e0e0e0', fontWeight: '500'}}>Accuracy</td>
                        {modelData.curve.map((bin, binIdx) => (
                          <td key={binIdx} style={{padding: '8px', border: '1px solid #e0e0e0', textAlign: 'center'}}>
                            {(bin.accuracy * 100).toFixed(1)}%
                          </td>
                        ))}
                      </tr>
                      <tr style={{background: '#f9fafb'}}>
                        <td style={{padding: '8px', border: '1px solid #e0e0e0', fontWeight: '500'}}>Samples</td>
                        {modelData.curve.map((bin, binIdx) => (
                          <td key={binIdx} style={{padding: '8px', border: '1px solid #e0e0e0', textAlign: 'center', color: '#718096'}}>
                            {bin.count}
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{color: '#718096', fontStyle: 'italic'}}>No calibration curve data available</p>
              )}
            </div>
          ))}
        </div>
        
          <div style={{marginTop: '30px', textAlign: 'center', color: '#718096', fontSize: '14px', padding: '20px', background: '#f9fafb', borderRadius: '8px'}}>
            <p style={{margin: 0}}>⭐ <strong>Highlighted row</strong> shows the best calibrated model (lowest ECE)</p>
            <p style={{margin: '10px 0 0 0'}}>💡 <strong>Calibration curves</strong> show how well confidence scores match actual accuracy across different confidence bins</p>
          </div>
        </div>

        {/* Model Selector */}
        <div className="model-selector">
          <h3>⚙️ Select Models</h3>
          <div className="model-checkbox-list">
            {allModels.map(modelName => (
              <div key={modelName} className="model-checkbox-item">
                <input
                  type="checkbox"
                  id={`calib-model-${modelName}`}
                  checked={selectedModels.includes(modelName)}
                  onChange={() => toggleModel(modelName)}
                />
                <label htmlFor={`calib-model-${modelName}`}>{modelName}</label>
              </div>
            ))}
          </div>
          <div className="model-selector-actions">
            <button className="selector-btn selector-btn-all" onClick={selectAll}>
              All
            </button>
            <button className="selector-btn selector-btn-none" onClick={selectNone}>
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Performance Heatmap Component
const PerformanceHeatmap = ({ view, onBack, modelNames, fileMap, allModels, selectedModels, toggleModel, selectAll, selectNone }) => {
  const [heatmapData, setHeatmapData] = useState(null);

  useEffect(() => {
    loadHeatmapData();
  }, [modelNames]); // Reload when selected models change

  const loadHeatmapData = async () => {
    const data = [];
    
    const categories = [
      { label: 'Overall', key: 'overall', metric: 'f1_micro' },
      { label: 'Common Codes', key: 'code_freq.common', metric: 'f1_micro' },
      { label: 'Rare Codes', key: 'code_freq.rare', metric: 'f1_micro' },
      { label: 'Short Docs', key: 'length.short', metric: 'f1_micro' },
      { label: 'Long Docs', key: 'length.long', metric: 'f1_micro' },
      { label: 'Low Comorb', key: 'comorbidity.low', metric: 'f1_micro' },
      { label: 'High Comorb', key: 'comorbidity.high', metric: 'f1_micro' },
      { label: 'White', key: 'race.White', metric: 'f1_micro' },
      { label: 'Black', key: 'race.Black', metric: 'f1_micro' },
      { label: 'Asian', key: 'race.Asian', metric: 'f1_micro' },
      { label: 'Hispanic', key: 'race.Hispanic', metric: 'f1_micro' }
    ];
    
    for (const modelName of modelNames) {
      const filePath = fileMap[modelName];
      if (filePath) {
        try {
          const response = await fetch(filePath);
          const json = await response.json();
          
          const modelRow = { model: modelName };
          
          categories.forEach(cat => {
            const keys = cat.key.split('.');
            let value = json;
            
            if (keys[0] === 'overall') {
              value = json.overall?.[cat.metric] || 0;
            } else {
              value = json.stratified?.[keys[0]]?.[keys[1]]?.[cat.metric] || 0;
            }
            
            modelRow[cat.label] = value * 100; // Convert to percentage
          });
          
          data.push(modelRow);
        } catch (err) {
          console.error(`Error loading ${modelName}:`, err);
        }
      }
    }
    
    setHeatmapData({ data, categories: categories.map(c => c.label) });
  };

  const getColorForValue = (value) => {
    // Value is 0-100 (percentage)
    if (value >= 50) return '#10b981'; // Green
    if (value >= 40) return '#3b82f6'; // Blue
    if (value >= 30) return '#f59e0b'; // Orange
    if (value >= 20) return '#ef4444'; // Red
    return '#9ca3af'; // Gray
  };

  const getBackgroundOpacity = (value) => {
    // Value is 0-100
    const opacity = Math.min(Math.max(value / 100, 0.1), 0.9);
    return opacity;
  };

  if (!heatmapData) {
    return (
      <div className="chart-view">
        <div className="chart-header">
          <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
        </div>
        <div className="loading-chart">
          <div className="spinner"></div>
          <p>Loading heatmap...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-view">
      <div className="chart-header">
        <button className="back-button" onClick={onBack}>← Back to Dashboard</button>
      </div>
      <div className="chart-container">
        <div className="chart-main">
          <h3 className="chart-title">{view.title}</h3>
          
          <div style={{
            background: '#e0e7ff',
            padding: '20px',
            borderRadius: '8px',
            marginBottom: '30px',
            borderLeft: '4px solid #667eea'
          }}>
            <div style={{color: '#3730a3', fontSize: '15px'}}>
              <strong style={{color: '#1e40af', fontSize: '16px'}}>🔥 Visual Performance Comparison</strong>
              <p style={{margin: '8px 0 0 0', lineHeight: '1.6'}}>
                F1 Micro scores across all stratifications. Greener/darker = better performance. 
              Instantly reveals model strengths, weaknesses, and equity gaps across patient subgroups.
            </p>
          </div>
        </div>
        
        <div style={{overflowX: 'auto', marginBottom: '30px'}}>
          <table style={{
            width: '100%',
            borderCollapse: 'separate',
            borderSpacing: '4px',
            minWidth: '1000px'
          }}>
            <thead>
              <tr>
                <th style={{
                  padding: '14px',
                  textAlign: 'left',
                  fontWeight: '600',
                  background: '#667eea',
                  color: 'white',
                  position: 'sticky',
                  left: 0,
                  zIndex: 2
                }}>Model</th>
                {heatmapData.categories.map((cat, idx) => (
                  <th key={idx} style={{
                    padding: '14px 8px',
                    textAlign: 'center',
                    fontWeight: '600',
                    background: '#667eea',
                    color: 'white',
                    fontSize: '13px',
                    minWidth: '90px'
                  }}>{cat}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatmapData.data.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <td style={{
                    padding: '12px',
                    fontWeight: '600',
                    background: '#f3f4f6',
                    position: 'sticky',
                    left: 0,
                    zIndex: 1
                  }}>{row.model}</td>
                  {heatmapData.categories.map((cat, colIdx) => {
                    const value = row[cat] || 0;
                    const color = getColorForValue(value);
                    const opacity = getBackgroundOpacity(value);
                    
                    return (
                      <td key={colIdx} style={{
                        padding: '12px 8px',
                        textAlign: 'center',
                        fontWeight: '600',
                        background: `${color}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`,
                        color: value > 35 ? 'white' : '#1f2937',
                        fontSize: '14px'
                      }}>
                        {value.toFixed(1)}%
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div style={{marginTop: '30px'}}>
          <h4 style={{
            color: '#2d3748',
            fontSize: '18px',
            marginBottom: '15px',
            borderBottom: '2px solid #667eea',
            paddingBottom: '10px'
          }}>
            Color Legend
          </h4>
          <div style={{display: 'flex', gap: '20px', flexWrap: 'wrap'}}>
            <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
              <div style={{width: '30px', height: '30px', background: '#10b981', borderRadius: '4px'}}></div>
              <span style={{fontSize: '14px', color: '#4a5568'}}>≥50% (Excellent)</span>
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
              <div style={{width: '30px', height: '30px', background: '#3b82f6', borderRadius: '4px'}}></div>
              <span style={{fontSize: '14px', color: '#4a5568'}}>40-49% (Good)</span>
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
              <div style={{width: '30px', height: '30px', background: '#f59e0b', borderRadius: '4px'}}></div>
              <span style={{fontSize: '14px', color: '#4a5568'}}>30-39% (Fair)</span>
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
              <div style={{width: '30px', height: '30px', background: '#ef4444', borderRadius: '4px'}}></div>
              <span style={{fontSize: '14px', color: '#4a5568'}}>20-29% (Poor)</span>
            </div>
            <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
              <div style={{width: '30px', height: '30px', background: '#9ca3af', borderRadius: '4px'}}></div>
              <span style={{fontSize: '14px', color: '#4a5568'}}>{'<'} 20% (Very Poor)</span>
            </div>
          </div>
        </div>
        
        <div style={{
          marginTop: '30px',
          background: '#f9fafb',
          padding: '20px',
          borderRadius: '8px',
          borderLeft: '4px solid #10b981'
        }}>
          <h4 style={{margin: '0 0 12px 0', color: '#065f46', fontSize: '16px'}}>
            🔍 Key Observations
          </h4>
          <ul style={{margin: 0, paddingLeft: '20px', color: '#374151', lineHeight: '1.8'}}>
            <li><strong>ConvNet Dominance</strong>: Consistently outperforms LLMs across nearly all categories, demonstrating value of specialized clinical training.</li>
            <li><strong>Equity Gaps</strong>: All models show performance variation across racial groups, requiring further investigation before deployment.</li>
            <li><strong>Complexity Handling</strong>: High comorbidity performance reveals true clinical reasoning capability—ConvNet excels here.</li>
            <li><strong>Qwen Failure</strong>: Catastrophic performance across all categories suggests fundamental prompt engineering or instruction-following issues, not demographic-specific failures.</li>
          </ul>
        </div>
        </div>

        {/* Model Selector */}
        <div className="model-selector">
          <h3>⚙️ Select Models</h3>
          <div className="model-checkbox-list">
            {allModels.map(modelName => (
              <div key={modelName} className="model-checkbox-item">
                <input
                  type="checkbox"
                  id={`heatmap-model-${modelName}`}
                  checked={selectedModels.includes(modelName)}
                  onChange={() => toggleModel(modelName)}
                />
                <label htmlFor={`heatmap-model-${modelName}`}>{modelName}</label>
              </div>
            ))}
          </div>
          <div className="model-selector-actions">
            <button className="selector-btn selector-btn-all" onClick={selectAll}>
              All
            </button>
            <button className="selector-btn selector-btn-none" onClick={selectNone}>
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartView;
