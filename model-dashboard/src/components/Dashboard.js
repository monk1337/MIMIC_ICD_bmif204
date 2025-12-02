import React from 'react';
import './Dashboard.css';

const Dashboard = ({ onSelectView, modelsData }) => {
  const visualizations = [
    {
      id: 'summary-table',
      title: 'Summary Table',
      description: 'Comprehensive table with all model metrics (900 samples)',
      icon: '📋'
    },
    {
      id: 'f1-micro',
      title: 'F1 Micro Comparison',
      description: 'Compare F1 Micro scores across all models',
      icon: '📊'
    },
    {
      id: 'f1-macro',
      title: 'F1 Macro Comparison',
      description: 'Compare F1 Macro scores across all models',
      icon: '📈'
    },
    {
      id: 'precision-micro',
      title: 'Precision Micro Comparison',
      description: 'Compare Precision Micro across all models',
      icon: '🎯'
    },
    {
      id: 'recall-micro',
      title: 'Recall Micro Comparison',
      description: 'Compare Recall Micro across all models',
      icon: '🔍'
    },
    {
      id: 'auc-micro',
      title: 'AUC Micro Comparison',
      description: 'Compare AUC Micro across all models',
      icon: '📉'
    },
    {
      id: 'precision-at-5',
      title: 'Precision@K Comparison',
      description: 'Compare Precision@k for k=5,8,15',
      icon: '🎲'
    },
    {
      id: 'recall-at-5',
      title: 'Recall@K Comparison',
      description: 'Compare Recall@k for k=5,8,15',
      icon: '🎲'
    },
    {
      id: 'f1-at-5',
      title: 'F1@K Comparison',
      description: 'Compare F1@k for k=5,8,15',
      icon: '🎲'
    },
    {
      id: 'stratified-frequency',
      title: 'Performance by Code Frequency',
      description: 'Compare performance across common, medium, and rare codes',
      icon: '📊'
    },
    {
      id: 'stratified-length',
      title: 'Performance by Document Length',
      description: 'Compare performance across short, medium, and long documents',
      icon: '📏'
    },
    {
      id: 'stratified-comorbidity',
      title: 'Performance by Comorbidity Burden',
      description: 'Compare performance across low, medium, and high comorbidity',
      icon: '🏥'
    },
    {
      id: 'stratified-race',
      title: 'Health Equity Analysis',
      description: 'Compare performance across racial/ethnic groups',
      icon: '⚖️'
    },
    {
      id: 'calibration',
      title: 'Model Calibration',
      description: 'Expected Calibration Error and calibration curves',
      icon: '🎚️'
    },
    {
      id: 'performance-heatmap',
      title: 'Performance Heatmap',
      description: 'Visual comparison of F1 scores across all stratifications',
      icon: '🔥'
    },
    {
      id: 'interesting-cases',
      title: '🔍 Interesting Case Studies',
      description: 'Explore cases where specific models excelled (93 ConvNet wins, 3 RAG wins, 18 Ensemble wins)',
      icon: '💡'
    }
  ];

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-icon">🎯</div>
          <h1>ICD-10 Model Comparison Dashboard</h1>
          <span className="dataset-badge">900 Samples</span>
        </div>
        <p className="header-subtitle">
          Comprehensive evaluation of Trained ConvNet vs LLMs on ICD-10 coding task.<br/>
          Includes clinical subgroup analyses: comorbidity, health equity, and calibration metrics.
        </p>
      </header>

      <div className="visualizations-section">
        <h2 className="section-title">
          <span className="title-icon">📊</span>
          Available Visualizations
        </h2>
        
        <div className="cards-grid">
          {visualizations.map((viz) => (
            <div
              key={viz.id}
              className="viz-card"
              onClick={() => onSelectView(viz)}
            >
              <div className="card-icon">{viz.icon}</div>
              <h3 className="card-title">{viz.title}</h3>
              <p className="card-description">{viz.description}</p>
              <div className="card-arrow">→</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
