import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import ChartView from './components/ChartView';
import InterestingCases from './components/InterestingCases';

function App() {
  const [selectedView, setSelectedView] = useState(null);
  const [modelsData, setModelsData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load all JSON files from the models_results folder
    // In production, you'd fetch from an API or have files served
    loadModelsData();
  }, []);

  const loadModelsData = async () => {
    try {
      // This will be populated with actual data loading logic
      // For now, we'll use a placeholder that can read from public folder
      setLoading(false);
    } catch (error) {
      console.error('Error loading models data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading model data...</p>
      </div>
    );
  }

  return (
    <div className="App">
      {!selectedView ? (
        <Dashboard onSelectView={setSelectedView} modelsData={modelsData} />
      ) : selectedView.id === 'interesting-cases' ? (
        <InterestingCases onBack={() => setSelectedView(null)} />
      ) : (
        <ChartView 
          view={selectedView} 
          modelsData={modelsData}
          onBack={() => setSelectedView(null)} 
        />
      )}
    </div>
  );
}

export default App;
