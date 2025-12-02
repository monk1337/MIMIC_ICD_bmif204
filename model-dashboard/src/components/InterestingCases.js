import React, { useState, useEffect } from 'react';
import './InterestingCases.css';

const InterestingCases = ({ onBack }) => {
  const [activeTab, setActiveTab] = useState('case1');
  const [cases, setCases] = useState({ case1: [], case2: [], case3: [] });
  const [expandedCase, setExpandedCase] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCases();
  }, []);

  const loadCases = async () => {
    try {
      const [case1Response, case2Response, case3Response] = await Promise.all([
        fetch('/interesting_cases/case1_convnet_right_llms_wrong.csv'),
        fetch('/interesting_cases/case2_rag_right_others_wrong_rare.csv'),
        fetch('/interesting_cases/case3_ensemble_right_others_wrong.csv')
      ]);

      const case1Text = await case1Response.text();
      const case2Text = await case2Response.text();
      const case3Text = await case3Response.text();

      setCases({
        case1: parseCSV(case1Text),
        case2: parseCSV(case2Text),
        case3: parseCSV(case3Text)
      });
      setLoading(false);
    } catch (error) {
      console.error('Error loading cases:', error);
      setLoading(false);
    }
  };

  const parseCSV = (csvText) => {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    return lines.slice(1).map(line => {
      // Handle quoted fields with commas
      const values = [];
      let current = '';
      let inQuotes = false;
      
      for (let char of line) {
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          values.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      values.push(current.trim());
      
      const obj = {};
      headers.forEach((header, i) => {
        obj[header] = values[i] || '';
      });
      return obj;
    });
  };

  const extractClinicalSummary = (dischargeSummary) => {
    if (!dischargeSummary) return { chief: '', hpi: '', findings: [] };
    
    const text = dischargeSummary.toLowerCase();
    
    // Extract chief complaint
    const chiefMatch = dischargeSummary.match(/chief complaint[:\s]+([^\.]+)/i);
    const chief = chiefMatch ? chiefMatch[1].trim().substring(0, 100) : '';
    
    // Extract key findings (look for diagnosis, procedure, or condition keywords)
    const findings = [];
    const sentences = dischargeSummary.split(/[\.!?]+/).slice(0, 3);
    sentences.forEach(sentence => {
      if (sentence.length > 30 && sentence.length < 200) {
        findings.push(sentence.trim());
      }
    });
    
    return { chief, findings: findings.slice(0, 3) };
  };

  const toggleCase = (caseId) => {
    setExpandedCase(expandedCase === caseId ? null : caseId);
  };

  const renderCase1 = (caseData, index) => {
    const isExpanded = expandedCase === `case1-${index}`;
    const summary = extractClinicalSummary(caseData.discharge_summary);
    
    return (
      <div key={index} className="case-card">
        <div className="case-header" onClick={() => toggleCase(`case1-${index}`)}>
          <div className="case-title">
            <span className="case-number">Case #{caseData.sample_id}</span>
            <span className="case-badge success">ConvNet Success</span>
            <span className="case-badge failure">All LLMs Failed</span>
          </div>
          <div className="case-metrics">
            <span className="metric">ConvNet F1: <strong>{(parseFloat(caseData.convnet_f1) * 100).toFixed(1)}%</strong></span>
            <span className="metric">Codes: {caseData.num_actual}</span>
            <span className="metric freq">{caseData.code_freq}</span>
          </div>
          <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
        </div>
        
        {isExpanded && (
          <div className="case-details">
            <div className="section">
              <h4>📋 Clinical Summary</h4>
              <div className="clinical-excerpt">
                {summary.chief && (
                  <div className="excerpt-field">
                    <strong>Chief Complaint:</strong> {summary.chief}
                  </div>
                )}
                {summary.findings.length > 0 && (
                  <div className="excerpt-field">
                    <strong>Key Clinical Points:</strong>
                    <ul>
                      {summary.findings.map((finding, i) => (
                        <li key={i}>{finding.substring(0, 150)}...</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            <div className="section">
              <h4>🎯 Ground Truth vs Predictions</h4>
              <div className="codes-comparison">
                <div className="codes-row actual">
                  <strong>Actual Codes ({caseData.num_actual}):</strong>
                  <div className="code-chips">{caseData.actual_codes}</div>
                </div>
                <div className="codes-row success">
                  <strong>✅ ConvNet (F1: {(parseFloat(caseData.convnet_f1) * 100).toFixed(1)}%):</strong>
                  <div className="code-chips correct">{caseData.convnet_correct}</div>
                  <div className="code-chips">{caseData.convnet_predicted}</div>
                </div>
              </div>
            </div>

            <div className="section">
              <h4>❌ LLM Failures</h4>
              <div className="llm-failures">
                {['gemini_2.0_flash', 'claude_3.7_sonnet', 'gpt_4o', 'deepseek_v3'].map(model => {
                  const modelName = model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                  const f1 = caseData[`${model}_f1`];
                  const predicted = caseData[`${model}_predicted`];
                  const correct = caseData[`${model}_correct`];
                  
                  return (
                    <div key={model} className="llm-result">
                      <strong>{modelName}</strong>
                      <span className="f1-score low">F1: {(parseFloat(f1) * 100).toFixed(1)}%</span>
                      {correct && <div className="code-chips correct mini">{correct}</div>}
                      <div className="code-chips mini">{predicted}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCase2 = (caseData, index) => {
    const isExpanded = expandedCase === `case2-${index}`;
    const summary = extractClinicalSummary(caseData.discharge_summary);
    
    return (
      <div key={index} className="case-card">
        <div className="case-header" onClick={() => toggleCase(`case2-${index}`)}>
          <div className="case-title">
            <span className="case-number">Case #{caseData.sample_id}</span>
            <span className="case-badge success">RAG Success</span>
            <span className="case-badge failure">All Others Failed</span>
          </div>
          <div className="case-metrics">
            <span className="metric">RAG F1: <strong>{(parseFloat(caseData.rag_f1) * 100).toFixed(1)}%</strong></span>
            <span className="metric">Codes: {caseData.num_actual}</span>
            <span className="metric freq rare">{caseData.code_freq}</span>
          </div>
          <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
        </div>
        
        {isExpanded && (
          <div className="case-details">
            <div className="section">
              <h4>📋 Clinical Summary</h4>
              <div className="clinical-excerpt">
                {summary.chief && (
                  <div className="excerpt-field">
                    <strong>Chief Complaint:</strong> {summary.chief}
                  </div>
                )}
                {summary.findings.length > 0 && (
                  <div className="excerpt-field">
                    <strong>Key Clinical Points:</strong>
                    <ul>
                      {summary.findings.map((finding, i) => (
                        <li key={i}>{finding.substring(0, 150)}...</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            <div className="section rag-section">
              <h4>🔍 RAG Pipeline Process</h4>
              <div className="rag-steps">
                <div className="step">
                  <div className="step-number">1</div>
                  <div className="step-content">
                    <strong>Entity Extraction ({caseData.rag_num_entities} entities)</strong>
                    <div className="entities">
                      {caseData.rag_entities_extracted.split('|').map((entity, i) => (
                        <span key={i} className="entity-chip">{entity.trim()}</span>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="step">
                  <div className="step-number">2</div>
                  <div className="step-content">
                    <strong>Code Retrieval</strong>
                    <div className="retrieval-stats">
                      Retrieved {caseData.rag_codes_retrieved} candidate codes from vector database
                    </div>
                  </div>
                </div>
                
                <div className="step">
                  <div className="step-number">3</div>
                  <div className="step-content">
                    <strong>LLM Reasoning & Selection</strong>
                    <div className="reasoning-summary">
                      Analyzed candidates with clinical reasoning to select final codes
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="section">
              <h4>🎯 Results Comparison</h4>
              <div className="codes-comparison">
                <div className="codes-row actual">
                  <strong>Actual Codes ({caseData.num_actual}):</strong>
                  <div className="code-chips">{caseData.actual_codes}</div>
                </div>
                <div className="codes-row success">
                  <strong>✅ RAG Pipeline (F1: {(parseFloat(caseData.rag_f1) * 100).toFixed(1)}%):</strong>
                  <div className="code-chips correct">{caseData.rag_correct}</div>
                  <div className="code-chips">{caseData.rag_predicted}</div>
                </div>
                <div className="codes-row failure">
                  <strong>❌ ConvNet (F1: {(parseFloat(caseData.convnet_f1) * 100).toFixed(1)}%):</strong>
                  <div className="code-chips">{caseData.convnet_predicted}</div>
                </div>
              </div>
            </div>

            <div className="section">
              <h4>❌ Other Model Failures</h4>
              <div className="llm-failures">
                {['gemini_2.0_flash', 'qwen_30b', 'claude_3.7_sonnet', 'claude_haiku_4.5', 'gpt_5_mini', 'gpt_4o', 'deepseek_v3'].map(model => {
                  const modelName = model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                  const f1 = caseData[`${model}_f1`];
                  const predicted = caseData[`${model}_predicted`];
                  const correct = caseData[`${model}_correct`];
                  
                  return (
                    <div key={model} className="llm-result">
                      <strong>{modelName}</strong>
                      <span className="f1-score low">F1: {(parseFloat(f1) * 100).toFixed(1)}%</span>
                      {correct && <div className="code-chips correct mini">{correct}</div>}
                      <div className="code-chips mini">{predicted}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderCase3 = (caseData, index) => {
    const isExpanded = expandedCase === `case3-${index}`;
    const summary = extractClinicalSummary(caseData.discharge_summary);
    
    return (
      <div key={index} className="case-card">
        <div className="case-header" onClick={() => toggleCase(`case3-${index}`)}>
          <div className="case-title">
            <span className="case-number">Case #{caseData.sample_id}</span>
            <span className="case-badge success">Ensemble Success</span>
            <span className="case-badge failure">All Individual Models Failed</span>
          </div>
          <div className="case-metrics">
            <span className="metric">Ensemble F1: <strong>{(parseFloat(caseData.ensemble_f1) * 100).toFixed(1)}%</strong></span>
            <span className="metric">Codes: {caseData.num_actual}</span>
            <span className="metric freq">{caseData.code_freq}</span>
          </div>
          <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
        </div>
        
        {isExpanded && (
          <div className="case-details">
            <div className="section">
              <h4>📋 Clinical Summary</h4>
              <div className="clinical-excerpt">
                {summary.chief && (
                  <div className="excerpt-field">
                    <strong>Chief Complaint:</strong> {summary.chief}
                  </div>
                )}
                {summary.findings.length > 0 && (
                  <div className="excerpt-field">
                    <strong>Key Clinical Points:</strong>
                    <ul>
                      {summary.findings.map((finding, i) => (
                        <li key={i}>{finding.substring(0, 150)}...</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            <div className="section ensemble-section">
              <h4>⚡ Ensemble Arbiter Process</h4>
              <div className="ensemble-flow">
                <div className="flow-step">
                  <div className="flow-label">Input 1: CNN</div>
                  <div className="code-chips mini">{caseData.ensemble_cnn_input}</div>
                </div>
                
                <div className="flow-arrow">+</div>
                
                <div className="flow-step">
                  <div className="flow-label">Input 2: LLM (GPT-4o)</div>
                  <div className="code-chips mini">{caseData.ensemble_llm_input}</div>
                </div>
                
                <div className="flow-arrow">→</div>
                
                <div className="flow-step">
                  <div className="flow-label">Candidate Union</div>
                  <div className="code-chips mini">{caseData.ensemble_candidate_union}</div>
                </div>
                
                <div className="flow-arrow">→</div>
                
                <div className="flow-step">
                  <div className="flow-label">Filter Invalid</div>
                  {caseData.ensemble_invalid_filtered && (
                    <div className="code-chips mini invalid">{caseData.ensemble_invalid_filtered}</div>
                  )}
                </div>
                
                <div className="flow-arrow">→</div>
                
                <div className="flow-step highlight">
                  <div className="flow-label">LLM Arbiter Decision</div>
                  <div className="code-chips">{caseData.ensemble_predicted}</div>
                </div>
              </div>
            </div>

            <div className="section">
              <h4>🎯 Results Comparison</h4>
              <div className="codes-comparison">
                <div className="codes-row actual">
                  <strong>Actual Codes ({caseData.num_actual}):</strong>
                  <div className="code-chips">{caseData.actual_codes}</div>
                </div>
                <div className="codes-row success">
                  <strong>✅ Ensemble (F1: {(parseFloat(caseData.ensemble_f1) * 100).toFixed(1)}%):</strong>
                  <div className="code-chips correct">{caseData.ensemble_correct}</div>
                  <div className="code-chips">{caseData.ensemble_predicted}</div>
                </div>
                <div className="codes-row failure">
                  <strong>❌ ConvNet (F1: {(parseFloat(caseData.convnet_f1) * 100).toFixed(1)}%):</strong>
                  <div className="code-chips">{caseData.convnet_predicted}</div>
                </div>
              </div>
            </div>

            <div className="section">
              <h4>❌ Individual Model Failures</h4>
              <div className="llm-failures">
                {['gemini_2.0_flash', 'qwen_30b', 'claude_3.7_sonnet', 'claude_haiku_4.5', 'gpt_5_mini', 'gpt_4o', 'deepseek_v3'].map(model => {
                  const modelName = model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                  const f1 = caseData[`${model}_f1`];
                  const predicted = caseData[`${model}_predicted`];
                  const correct = caseData[`${model}_correct`];
                  
                  return (
                    <div key={model} className="llm-result">
                      <strong>{modelName}</strong>
                      <span className="f1-score low">F1: {(parseFloat(f1) * 100).toFixed(1)}%</span>
                      {correct && <div className="code-chips correct mini">{correct}</div>}
                      <div className="code-chips mini">{predicted}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="interesting-cases">
        <div className="loading-chart">
          <div className="spinner"></div>
          <p>Loading interesting cases...</p>
        </div>
      </div>
    );
  }

  const activeCases = cases[activeTab];

  return (
    <div className="interesting-cases">
      <div className="cases-header">
        <button className="back-button" onClick={onBack}>
          ← Back to Dashboard
        </button>
        <h2>🔍 Interesting Case Studies</h2>
        <p className="subtitle">Explore cases where specific models succeeded while others failed</p>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'case1' ? 'active' : ''}`}
          onClick={() => setActiveTab('case1')}
        >
          <span className="tab-icon">🧠</span>
          <span className="tab-label">ConvNet Wins</span>
          <span className="tab-count">{cases.case1.length}</span>
        </button>
        <button
          className={`tab ${activeTab === 'case2' ? 'active' : ''}`}
          onClick={() => setActiveTab('case2')}
        >
          <span className="tab-icon">🔍</span>
          <span className="tab-label">RAG Wins (Rare)</span>
          <span className="tab-count">{cases.case2.length}</span>
        </button>
        <button
          className={`tab ${activeTab === 'case3' ? 'active' : ''}`}
          onClick={() => setActiveTab('case3')}
        >
          <span className="tab-icon">⚡</span>
          <span className="tab-label">Ensemble Wins</span>
          <span className="tab-count">{cases.case3.length}</span>
        </button>
      </div>

      <div className="cases-content">
        <div className="cases-list">
          {activeTab === 'case1' && activeCases.map((caseData, i) => renderCase1(caseData, i))}
          {activeTab === 'case2' && activeCases.map((caseData, i) => renderCase2(caseData, i))}
          {activeTab === 'case3' && activeCases.map((caseData, i) => renderCase3(caseData, i))}
        </div>
      </div>
    </div>
  );
};

export default InterestingCases;
