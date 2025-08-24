import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def extract_sequence_features(seq):
    """Extract numerical features from protein sequences"""
    if pd.isna(seq) or seq == '':
        return np.zeros(23)  # 20 amino acids + 3 physicochemical properties
    
    # Amino acid composition
    aa_count = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0,
                'K': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0,
                'T': 0, 'V': 0, 'W': 0, 'Y': 0}
    
    for aa in seq.upper():
        if aa in aa_count:
            aa_count[aa] += 1
    
    total_length = len(seq)
    if total_length == 0:
        return np.zeros(23)
    
    # Normalize by sequence length
    features = [count / total_length for count in aa_count.values()]
    
    # Add physicochemical properties
    features.append(total_length)  # Sequence length
    
    # Hydrophobicity (approximate)
    hydrophobic_aas = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
    hydrophobicity = sum(aa_count[aa] for aa in hydrophobic_aas) / total_length
    features.append(hydrophobicity)
    
    # Charge (approximate)
    positive_aas = ['K', 'R', 'H']
    negative_aas = ['D', 'E']
    net_charge = (sum(aa_count[aa] for aa in positive_aas) - 
                  sum(aa_count[aa] for aa in negative_aas)) / total_length
    features.append(net_charge)
    
    return np.array(features)

def load_and_preprocess_data(csv_path):
    """Load and preprocess the antibody-antigen dataset"""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Remove any rows with missing critical data
    df = df.dropna(subset=['antibody_seq_a', 'antibody_seq_b', 'antigen_seq', 'delta_g'])
    
    print(f"After cleaning: {df.shape}")
    
    # Extract features for each sequence type
    print("Extracting sequence features...")
    
    # Heavy chain features
    heavy_features = np.array([extract_sequence_features(seq) for seq in df['antibody_seq_a']])
    heavy_df = pd.DataFrame(heavy_features, columns=[f'heavy_{i}' for i in range(23)])
    
    # Light chain features  
    light_features = np.array([extract_sequence_features(seq) for seq in df['antibody_seq_b']])
    light_df = pd.DataFrame(light_features, columns=[f'light_{i}' for i in range(23)])
    
    # Antigen features
    antigen_features = np.array([extract_sequence_features(seq) for seq in df['antigen_seq']])
    antigen_df = pd.DataFrame(antigen_features, columns=[f'antigen_{i}' for i in range(23)])
    
    # Combine all features
    X = pd.concat([heavy_df, light_df, antigen_df], axis=1)
    y = df['delta_g'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y, df

def train_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train multiple regression models and return results"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use scaled data for linear models, original for tree-based
        if 'Linear' in name or 'Ridge' in name:
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results[name] = {
            'val_mse': val_mse,
            'val_r2': val_r2, 
            'val_mae': val_mae,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
        
        predictions[name] = {
            'val_pred': y_val_pred,
            'test_pred': y_test_pred
        }
    
    return results, predictions

def create_visualizations(results, predictions, y_val, y_test):
    """Create comprehensive result visualizations"""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Performance metrics comparison
    ax1 = plt.subplot(2, 3, 1)
    models = list(results.keys())
    test_r2 = [results[model]['test_r2'] for model in models]
    test_mse = [results[model]['test_mse'] for model in models]
    
    x = np.arange(len(models))
    bars = ax1.bar(x, test_r2, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison (R²)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. MSE comparison
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(x, test_mse, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax2.set_title('Model Performance Comparison (MSE)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(test_mse)*0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Validation vs Test performance
    ax3 = plt.subplot(2, 3, 3)
    val_r2 = [results[model]['val_r2'] for model in models]
    
    ax3.scatter(val_r2, test_r2, s=100, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    for i, model in enumerate(models):
        ax3.annotate(model, (val_r2[i], test_r2[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    # Perfect correlation line
    min_r2, max_r2 = min(val_r2 + test_r2), max(val_r2 + test_r2)
    ax3.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', alpha=0.5, label='Perfect Correlation')
    ax3.set_xlabel('Validation R²', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Test R²', fontsize=12, fontweight='bold')
    ax3.set_title('Validation vs Test Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4-6. Prediction scatter plots for best models
    best_models = sorted(results.keys(), key=lambda x: results[x]['test_r2'], reverse=True)[:3]
    
    for i, model in enumerate(best_models):
        ax = plt.subplot(2, 3, 4 + i)
        test_pred = predictions[model]['test_pred']
        
        ax.scatter(y_test, test_pred, alpha=0.6, s=50, color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i])
        
        # Perfect prediction line
        min_val, max_val = min(min(y_test), min(test_pred)), max(max(y_test), max(test_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('True ΔG', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted ΔG', fontsize=12, fontweight='bold')
        ax.set_title(f'{model}\nR²={results[model]["test_r2"]:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('antibody_ml_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed metrics table
    print("\n" + "="*80)
    print("DETAILED MODEL PERFORMANCE METRICS")
    print("="*80)
    print(f"{'Model':<20} {'Val R²':<8} {'Test R²':<8} {'Val MSE':<8} {'Test MSE':<8} {'Test MAE':<8}")
    print("-"*80)
    
    for model in models:
        r = results[model]
        print(f"{model:<20} {r['val_r2']:<8.3f} {r['test_r2']:<8.3f} {r['val_mse']:<8.2f} {r['test_mse']:<8.2f} {r['test_mae']:<8.2f}")
    
    return fig

def run_antibody_ml_pipeline(csv_path):
    """Main pipeline function"""
    
    # Load and preprocess data
    X, y, df = load_and_preprocess_data(csv_path)
    
    # Split data: 80% train, 10% validation, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)  # 0.111 of 0.9 = 0.1 of total
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    
    # Train models
    results, predictions = train_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Create visualizations
    fig = create_visualizations(results, predictions, y_val, y_test)
    
    # Summary statistics
    print(f"\nDataset Summary:")
    print(f"Target variable (ΔG) statistics:")
    print(f"Mean: {y.mean():.2f}")
    print(f"Std:  {y.std():.2f}")
    print(f"Min:  {y.min():.2f}")
    print(f"Max:  {y.max():.2f}")
    
    return results, predictions, fig