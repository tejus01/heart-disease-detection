from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import xgboost as xgb

app = Flask(__name__)

# Load the dataset and train the model
def train_model():
    data = pd.read_csv('heart.csv')
    
    # Features and Target
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Scaling and splitting the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred
    }
    
    return results, model

# Create plots
def create_plots(results, model):
    # Confusion matrix plot
    confusion_matrix = pd.crosstab(results['y_test'], results['y_pred'], rownames=['Actual'], colnames=['Predicted'])
    
    # Feature importance plot
    importance = model.get_booster().get_score(importance_type='weight')
    feature_names = list(importance.keys())
    feature_scores = list(importance.values())
    
    # Create a subplot with two rows (confusion matrix, feature importance)
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Confusion Matrix", "Feature Importance"))
    
    # Confusion Matrix Plot
    trace1 = go.Heatmap(z=confusion_matrix.values, 
                        x=confusion_matrix.columns, 
                        y=confusion_matrix.index,
                        colorscale='Viridis')
    
    # Feature Importance Bar Plot
    trace2 = go.Bar(x=feature_names, y=feature_scores, marker=dict(color='blue'))
    
    fig.append_trace(trace1, row=1, col=1)
    fig.append_trace(trace2, row=2, col=1)
    
    fig.update_layout(height=700, title_text="Model Results")
    
    return fig.to_html()

@app.route('/')
def index():
    # Train the model and get results
    results, model = train_model()
    
    # Generate plot HTML
    plot_html = create_plots(results, model)
    
    # Render the results in the HTML template
    return render_template('result.html', accuracy=results['accuracy'],
                           precision=results['precision'], recall=results['recall'],
                           f1_score=results['f1_score'], plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)
