#----------------prestep----------------#

import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

def load_data(file_path):
    return pd.read_csv(file_path)

testdata = load_data('PA2\\adult_test.csv')
traindata = load_data('PA2\\adult_train.csv')

Ytest = testdata['target']
Ytrain = traindata['target']
Xtest = testdata.drop('target', axis=1)
Xtrain = traindata.drop('target', axis=1)

# Convert to dict format once (for efficiency)
Xtrain_dict = Xtrain.to_dict(orient='records')
Xtest_dict = Xtest.to_dict(orient='records')

#----------------Task 2: Part 1 - Decision Tree Underfitting/Overfitting----------------#

def plot_scores(results_dict, title, xlabel='Max Depth'):
    """Plot train and test scores for different experiments."""
    plt.figure(figsize=(12, 8))
    
    for label, (train_scores, test_scores) in results_dict.items():
        depths = sorted(train_scores.keys())
        train_y = [train_scores[d] for d in depths]
        test_y = [test_scores[d] for d in depths]
        
        plt.plot(depths, train_y, marker='o', linestyle='--', label=f'{label} (Train)')
        plt.plot(depths, test_y, marker='s', linestyle='-', label=f'{label} (Test)')
    
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim(0.6, 1.0)
    plt.show()

# Experiment with DecisionTreeClassifier
print("=" * 60)
print("Task 2, Part 1: Decision Tree Underfitting/Overfitting")
print("=" * 60)

custom_depths = range(1, 25)
dt_train_scores = {}
dt_test_scores = {}

for d in custom_depths:
    pipeline = make_pipeline(
        DictVectorizer(),
        DecisionTreeClassifier(max_depth=d, random_state=42)
    )
    
    pipeline.fit(Xtrain_dict, Ytrain)
    dt_train_scores[d] = pipeline.score(Xtrain_dict, Ytrain)
    dt_test_scores[d] = pipeline.score(Xtest_dict, Ytest)
    
print(f"Decision Tree - Best test accuracy: {max(dt_test_scores.values()):.4f} at depth {max(dt_test_scores, key=dt_test_scores.get)}")

# Plot Decision Tree results
plot_scores(
    {'Decision Tree': (dt_train_scores, dt_test_scores)},
    'Decision Tree: Underfitting vs Overfitting'
)

#----------------Task 2: Part 2 - Random Forest Underfitting/Overfitting----------------#

print("\n" + "=" * 60)
print("Task 2, Part 2: Random Forest Underfitting/Overfitting")
print("=" * 60)

# Different ensemble sizes to investigate
n_estimators_list = [1, 10, 50, 100, 200]
custom_depths_rf = range(1, 25)

# Store results for all experiments
all_results = {}
training_times = {}
best_test_accuracies = {}

for n_est in n_estimators_list:
    rf_train_scores = {}
    rf_test_scores = {}
    total_time = 0
    
    print(f"\nTraining Random Forest with n_estimators={n_est}...")
    
    for d in custom_depths_rf:
        start_time = time.time()
        
        pipeline = make_pipeline(
            DictVectorizer(),
            RandomForestClassifier(
                n_estimators=n_est, 
                max_depth=d, 
                random_state=42,
                n_jobs=-1  # Use all CPU cores for parallel training
            )
        )
        
        pipeline.fit(Xtrain_dict, Ytrain)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        
        rf_train_scores[d] = pipeline.score(Xtrain_dict, Ytrain)
        rf_test_scores[d] = pipeline.score(Xtest_dict, Ytest)
    
    all_results[f'RF (n={n_est})'] = (rf_train_scores, rf_test_scores)
    training_times[n_est] = total_time
    best_test_accuracies[n_est] = max(rf_test_scores.values())
    
    print(f"  Best test accuracy: {best_test_accuracies[n_est]:.4f}")
    print(f"  Total training time: {total_time:.2f} seconds")

# Plot Random Forest results - all ensemble sizes
plt.figure(figsize=(14, 10))
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, (n_est, color) in enumerate(zip(n_estimators_list, colors)):
    train_scores, test_scores = all_results[f'RF (n={n_est})']
    depths = sorted(train_scores.keys())
    train_y = [train_scores[d] for d in depths]
    test_y = [test_scores[d] for d in depths]
    
    plt.plot(depths, train_y, marker='o', linestyle='--', color=color, 
             alpha=0.5, label=f'n_estimators={n_est} (Train)')
    plt.plot(depths, test_y, marker='s', linestyle='-', color=color, 
             label=f'n_estimators={n_est} (Test)')

plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Random Forest: Effect of Ensemble Size on Underfitting/Overfitting')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.ylim(0.6, 1.0)
plt.tight_layout()
plt.show()

# Compare Decision Tree with Random Forest (n=1)
print("\n" + "=" * 60)
print("Comparison: Decision Tree vs Random Forest (n_estimators=1)")
print("=" * 60)

comparison_results = {
    'Decision Tree': (dt_train_scores, dt_test_scores),
    'RF (n=1)': all_results['RF (n=1)']
}
plot_scores(
    comparison_results,
    'Decision Tree vs Random Forest (n_estimators=1)'
)

#----------------Summary Statistics----------------#

print("\n" + "=" * 60)
print("Summary: Best Test Accuracies by Ensemble Size")
print("=" * 60)

print(f"{'Model':<25} {'Best Test Accuracy':<20} {'Training Time (s)':<20}")
print("-" * 65)
print(f"{'Decision Tree':<25} {max(dt_test_scores.values()):<20.4f} {'N/A':<20}")
for n_est in n_estimators_list:
    print(f"{'RF (n=' + str(n_est) + ')':<25} {best_test_accuracies[n_est]:<20.4f} {training_times[n_est]:<20.2f}")

# Plot training time vs ensemble size
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, [training_times[n] for n in n_estimators_list], 
         marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Total Training Time (seconds)')
plt.title('Training Time vs Ensemble Size')
plt.grid(True)
plt.show()

# Plot best test accuracy vs ensemble size
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, [best_test_accuracies[n] for n in n_estimators_list], 
         marker='o', linewidth=2, markersize=8, color='green')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Best Test Accuracy')
plt.title('Best Test Accuracy vs Ensemble Size')
plt.grid(True)
plt.show()

#----------------Discussion Points----------------#

print("\n" + "=" * 60)
print("Discussion Points for Report")
print("=" * 60)

print("""
1. Decision Tree vs Random Forest (n=1):
   - A single decision tree is deterministic (same result every time with same data)
   - Random Forest with n=1 uses random feature subsampling at each split
   - This randomness causes different splits and potentially different accuracy
   - RF(n=1) may show slightly different overfitting behavior due to feature randomization

2. Effect of increasing ensemble size:
   - As ensemble size grows, the overfitting curve becomes smoother
   - The gap between training and test accuracy tends to decrease
   - The model becomes more robust to noise

3. Best test accuracy vs ensemble size:
   - Generally improves with more trees, but with diminishing returns
   - After a certain point, adding more trees provides minimal benefit

4. Training time vs ensemble size:
   - Training time increases roughly linearly with the number of trees
   - n_jobs=-1 helps parallelize training across CPU cores
""")


