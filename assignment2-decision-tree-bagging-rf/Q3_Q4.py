import numpy as np
import pandas as pd

# Impurity Metrics
def region_gini_index(region):
    classes, num_per_class = np.unique(region, return_counts=True)
    pmk = num_per_class / len(region)
    return np.sum(pmk * (1 - pmk))

def next_split(train_data, features, min_samples_per_leaf):
    split_feature = None
    split_value = None
    split_gini_index = float('inf')
    
    # Binary Splitting
    for feature in features:
        values = train_data[feature].unique()
        for value in values:
            if np.issubdtype(train_data[feature].dtype, np.number):
                left_region = train_data[train_data[feature] <= value]
                right_region = train_data[train_data[feature] > value]
            else:
                left_region = train_data[train_data[feature] == value]
                right_region = train_data[train_data[feature] != value]
            
            # Stopping Condition
            if len(left_region) < min_samples_per_leaf or len(right_region) < min_samples_per_leaf:
                # print("reach")
                continue
            
            left_gini_index = region_gini_index(left_region['output'])
            right_gini_index = region_gini_index(right_region['output'])
            total_gini_index = (len(left_region) / len(train_data)) * left_gini_index + (len(right_region) / len(train_data)) * right_gini_index
            
            if total_gini_index < split_gini_index:
                split_gini_index = total_gini_index
                split_feature = feature
                split_value = value
    
    return split_feature, split_value

# Recursive Tree Construction
def construct_tree(train_data, features, depth, max_depth, min_samples_per_leaf):
    # Stopping Condition
    if len(train_data['output'].unique()) == 1 or len(train_data) < min_samples_per_leaf or depth >= max_depth or not features:
        return train_data['output'].mode()[0]
    
    split_feature, split_value = next_split(train_data, features, min_samples_per_leaf)
    if split_feature is None:
        return train_data['output'].mode()[0]
    
    if np.issubdtype(train_data[split_feature].dtype, np.number):
        left_region = train_data[train_data[split_feature] <= split_value]
        right_region = train_data[train_data[split_feature] > split_value]
        condition_left = f"<= {split_value}"
        condition_right = f"> {split_value}"
    else:
        left_region = train_data[train_data[split_feature] == split_value]
        right_region = train_data[train_data[split_feature] != split_value]
        condition_left = f"{split_value}"
        condition_right = f"not {split_value}"
    
    remaining_features = [f for f in features if f != split_feature]
    
    return {split_feature: {
        condition_left: construct_tree(left_region, remaining_features, depth + 1, max_depth, min_samples_per_leaf),
        condition_right: construct_tree(right_region, remaining_features, depth + 1, max_depth, min_samples_per_leaf)
    }}

# Prediction
def predict_sample(sample, tree):
    if not isinstance(tree, dict):
        return tree  
    feature = next(iter(tree))  
    value = sample[feature]  
    
    for condition, subtree in tree[feature].items():
        if "<=" in condition and np.issubdtype(type(value), np.number):
            if value <= float(condition.split()[1]):
                return predict_sample(sample, subtree)
        elif ">" in condition and np.issubdtype(type(value), np.number):
            if value > float(condition.split()[1]):
                return predict_sample(sample, subtree)
        elif condition == str(value):  
            return predict_sample(sample, subtree)
        elif "not" in condition and str(value) != condition.split("not ")[1]:  
            return predict_sample(sample, subtree)
    return list(tree[feature].values())[0]

def predict(X, tree):
    return X.apply(lambda sample: predict_sample(sample, tree), axis=1)

def train_decision_tree(X, Y, max_depth=3, min_samples_per_leaf=1):
    train_data = X.copy()
    train_data['output'] = Y
    features = list(X.columns)
    return construct_tree(train_data, features, 0, max_depth, min_samples_per_leaf)

train_data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Income': ['High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'High'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No'],
    'Credit': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair'],
    'Buy': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
})


X_train = train_data.drop(columns=['Buy'])
Y_train = train_data['Buy']

decision_tree = train_decision_tree(X_train, Y_train, max_depth=3, min_samples_per_leaf=1)
print(decision_tree)

new_sample = pd.DataFrame({'Age': [42], 'Income': ['Low'], 'Student': ['No'], 'Credit': ['Excellent']})

prediction = predict(new_sample, decision_tree)

print("Decision Tree Prediction :- ")
print("New person will buy a computer :", prediction.values[0])
print()

# Bagging 10 different trees
def bagging(X, Y, num_trees, max_depth, min_samples_per_leaf):
    trees = []
    n = len(X)
    
    oob_prediction_error_count = 0  
    total_oob_prediction_count = 0  

    for i in range(num_trees):
        bootstrap_samples = np.random.choice(n, n, replace=True)
        oob_samples = [sample for sample in range(n) if sample not in bootstrap_samples]
        
        X_bootstrap, Y_bootstrap = X.iloc[bootstrap_samples], Y.iloc[bootstrap_samples]
        tree = train_decision_tree(X_bootstrap, Y_bootstrap, max_depth, min_samples_per_leaf)
        trees.append(tree)
        
        for sample in oob_samples:
            X_oob = X.iloc[sample:sample+1]  
            y_true = Y.iloc[sample]  
            y_pred = predict(X_oob, tree).iloc[0]  
            
            if y_pred != y_true:
                oob_prediction_error_count += 1  
            total_oob_prediction_count += 1  

    oob_error = oob_prediction_error_count / total_oob_prediction_count if total_oob_prediction_count > 0 else 0  
    # print(oob_prediction_error_count,total_oob_prediction_count)

    return trees, oob_error

bagging_trees, oob_error = bagging(X_train, Y_train, num_trees=10, max_depth=3, min_samples_per_leaf=1)

print(f"Out of Bag (OOB) Error for Bagging: {oob_error:.4f}\n")

def predict_bagged(X, trees):
    predictions = np.array([predict(X, tree) for tree in trees])
    # print(predictions.T)
    return pd.Series([pd.Series(row).mode()[0] for row in predictions.T])
bagging_prediction = predict_bagged(new_sample, bagging_trees)

print("Bagging Prediction :- ")
print("New person will buy a computer :", bagging_prediction.values[0])
print()


# Bagging 10 different trees but using only two random predictors while building the trees.
# Random Forest

def construct_tree_rf(train_data, all_features, m_features, depth, max_depth, min_samples_per_leaf):
    if len(np.unique(train_data['output'])) == 1 or depth >= max_depth or len(train_data) < min_samples_per_leaf:
        return train_data['output'].mode()[0]
    
    selected_features = np.random.choice(all_features, m_features, replace=False)
    split_feature, split_value = next_split(train_data, selected_features, min_samples_per_leaf)
    
    if split_feature is None:
        return train_data['output'].mode()[0]
    
    if np.issubdtype(train_data[split_feature].dtype, np.number):
        left_region = train_data[train_data[split_feature] <= split_value]
        right_region = train_data[train_data[split_feature] > split_value]
        condition_left = f"<= {split_value}"
        condition_right = f"> {split_value}"
    else:
        left_region = train_data[train_data[split_feature] == split_value]
        right_region = train_data[train_data[split_feature] != split_value]
        condition_left = f"{split_value}"
        condition_right = f"not {split_value}"
    
    
    return {split_feature: {
        condition_left: construct_tree_rf(left_region, all_features,m_features, depth + 1, max_depth, min_samples_per_leaf),
        condition_right: construct_tree_rf(right_region, all_features,m_features, depth + 1, max_depth, min_samples_per_leaf)
    }}

def train_tree_rf(X, Y, m_features=2 ,max_depth=3, min_samples_per_leaf=1):
    train_data = X.copy()
    train_data['output'] = Y
    features = list(X.columns)
    return construct_tree_rf(train_data, features, m_features, 0, max_depth, min_samples_per_leaf)

def random_forest(X, Y, num_trees=10, max_depth=3, min_samples_per_leaf=1, m_features=2):
    trees = []
    n = len(X)  
    
    oob_prediction_error_count = 0  
    total_oob_prediction_count = 0  

    for i in range(num_trees):
        bootstrap_samples = np.random.choice(n, n, replace=True)
        oob_samples = [sample for sample in range(n) if sample not in bootstrap_samples]
        
        X_bootstrap, Y_bootstrap = X.iloc[bootstrap_samples], Y.iloc[bootstrap_samples]
        tree = train_tree_rf(X_bootstrap,Y_bootstrap,m_features,max_depth,min_samples_per_leaf)
        trees.append(tree)
        # print(tree)
        
        for sample in oob_samples:
            X_oob = X.iloc[sample:sample+1]  
            y_true = Y.iloc[sample]  
            y_pred = predict(X_oob, tree).iloc[0]  
            
            if y_pred != y_true:
                oob_prediction_error_count += 1  
            total_oob_prediction_count += 1  

    oob_error = oob_prediction_error_count / total_oob_prediction_count if total_oob_prediction_count > 0 else 0  
    # print(oob_prediction_error_count,total_oob_prediction_count)

    return trees, oob_error

forest, oob_error_random = random_forest(X_train, Y_train, num_trees=10, max_depth=3, min_samples_per_leaf=1, m_features=2)
print(f"Out of Bag (OOB) Error for Random Forest: {oob_error_random:.4f}\n")

bagging_prediction_random = predict_bagged(new_sample, forest)

print("Random Forest Prediction :- ")
print("New person will buy a computer :", bagging_prediction_random.values[0])
print()


