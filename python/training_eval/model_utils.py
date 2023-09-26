from sklearn.model_selection import cross_validate, LeaveOneGroupOut


def evaluate_model(validation, y_train, y_preds):
    # Evaluate predictions
    if validation is not isinstance(validation, LeaveOneGroupOut):
        results = cross_validate(model, X_train, y_train, cv=validation, scoring)
    else:
        for i, (train_index, test_index) in enumerate(validation.split(X, y, groups)):
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}, group={groups[train_index]}")
            print(f"  Test:  index={test_index}, group={groups[test_index]}")
            return results