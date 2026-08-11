class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        dt = pd.DataFrame(self.df.iloc[a:b])
        input_data(data=dt[self.features], label=dt[self.target]) #, weight=dt['weight'])
        self.it += 1
        return 1

class XGBBoosterWrapper:
    def __init__(self, booster, features):
        self.booster = booster
        self.features = features
    def predict_proba(self, X):
        d = xgb.DMatrix(X[self.features])
        p = self.booster.predict(d)
        return np.vstack([1 - p, p]).T

def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def train_xgb_cv(df,
    FEATURES,
    xgb_parms,
    target_col='target',
    id_col='customer_ID',
    n_splits=5,
    train_subsample=1.0,
    model_version=1,
    num_boost_round=9999,
    early_stopping_rounds=100,
    verbose_eval=1000,
    max_bin=256,
    random_state=42,
    save_dir='/kaggle/working/train_model'
):
    importances = []
    oof = []
    os.makedirs(save_dir, exist_ok=True)
    
    gc.collect()
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_df, train_df[target_col])
    ):
        if train_subsample < 1.0:
            np.random.seed(random_state)
            train_idx = np.random.choice(
                train_idx,
                int(len(train_idx) * train_subsample),
                replace=False,
            )
            np.random.seed(None)

        print('#' * 25)
        print('### Fold', fold + 1)
        print('### Train size', len(train_idx), 'Valid size', len(valid_idx))
        print(f'### Training with {int(train_subsample * 100)}% fold data...')
        print('#' * 25)

        Xy_train = IterLoadForDMatrix(
            train_df.loc[train_idx], FEATURES, target_col
        )
        X_valid = train_df.loc[valid_idx, FEATURES]
        y_valid = train_df.loc[valid_idx, target_col]

        dtrain = xgb.QuantileDMatrix(Xy_train, max_bin=max_bin)
        dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

        model = xgb.train(
            xgb_parms,
            dtrain=dtrain,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        model.save_model(f'{save_dir}/XGB_v{model_version}_fold{fold}.ubj')

        dd = model.get_score(importance_type='weight')
        imp_df = pd.DataFrame({'feature': dd.keys(), f'importance_{fold}': dd.values()})
        importances.append(imp_df)

        oof_preds = model.predict(dvalid)
        acc = amex_metric_mod(y_valid.values, oof_preds)
        print('Kaggle Metric =', acc, '\n')

        fold_df = train_df.loc[valid_idx, [id_col, target_col]].copy()
        fold_df['oof_pred'] = oof_preds
        oof.append(fold_df)

        del dtrain, Xy_train, dd, imp_df
        del X_valid, y_valid, dvalid, model
        _ = gc.collect()

    print('#' * 25)
    oof = pd.concat(oof, axis=0, ignore_index=True).set_index(id_col)
    overall_score = amex_metric_mod(oof[target_col].values, oof.oof_pred.values)
    print('OVERALL CV Kaggle Metric =', overall_score)
    _ = gc.collect()

    importances_df = importances[0]
    for imp_df in importances[1:]:
        importances_df = importances_df.merge(imp_df, on='feature', how='outer')

    return oof, importances_df, overall_score

def rfe(df, id_col, target_col, xgb_parms):
    FEATURES = [c for c in df.columns if c not in {id_col, target_col}]
    current_features = list(FEATURES)
    feature_count_history = []
    score_history = []
    dropped_history = []
    feature_set_history = []

    MIN_FEATURES_TO_KEEP = 30
    TRAIN_SUBSAMPLE = 1.0

    FINE_TUNE_THRESHOLD = 50
    COARSE_DROP_COUNT = 10
    FINE_DROP_COUNT = 1

    print(f"Starting Recursive Feature Elimination. Initial features: {len(current_features)}")
    print(f"Loop will drop {COARSE_DROP_COUNT} features per round until {FINE_TUNE_THRESHOLD} features remain, "
          f"then drop {FINE_DROP_COUNT} at a time down to {MIN_FEATURES_TO_KEEP}.")
    print("=" * 60)

    round_num = 0

    while len(current_features) >= MIN_FEATURES_TO_KEEP:
        round_num += 1
        print(f"\n>>> ROUND {round_num} WITH {len(current_features)} FEATURES <<<")
        round_dir = os.path.join('/kaggle/working/backward_elimination', f"round{round_num}")

        feature_set_history.append(list(current_features))

        oof_round, importances_round, round_score = train_xgb_cv(
            df=train_df,
            FEATURES=current_features,
            xgb_parms=xgb_parms,
            n_splits=5,
            train_subsample=TRAIN_SUBSAMPLE,
            model_version=f'rfe_round{round_num}',
            num_boost_round=2000,
            early_stopping_rounds=100,
            verbose_eval=False,
            max_bin=256,
            random_state=42,
            save_dir=round_dir,
        )

        score_history.append(round_score)
        feature_count_history.append(len(current_features))
        print(f"Overall CV with {len(current_features)} features: {round_score:.5f}")

        if len(current_features) <= MIN_FEATURES_TO_KEEP:
            break

        imp_cols = [c for c in importances_round.columns if c.startswith('importance_')]
        importances_round[imp_cols] = importances_round[imp_cols].fillna(0)
        importances_round['importance'] = importances_round[imp_cols].mean(axis=1)

        seen_features = set(importances_round['feature'])
        missing_features = [f for f in current_features if f not in seen_features]
        if missing_features:
            missing_df = pd.DataFrame({'feature': missing_features, 'importance': 0.0})
            importance_df = pd.concat(
                [importances_round[['feature', 'importance']], missing_df],
                ignore_index=True,
            )
        else:
            importance_df = importances_round[['feature', 'importance']]

        importance_df = importance_df.sort_values('importance', ascending=True)

        step_size = FINE_DROP_COUNT if len(current_features) <= FINE_TUNE_THRESHOLD else COARSE_DROP_COUNT
        features_to_drop_count = min(step_size, len(current_features) - MIN_FEATURES_TO_KEEP)
        lowest_features = importance_df.head(features_to_drop_count)['feature'].tolist()

        print(f"Removing bottom {features_to_drop_count} feature(s) "
              f"[{'fine-tune mode' if step_size == FINE_DROP_COUNT else 'coarse mode'}]: {lowest_features}")
        dropped_history.append(lowest_features)

        for feat in lowest_features:
            current_features.remove(feat)

    print("\n" + "=" * 60)
    print("Backward Elimination Finished!")

    dropped_history.append(["None - Final Set"])
    summary_df = pd.DataFrame({
        'num_features': feature_count_history,
        'cv_score': score_history,
        'dropped_this_round': [", ".join(x) for x in dropped_history]
    })

    print("\nSummary of the Feature Reduction Loop:")
    print(summary_df.to_string(index=False))

    best_round_idx = summary_df['cv_score'].idxmax()
    best_num = summary_df.loc[best_round_idx, 'num_features']
    best_score = summary_df.loc[best_round_idx, 'cv_score']
    best_features = feature_set_history[best_round_idx]
    print(f"\nOptimal configuration: Keep {best_num} features for a peak CV score of {best_score:.5f}")

    return best_features