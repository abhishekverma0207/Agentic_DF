"""Config for the FR (France) stacked/segment forecasting engine.

Mirrors config_de_base_local.py — same DE pipeline approach (segment × category × LEGO
3-model blend via segment_engine), adapted for France data.

France uses the same DT/FS/TF structure as DE, so no data preparation step is needed.
The benchmark subfolder is TF (not TF_pristine_allkeys like DE).

Loaded via utils.config_loader.load_market_config("fr").
"""

CONFIG = {
    'target': 'NonPromoVolValue',
    'key_col': 'key',
    'time_col': 'year_week',
    'cat_col': 'LocalLevel4Name',
    'horizons': 13,

    # ---- market_io: mirrors DE structure for segment_engine routing ----
    'market_io': {
        'category_col': 'LocalLevel4Name',
        'cats_subdir': 'DT',
        'fs_subdir': 'FS',
        'lego_pred_col': 'prediction_rf',
        'lego_subdir': 'TF',
        'ensemble_weights': 'config/fr_ensemble_weights.yaml',
        'segment_col': 'lego_segment',
    },

    # ---- LightGBM hyperparameters (matching DE's validated settings) ----
    'objective': 'tweedie',
    'tweedie_variance_power': 1.3,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 700,
    'min_data_in_leaf': 100,
    'lambda_l2': 1.0,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'max_depth': -1,
    'recency_halflife_weeks': 39,
    'seed': 42,
    'num_threads': 0,

    # ---- feature engineering (backward features computed dynamically) ----
    'lags': [1, 2, 3, 4, 5, 6, 8, 13, 26, 52],
    'roll_windows': [4, 8, 13, 26],

    # Extra signal columns for lagged features.
    'extra_signal_cols': ['smooth_sales', 'actual_dispatched_quantity',
                          'customer_pos_sales_in_cases'],

    # Prefixes for future-known covariates.
    # _future_cols picks numeric columns starting with any of these.
    'future_cov_prefixes': [
        # Promo
        'promo_indicator',
        # French holidays (raw flags in DT)
        'All_Saints_Day',
        'Armistice_Day',
        'Ascension_Day',
        'Assumption_of_Mary',
        'Bastille_Day',
        'Christmas_Day',
        'Christmas_Eve',
        'Daylight_Saving_Time',
        'December_Solstice',
        'Easter_Monday',
        'Easter_Sunday',
        'Fathers_Day',
        'Good_Friday',
        'June_Solstice',
        'Labor_Day',
        'March_Equinox',
        'Mothers_Day',
        'New_Years_Day',
        'New_Years_Eve',
        'September_Equinox',
        'St_Stephens_Day',
        'WWII_Victory_Day',
        'Whit_Monday',
        'Whit_Sunday',
        'holiday_week',
        # Season
        'Spring',
        'Summer',
        'Autumn',
        'Winter',
        # Weather
        'avgTemp',
        'Prcp',
        'Sunhours',
        'solarRadiation',
        # Momentum
        'momentum_',
        # Price
        'NIV_CPP',
        # LEGO's f_* engineered features (from FS join)
        'f_',
    ],

    # Categorical features (label-encoded for LightGBM)
    'cat_features': ['LocalLevel4Name', 'BrandDescription', 'APG_Name',
                     'ForecastFamilyDescription'],

    'rich_features': False,
    'input_source': 'dbx',
    'collapse_promo': False,
    'drop_promo_onehots': True,
    'holiday_distance_features': False,
    'seasonal_anchor': 'adaptive',
    'seasonal_lo': 0.5,
    'seasonal_hi': 2.0,
    'bias_calibration': False,
    'bias_calibration_holdout_weeks': 6,
    'bias_calibration_damp': 0.7,
    'calibrate_categories': [],
    'bias_factor_min': 0.5,
    'bias_factor_max': 1.5,
    'per_category': False,
    'per_category_categories': None,
    'min_train_weeks': 8,
    'train_sample_frac': 1.0,
    'clip_negatives': True,

    # ---- weight_fit: mirrors DE_FitWeights parameters ----
    'weight_fit': {
        'train_snapshots': ['202603', '202604', '202607', '202612'],
        'live_snapshot': '202616',
        'parent_dir_template': '/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output/{snapshot}',
        'lego_path_template': '/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output/{snapshot}/TF',
        'actuals_source': '/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output/202629/DT',
        'components_out_dir': '/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output/_diq_fit/components',
        'live_parent_dir': '/Volumes/pds_feu_931272_dev/eu_france/transform/FR/DIQ_Test/lego_output/202616',
        'live_history_till': '202615',
        'horizons': 13,
        'grain': 'gtin',
        'objective': 'wape_asym',
        'bias_tolerance': 0.02,
        'guardrail': False,
        'guardrail_margin': 0.01,
        'calibrate': False,
        'in_scope_categories': [
            'CONDIMENT',
            'COOKING AID',
            'DEODORANT & FRAGRANCE',
            'FABRIC CLEANING',
            'FABRIC ENHANCER',
            'HAIR CARE',
            'HOME & HYGIENE',
            'ICE CREAM CATEGORY',
            'MINI MEAL',
            'ORAL CARE',
            'OTH FOOD',
            'PLANT BASED MEAT',
            'SKIN CLEANSING',
        ],
    },
}
