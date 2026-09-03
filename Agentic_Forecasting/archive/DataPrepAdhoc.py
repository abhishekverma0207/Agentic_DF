# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

df=pd.read_csv('../JJ_Folder/Hair_Care_TH.csv')

# COMMAND ----------

import pandas as pd

def collapse_indicator_columns(
    df: pd.DataFrame,
    indicator_cols: list[str],
    out_col: str = "promo_instore_secondary_mechanic",
    prefix: str | None = "promo_instore_secondary_mechanic_",
    sep: str = "_",
    none_value: str | None = None,
) -> pd.DataFrame:
    """
    Collapse multiple 0/1 indicator columns into a single string column.

    - For each row, picks the suffix (mechanic name) of columns where value == 1.
    - If multiple are 1, concatenates them using `sep`.
    - If none are 1, sets `none_value` (default None -> NaN/None).

    Returns df (same object) with the new column added.
    """

    # Ensure we only work on columns that exist
    cols = [c for c in indicator_cols if c in df.columns]
    if not cols:
        raise ValueError("None of the provided indicator_cols exist in the dataframe.")

    # Extract labels from column names
    if prefix is None:
        labels = cols[:]  # keep full names
    else:
        labels = [c[len(prefix):] if c.startswith(prefix) else c for c in cols]

    # Work with a boolean mask of "is active"
    active = df[cols].fillna(0).astype(int).eq(1)

    # Build collapsed string efficiently row-wise
    # (uses list-comprehension over numpy arrays; fast enough for most cases)
    label_arr = pd.Index(labels).to_numpy()
    mask_arr = active.to_numpy()

    collapsed = [
        (sep.join(label_arr[row_mask]) if row_mask.any() else none_value)
        for row_mask in mask_arr
    ]

    df[out_col] = collapsed
    return df


# COMMAND ----------

df=pd.read_csv('./sourcedata/DEO_US.csv')
#df.columns=[str(x).lower() for x in df.columns]

# COMMAND ----------

df['year_week']=df.year_week.apply(lambda x: str(x).replace('-',''))

# COMMAND ----------

holiday_cols=['Anniversary_of_the_Death_of_King_Bhumibol',	'Asalha_Bucha',	'Buddhist_Lent_Day',	
              'Chakri_Day',	'Chinese_Lunar_New_Year_Festive',	'Christmas_&_New_Year_Festive',	'Chulalongkorn_Day',	
              'Constitution_Day',	'Coronation_Day',	'King_Bhumibols_Birthday',	'King_Vajiralongkorns_Birthday',	'Labor_Day',	'Makha_Bucha',
              'Queen_Suthidas_Birthday',	'Royal_Ploughing_Ceremony_Day',	'Songkran_Festive',	'Special_Holiday',	
              'Special_public_holiday',	'The_Queens_Birthday',	'Visakha_Bucha']
season=['Summer',	'Winter',	'Monsoon','Quarter_start',	'Quarter_end',	'Month_End']
promo=['bogo',	'buyafreea',	'buyafreeb',	'buyafreeuttproductaorbmt',	
       'buybahtdiscount',	'buybahtdiscountuttunitmonthly',	
       'buybahtdiscountuttunitweekly',	'buybahtfreeutt',	
       'buymultipleskusdiscountbaht',	'buyunittwondpcsdiscount',	
       'buyunittwondpcsdiscountbaht',	'clearance',	
       'coupon','laksuebiweekly',	'laksuemonthly',	'laksuetriweekly',	'laksueweekly',	'others',
       'stamppoint',	
       'twoforbiweekly',	'twoformonthly',	'twofortriweekly',	
       'twoforweekly',	'twofree1',	'twondpcs1b',	'twondpcs50',	
       'wdt',	'wedlp',	
       'wedlplmt',	'wlmt',	'wmt',	
       'wspecialpack',	'wspecialpacklmt',	'wspecialpackmt',	
       'wsschemeafreea',	'wsschemeafreeb',	'xfor',
       'xmultipackdiscount','promo_seq_num'
       ]
price_promo=['priceoffdt',	'priceofflmt',	'priceoffmt5day',	
       'priceoffmtbiweekly',	'priceoffmtmonthly',	
       'priceoffmttriweekly',	'priceoffmtweekly']

other_combo=['combosetfixskustotalvolumediscount',	'bogowoninv',	'twondpcs1bwoninv',	'twoforwoninv',	'laksuewoninv',	'priceoffwoninv',	
             'twofree1woninv',	'Post_Promotion_Flag',	'Baseline_Flag',	'priceoffoutlettypeoutletlevel']



# COMMAND ----------



# COMMAND ----------

import pandas as pd

def collapse_indicators(df, cols, new_col, sep="|"):
    """
    Creates a single column by concatenating indicator column names
    where the indicator == 1.
    """
    df[new_col] = (
        df[cols]
        .astype(bool)
        .apply(lambda row: sep.join(row.index[row]), axis=1)
    )
    # Optional: replace empty string with None / NaN
    df[new_col] = df[new_col].replace("", pd.NA)
    return df


# COMMAND ----------

df = collapse_indicators(df, holiday_cols, "holiday_type")
df = collapse_indicators(df, season, "season_type")
df = collapse_indicators(df, promo, "promo_type")
df = collapse_indicators(df, price_promo, "price_promo_type")
df = collapse_indicators(df, other_combo, "other_combo_type")

# COMMAND ----------

df.to_csv('../JJ_Folder/Hair_Care_TH.csv',index=False)

# COMMAND ----------

len(df)

# COMMAND ----------

prefixes=['promo_primary_mechanic_','promo_primary_objective_','promo_secondary_objective_','promo_feature_','promo_secondary_mechanic_',
          'promo_instore_primary_mechanic_','promo_instore_primary_objective_','promo_instore_secondary_objective_','promo_instore_feature_',
          'promo_instore_secondary_mechanic_','holiday_','season_']

out_col=['promo_primary_mechanic','promo_primary_objective','promo_secondary_objective','promo_feature','promo_secondary_mechanic',
          'promo_instore_primary_mechanic','promo_instore_primary_objective','promo_instore_secondary_objective','promo_instore_feature',
          'promo_instore_secondary_mechanic','holiday','season']

# COMMAND ----------

for pre_ in prefixes:
    print ('running for ---'+str(pre_))
    indicator_cols = [c for c in df.columns if c.startswith(str(pre_))]
    out_col_name=out_col[prefixes.index(pre_)]
    df = collapse_indicator_columns(
        df,
        indicator_cols=indicator_cols,
        out_col=out_col_name,
        prefix=pre_,
        sep="_",
        none_value='NA',   # or "None"/"" if you prefer
    )


# COMMAND ----------

pd.DataFrame(df.columns.values).to_clipboard()

# COMMAND ----------

for c in df.category_name.unique():
    __tmp=df[df.category_name==str(c)].copy(deep=True)
    __tmp.to_csv('./sourcedata/'+str(c).replace(' ','').upper()+'_data.csv',index=False)

# COMMAND ----------

df.to_csv('./sourcedata/DEO_US.csv',index=False)

# COMMAND ----------

df[df.actual_sales>0].year_week.min()

# COMMAND ----------

sorted(df.year_week.unique())

# COMMAND ----------

df.to_csv('./sourcedata/raw_data_filtered_cols_collapsed_skincleansing.csv',index=False)

# COMMAND ----------

df_deo=pd.read_csv('./artifacts_base_US_Deo/backtest_output/backtest_forecasts.csv')
df_deo['Category']='Deo Male Toiletries'
df_pw=pd.read_csv('./artifacts_base_US_Personal_Wash/backtest_output/backtest_forecasts.csv')
df_pw['Category']='Personal Wash'
all_dat=pd.concat([df_pw,df_deo])
all_dat.head()

# COMMAND ----------

all_dat['GTIN']=all_dat['key'].apply(lambda x: str(x)[-14:])

# COMMAND ----------

summary=all_dat[(all_dat.is_dead_key==False) & (all_dat.is_new_key==False) & (all_dat.lag==4)].groupby(['Category','GTIN','is_new_key','is_dead_key','year_week','forecast_step','lag'])[['predicted','actual']].sum().reset_index()

# COMMAND ----------

summary['ABS']=abs(summary['predicted']-summary['actual'])
summary_=summary[(summary.year_week>=202533.0) & (summary.year_week<=202536.0)].copy(deep=True)
1-(summary[summary.lag==4].groupby(['Category'])['ABS'].sum()/summary[summary.lag==4].groupby(['Category'])['actual'].sum())


# COMMAND ----------

summary[summary.lag==4].groupby(['Category'])['actual'].sum()