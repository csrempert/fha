import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add the directory containing this file (app.py) to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from datetime import datetime

current_year = datetime.now().year
print(current_year)

######################
# CONFIG & CONSTANTS #
######################

##########################################
# STEP 1: READ & CLEAN THE TRANSACTIONS  #
##########################################

def assign_fiscal_year(row, start_month: int) -> int:
    """
    Assigns the fiscal year based on the user’s 'start_month',
    labeling by the start of that fiscal year.
    """
    y = row['YEAR']
    m = row['MONTH']
    if start_month == 1:
        return y
    return y if m >= start_month else (y - 1)

def clean_data(
    df: pd.DataFrame,
    column_map: dict,
    max_date: pd.Timestamp = None,
    min_date: pd.Timestamp = None,
    max_amount: float = None,
    fy_start_month: int = 1
) -> pd.DataFrame:
    """
    Reads transaction data, applies cleaning/filters, and assigns FY.
    """
    rename_dict = {
        column_map['id']: 'ID',
        column_map['amount']: 'AMOUNT',
        column_map['date']: 'DATE'
    }
    df = df.rename(columns=rename_dict)

    required_cols = ['ID', 'AMOUNT', 'DATE']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing: {missing}.\n"
            f"Please re-check your column_map: {column_map}\n"
            f"DataFrame columns are: {list(df.columns)}"
        )

    # Parse date and amount
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['AMOUNT'] = (
        df['AMOUNT']
        .astype(str)
        .str.replace('[$,]', '', regex=True)
        .astype(float)
    )

    # Drop rows missing ID, DATE, or AMOUNT
    df = df.dropna(subset=['ID', 'DATE', 'AMOUNT']).reset_index(drop=True)

    # Filter by optional date range
    if min_date is not None:
        df = df[df['DATE'] >= min_date]
    if max_date is not None:
        df = df[df['DATE'] <= max_date]

    # Filter out donations above max_amount
    if max_amount is not None:
        df = df[df['AMOUNT'] <= max_amount]

    # Create YEAR, MONTH, DAY
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day

    # Compute FY
    df['FY'] = df.apply(lambda r: assign_fiscal_year(r, fy_start_month), axis=1)

    return df


##########################
# STEP 2: HELPER METHODS #
##########################

def create_output(name: str, data: pd.DataFrame) -> dict:
    """
    Example aggregator for demonstration.
    """
    output = {'name': name}
    output['counts'] = (
        data.groupby(['YEAR','MONTH']).size().reset_index(name='n')
    )
    distinct_df = data.drop_duplicates(subset=['ID','YEAR','MONTH'])
    output['counts_unique'] = (
        distinct_df.groupby(['YEAR','MONTH']).size().reset_index(name='n')
    )
    totals_df = (
        data.groupby(['YEAR','MONTH'])['AMOUNT']
        .sum()
        .reset_index(name='TOTAL')
    )
    output['totals'] = totals_df
    return output

def get_diff(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate # of days (in years) between consecutive gifts per donor.
    """
    data = data.sort_values(['ID','DATE'])
    data['Diff'] = data.groupby('ID')['DATE'].diff().dt.days / 365.0
    return data


#########################
# STEP 3: CATEGORY FUNS #
#########################

def getNewSingle(df: pd.DataFrame, data_only: bool=False) -> pd.DataFrame or dict:
    """
    Return each donor's very first gift record.
    """
    name = 'new'
    new_data = df.sort_values(by=['ID','DATE']).groupby('ID', group_keys=False).head(1)
    if data_only:
        return new_data
    else:
        return create_output(name, new_data)

def getLapsedSingle(df: pd.DataFrame, data_only: bool=False) -> pd.DataFrame or dict:
    """
    Example 'lapsed' logic: if time between consecutive gifts is > 1 year.
    """
    name = 'lapsed'
    new_data = df.copy()
    new_data = new_data.sort_values(by=['ID','DATE'])
    new_data = get_diff(new_data)
    new_data = new_data[new_data['Diff'] > 1]
    if data_only:
        return new_data
    else:
        return create_output(name, new_data)


#################################
# STEP 4: HIGH-LEVEL SUBSET FUN #
#################################

def getAllDonors(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Returns all donors who have given on or before the specified FY.
    """
    new_df = df.copy().sort_values(by=['ID','DATE'])
    mask = new_df.groupby('ID')['FY'].transform(lambda col: (col <= fy).any())
    return new_df[mask].reset_index(drop=True)

def getNewDonorsYear(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors whose first-ever gift is in the specified FY.
    """
    new_ids = getNewSingle(df, data_only=True)
    new_ids = new_ids[new_ids['FY'] == fy]['ID'].unique()
    return df[df['ID'].isin(new_ids)].reset_index(drop=True)

def getNewDonorsYearPrevious(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors whose first-ever gift was in the prior FY.
    """
    new_ids = getNewSingle(df, data_only=True)
    new_ids = new_ids[new_ids['FY'] == (fy - 1)]['ID'].unique()
    return df[df['ID'].isin(new_ids)].reset_index(drop=True)

def getConsecutive(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Example logic: Donors who gave in both (fy-1) and (fy-2).
    """
    new_df = df.copy().sort_values(by=['ID','DATE'])
    mask = new_df.groupby('ID')['FY'].transform(
        lambda col: ((col == (fy - 1)).any() and (col == (fy - 2)).any())
    )
    return new_df[mask].reset_index(drop=True)

def getLapsed(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors who gave before (fy-1), but not in (fy-1).
    """
    new_df = df.copy().sort_values(by=['ID','DATE'])
    def condition(col):
        return (not (col == fy - 1).any()) and ((col < (fy - 1)).any())
    mask = new_df.groupby('ID')['FY'].transform(condition)
    return new_df[mask].reset_index(drop=True)

def getReinstatedLastYear(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors who gave in (fy-1), did NOT give in (fy-2),
    but had given at some point prior to (fy-2).
    """
    new_df = df.copy().sort_values(by=['ID','DATE'])
    def condition(col):
        return ((col == fy - 1).any()) and (not (col == fy - 2).any()) and ((col < (fy - 2)).any())
    mask = new_df.groupby('ID')['FY'].transform(condition)
    return new_df[mask].reset_index(drop=True)


##############################################
# STEP 5: METRICS (getAnnualBreakdowns in R) #
##############################################

def getAnnualBreakdowns(df: pd.DataFrame, fy: int) -> dict:
    """
    Basic metrics for the given subset, in the given FY.
    """
    output = {}
    population = df['ID'].nunique()
    df_fy = df[df['FY'] == fy]
    donors_this_fy = df_fy['ID'].nunique()
    total_revenue_this_fy = df_fy['AMOUNT'].sum()
    gifts_this_fy = len(df_fy)
 
    output['population'] = population
    output['donors'] = donors_this_fy
    output['totalRevenue'] = total_revenue_this_fy
    output['gifts'] = gifts_this_fy
 
    output['valuePerDonor'] = (
        total_revenue_this_fy / donors_this_fy if donors_this_fy != 0 else 0
    )
    output['activation'] = (
        donors_this_fy / population if population != 0 else 0
    )
    output['giftsPerDonor'] = (
        gifts_this_fy / donors_this_fy if donors_this_fy != 0 else 0
    )
    output['avgGift'] = (
        total_revenue_this_fy / gifts_this_fy if gifts_this_fy != 0 else 0
    )
 
    return output


###############################
# STEP 6: OPTIONAL ROLLING    #
###############################

def toRolling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Example rolling-year shift logic.
    """
    max_date = data['DATE'].max()
    new_end_date = pd.to_datetime(f"{max_date.year - 1}-12-31")
    diff = max_date - new_end_date
 
    data_rolled = data.copy()
    data_rolled['DATE'] = data_rolled['DATE'] - diff
 
    data_rolled['YEAR'] = data_rolled['DATE'].dt.year
    data_rolled['MONTH'] = data_rolled['DATE'].dt.month
    data_rolled['DAY'] = data_rolled['DATE'].dt.day
 
    data_rolled['FY'] = data_rolled['YEAR']
    return data_rolled


################################
# STEP 7: MAIN RUN FUNCTION(S)  #
################################

def runAll(
    data: pd.DataFrame,
    fy_list: list[int]
) -> dict:
    """
    Runs category breakdowns for each FY in fy_list,
    returns a dictionary of DataFrames keyed by stringified FY.
    """
    results = {}
    df = data.copy()
 
    for fy in fy_list:
        temp = {}
        temp['allDon']      = getAllDonors(df, fy)
        temp['new']         = getNewDonorsYear(df, fy)
        temp['newLast']     = getNewDonorsYearPrevious(df, fy)
        temp['consecutive'] = getConsecutive(df, fy)
        temp['lapsed']      = getLapsed(df, fy)
        temp['reLast']      = getReinstatedLastYear(df, fy)
 
        # Compute metrics for each subset
        breakdowns = {}
        for category_name, subset_df in temp.items():
            metrics = getAnnualBreakdowns(subset_df, fy)
            breakdowns[category_name] = metrics
 
        breakdown_df = pd.DataFrame.from_dict(breakdowns, orient='index')
        results[str(fy)] = breakdown_df
 
    return results


##########################
# NEW: COMBINE RESULTS   #
##########################

def combine_results(results: dict, rolling_results: dict, last_fy: int) -> pd.DataFrame:
    """
    Takes the per-FY results dict plus the rolling results dict,
    and merges them into a single wide table:
       Segment | Metric | FYyyyy | FYyyyy | ... | Rolling
    """
    frames = []
    
    # 1) Melt each FY's DataFrame
    for fy_str, df_out in results.items():
        df_temp = df_out.copy()
        df_temp['segment'] = df_temp.index
        df_melt = df_temp.melt(id_vars='segment', var_name='metric', value_name='value')
        df_melt['FY'] = "FY" + fy_str  # e.g. "FY2021"
        frames.append(df_melt)
    
    # 2) Melt the rolling results (only 1 FY key in rolling_results)
    if str(last_fy) in rolling_results:
        df_roll = rolling_results[str(last_fy)].copy()
        df_roll['segment'] = df_roll.index
        df_melt_roll = df_roll.melt(id_vars='segment', var_name='metric', value_name='value')
        df_melt_roll['FY'] = "Rolling"
        frames.append(df_melt_roll)
    
    # Combine all melted frames
    df_all = pd.concat(frames, ignore_index=True)

    # Pivot so that rows are [segment, metric], columns are FY, values = 'value'
    df_pivot = df_all.pivot_table(
        index=['segment','metric'],
        columns='FY',
        values='value'
    )
    
    # Sort columns so that FYs appear in ascending order, Rolling at the end
    def sort_key(col):
        return 999999 if col == "Rolling" else int(col.replace("FY", ""))
    sorted_cols = sorted(df_pivot.columns, key=sort_key)
    df_pivot = df_pivot[sorted_cols]
    
    # Reset index so that segment and metric are columns
    df_final = df_pivot.reset_index()

    # (1) Rename the segments for readability
    segment_mapping = {
        'new':      'New Donors',
        'newLast':  'New Previous Year',
        'consecutive': 'Consecutive',
        'reLast':   'Reinstated Last Year',
        'lapsed':   'Lapsed',
        'allDon':   'All Donors'
    }
    df_final['segment'] = df_final['segment'].replace(segment_mapping)

    # (2) Force the segments to appear in the desired order
    desired_order = [
        'New Donors',
        'New Previous Year',
        'Consecutive',
        'Reinstated Last Year',
        'Lapsed',
        'All Donors'
    ]
    df_final['segment'] = pd.Categorical(df_final['segment'], categories=desired_order, ordered=True)
    df_final = df_final.sort_values(by=['segment','metric'])

    return df_final


####################
# EXECUTION EXAMPLE
####################

if __name__ == "__main__":
    print("fha.py: Updated version with single combined table output and custom segment order.")

##########################
# STREAMLIT FRONT END
##########################

col1, col2, col3, col4, col5 = st.columns(5)
col3.image("IS-Logo_RGB_Vertical-onWhite.png", width=112)
st.title("Donor File Health Analysis Portal")

# 1) File Upload
uploaded_file = st.file_uploader("Upload donor transaction file and run analysis.", type=["csv"])

df_uploaded = None
column_map = {}

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        df_uploaded = pd.read_csv(uploaded_file)
    else:
        df_uploaded = pd.read_excel(uploaded_file)

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # 2) Column Mapping Step
    st.subheader("Map Your Columns")

    columns_in_file = df_uploaded.columns.tolist()

    id_col = st.selectbox("Select the column for Donor ID", columns_in_file)
    amount_col = st.selectbox("Select the column for Amount", columns_in_file)
    date_col = st.selectbox("Select the column for Date", columns_in_file)

    column_map = {
        'id': id_col,
        'amount': amount_col,
        'date': date_col
    }

    if (id_col == amount_col) or (id_col == date_col) or (amount_col == date_col):
        st.error("You have selected the same column for multiple required fields (ID, Amount, Date).")
        st.stop()

    st.write("Column mapping complete. Proceed with analysis parameters below.")
else:
    st.warning("Please upload a file to begin. (CSV only)")


# 3) Parameter Inputs
st.subheader("Analysis Parameters")

fy_format_options = {
    "January–December": 1,
    "April–March": 4,
    "July–June": 7,
    "October–September": 10
}
fy_choice = st.selectbox(
    "Select Fiscal Year Format",
    list(fy_format_options.keys())
)
fy_start_month = fy_format_options[fy_choice]

years_available = list(range(2018, 2026))
selected_fys = st.multiselect(
    "Select which Fiscal Years to include in the output",
    years_available,
    default=[2021, 2022, 2023, 2024]
)

analysis_start_date = st.date_input("Filter Start Date (optional)", value=None)
analysis_end_date = st.date_input("Filter End Date (optional)", value=None)

max_donation_cutoff = st.number_input(
    "Max Donation Amount (exclude donations above this amount)",
    value=10000,
    step=1000
)

run_analysis_button = st.button("Run Analysis")

if run_analysis_button:
    if df_uploaded is None:
        st.warning("No file uploaded. Please upload a file first.")
    else:
        start_date = pd.to_datetime(analysis_start_date) if analysis_start_date else None
        end_date = pd.to_datetime(analysis_end_date) if analysis_end_date else None

        st.write("Processing your file. Sit tight! This may take a while.")

        progress_bar = st.progress(0)

        try:
            trx = clean_data(
                df_uploaded,
                column_map=column_map,
                max_date=end_date,
                min_date=start_date,
                max_amount=max_donation_cutoff,
                fy_start_month=fy_start_month
            )
        except ValueError as e:
            st.error(f"Column mapping error: {e}")
            st.stop()

        if not selected_fys:
            st.error("Please select at least one Fiscal Year to include in the output.")
            st.stop()

        progress_bar.progress(20)

        # 2) Run analysis
        results = runAll(trx, fy_list=sorted(selected_fys))
        progress_bar.progress(50)

        # 3) Rolling analysis for the last selected FY
        last_fy = sorted(selected_fys)[-1]
        rolled_data = toRolling(trx)
        rolling_results = runAll(rolled_data, [last_fy])

        progress_bar.progress(70)

        # 4) Combine everything into a single table
        final_df = combine_results(results, rolling_results, last_fy)
        final_csv = final_df.to_csv(index=False)

        # 5) Single CSV download
        st.download_button(
            label="Download Combined Analysis (CSV)",
            data=final_csv,
            file_name="analysis_results.csv",
            mime="text/csv"
        )
        st.success("Analysis complete! Download your single-table results above.")

        progress_bar.progress(100)

        # (Optional) display final table in-app
        st.subheader("Combined Table Preview")
        st.dataframe(final_df)