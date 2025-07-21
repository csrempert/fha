import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
import io
import numpy as np
from openai import OpenAI # MODIFIED FOR OPENAI

# This logic is for local execution, may not be needed in all environments
# try:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     sys.path.insert(0, current_dir)
# except NameError:
#      pass # Fails when running in Streamlit cloud or similar

current_year = datetime.now().year

######################
# CONFIG & CONSTANTS #
######################

##########################################
# STEP 1: READ & CLEAN THE TRANSACTIONS  #
##########################################

def assign_fiscal_year(row, start_month: int) -> int:
    """
    Assign the fiscal year **ending** year based on the organization’s fiscal‑year
    start month.

    The fiscal-year label should always be the calendar year in which the FY
    ENDS.

    Examples
    --------
    • start_month = 1 (Jan–Dec):  2025‑02‑15 → FY 2025
    • start_month = 4 (Apr–Mar):  2024‑04‑10 → FY 2025
    • start_month = 7 (Jul–Jun):  2024‑07‑01 → FY 2025
    """
    y = row["YEAR"]
    m = row["MONTH"]

    # Jan–Dec fiscal years end in the same calendar year.
    if start_month == 1:
        return y

    # Months on/after the FY start belong to the FY that ends the *next* calendar year.
    return (y + 1) if m >= start_month else y

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

def getNewSingle(df: pd.DataFrame, data_only: bool=True) -> pd.DataFrame:
    """
    Return each donor's very first gift record.
    """
    return df.sort_values(by=['ID','DATE']).groupby('ID', group_keys=False).head(1)

#################################
# STEP 4: HIGH-LEVEL SUBSET FUN #
#################################

def getAllDonors(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Returns all donors who have given on or before the specified FY.
    """
    new_df = df.copy()
    mask = new_df.groupby('ID')['FY'].transform(lambda col: (col <= fy).any())
    return new_df[mask].reset_index(drop=True)

def getNewDonorsYear(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors whose first-ever gift is in the specified FY.
    """
    new_ids = getNewSingle(df)
    new_ids = new_ids[new_ids['FY'] == fy]['ID'].unique()
    return df[df['ID'].isin(new_ids)].reset_index(drop=True)

def getNewDonorsYearPrevious(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors whose first-ever gift was in the prior FY.
    """
    new_ids = getNewSingle(df)
    new_ids = new_ids[new_ids['FY'] == (fy - 1)]['ID'].unique()
    return df[df['ID'].isin(new_ids)].reset_index(drop=True)

def getConsecutive(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Example logic: Donors who gave in both (fy-1) and (fy-2).
    """
    new_df = df.copy()
    mask = new_df.groupby('ID')['FY'].transform(
        lambda col: ((col == (fy - 1)).any() and (col == (fy - 2)).any())
    )
    return new_df[mask].reset_index(drop=True)

def getLapsed(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors who gave before (fy-1), but not in (fy-1).
    """
    new_df = df.copy()
    def condition(col):
        return (not (col == fy - 1).any()) and ((col < (fy - 1)).any())
    mask = new_df.groupby('ID')['FY'].transform(condition)
    return new_df[mask].reset_index(drop=True)

def getReinstatedLastYear(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Donors who gave in (fy-1), did NOT give in (fy-2),
    but had given at some point prior to (fy-2).
    """
    new_df = df.copy()
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
    # For Rolling snapshots, limit the population to donors who appear in the
    # trailing‑12‑month window so the cohort sizes reflect the same period.
    if 'IN_WINDOW' in df.columns:
        population = df[df['IN_WINDOW']]['ID'].nunique()
    else:
        population = df['ID'].nunique()
    # If this is a Rolling dataset, use IN_WINDOW to define the current period.
    if 'IN_WINDOW' in df.columns:
        df_period = df[df['IN_WINDOW']]
    else:
        df_period = df[df['FY'] == fy]

    donors_this_period = df_period['ID'].nunique()
    total_revenue_this_period = df_period['AMOUNT'].sum()
    gifts_this_period = len(df_period)

    output['population'] = population
    output['donors'] = donors_this_period
    output['totalRevenue'] = total_revenue_this_period
    output['gifts'] = gifts_this_period

    output['valuePerDonor'] = (
        total_revenue_this_period / donors_this_period if donors_this_period != 0 else 0
    )
    output['activation'] = (
        donors_this_period / population if population != 0 else 0
    )
    output['giftsPerDonor'] = (
        gifts_this_period / donors_this_period if donors_this_period != 0 else 0
    )
    output['avgGift'] = (
        total_revenue_this_period / gifts_this_period if gifts_this_period != 0 else 0
    )

    return output


###############################
# STEP 6: OPTIONAL ROLLING    #
###############################

def toRolling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for a trailing‑12‑month (*Rolling*) snapshot.

    1. Identify the window: `(max_date - 1 year + 1 day)` → `max_date`.
    2. Add a boolean column `IN_WINDOW` to mark gifts **inside** that window.
    3. Return **the full DataFrame** (no donor filtering) so that population
       counts for segments (e.g. Lapsed) can still include donors who did not
       give in the trailing window.
    """
    max_date = data['DATE'].max()
    if pd.isna(max_date):
        return pd.DataFrame()

    start_date = (max_date - pd.DateOffset(years=1)) + pd.Timedelta(days=1)

    data_rolled = data.copy()
    data_rolled['IN_WINDOW'] = (data_rolled['DATE'] >= start_date) & (data_rolled['DATE'] <= max_date)

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
# STEP 8: COMBINE RESULTS   #
##########################

def combine_results(results: dict, rolling_results: dict, last_fy: int) -> pd.DataFrame:
    """
    Takes the per-FY results dict plus the rolling results dict,
    and merges them into a single wide table.
    """
    frames = []
    
    # 1) Melt each FY's DataFrame
    for fy_str, df_out in results.items():
        df_temp = df_out.copy()
        df_temp['segment'] = df_temp.index
        df_melt = df_temp.melt(id_vars='segment', var_name='metric', value_name='value')
        df_melt['FY'] = "FY" + fy_str
        frames.append(df_melt)
    
    # 2) Melt the rolling results
    if str(last_fy) in rolling_results:
        df_roll = rolling_results[str(last_fy)].copy()
        df_roll['segment'] = df_roll.index
        df_melt_roll = df_roll.melt(id_vars='segment', var_name='metric', value_name='value')
        df_melt_roll['FY'] = "Rolling"
        frames.append(df_melt_roll)
    
    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    df_pivot = df_all.pivot_table(
        index=['segment','metric'],
        columns='FY',
        values='value'
    )
    
    def sort_key(col):
        if col.startswith("FY"):
            return int(col[2:])
        return 999999 # Place 'Rolling' at the end
        
    sorted_cols = sorted(df_pivot.columns, key=sort_key)
    df_pivot = df_pivot[sorted_cols]
    
    df_final = df_pivot.reset_index()

    segment_mapping = {
        'new': 'New Donors',
        'newLast': 'New Previous Year',
        'consecutive': 'Consecutive',
        'reLast': 'Reinstated Last Year',
        'lapsed': 'Lapsed',
        'allDon': 'All Donors'
    }
    df_final['segment'] = df_final['segment'].replace(segment_mapping)

    desired_order = [
        'New Donors', 'New Previous Year', 'Consecutive', 
        'Reinstated Last Year', 'Lapsed', 'All Donors'
    ]
    df_final['segment'] = pd.Categorical(df_final['segment'], categories=desired_order, ordered=True)

    # ── Order metrics in a logical, user‑specified sequence ──────────────
    metric_order = [
        'population', 'donors', 'activation',
        'valuePerDonor', 'giftsPerDonor',
        'totalRevenue', 'avgGift', 'gifts'
    ]
    df_final['metric'] = pd.Categorical(
        df_final['metric'],
        categories=metric_order,
        ordered=True
    )
    df_final = df_final.sort_values(by=['segment','metric'])

    return df_final


##########################
# STEP 9: CHARTING FUNCTIONS
##########################

def calculate_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Year-over-Year (YoY) percentage change for each metric.
    """
    df_yoy = df.copy().set_index(['segment', 'metric'])
    fy_cols = sorted([col for col in df.columns if col.startswith('FY')])
    
    if len(fy_cols) < 2:
        return pd.DataFrame()

    for i in range(1, len(fy_cols)):
        prev_fy_col = fy_cols[i-1]
        curr_fy_col = fy_cols[i]
        yoy_col_name = f"{curr_fy_col} vs {prev_fy_col} YoY %"
        
        prev_vals = df_yoy[prev_fy_col]
        curr_vals = df_yoy[curr_fy_col]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            yoy_pct = np.divide(curr_vals - prev_vals, prev_vals) * 100
        
        yoy_pct.replace([np.inf, -np.inf], np.nan, inplace=True)
        yoy_pct.fillna(0, inplace=True)
        
        df_yoy[yoy_col_name] = yoy_pct

    return df_yoy.reset_index()


def create_metric_trend_chart(df: pd.DataFrame, metric_name: str):
    """
    Creates a line chart showing the trend of a single metric across FYs for all segments.
    """
    df_metric = df[df['metric'] == metric_name]
    fy_cols = sorted([col for col in df.columns if col.startswith('FY')])
    
    df_melt = df_metric.melt(id_vars=['segment'], value_vars=fy_cols, var_name='Fiscal Year', value_name=metric_name)
    
    y_axis_format = None
    if 'Revenue' in metric_name or 'value' in metric_name or 'Gift' in metric_name:
        y_axis_format = '$,.0f'
    elif 'activation' in metric_name:
        y_axis_format = '.0%'

    fig = px.line(df_melt, x='Fiscal Year', y=metric_name, color='segment',
                  title=f'Trend for: {metric_name}', markers=True,
                  labels={'segment': 'Donor Segment', 'value': metric_name})
    
    fig.update_layout(
        yaxis_title=metric_name,
        legend_title_text='Donor Segment',
        yaxis_tickformat=y_axis_format
    )
    return fig

def create_yoy_chart(df_yoy: pd.DataFrame, metric_name: str):
    """
    Creates a bar chart showing YoY % change for a single metric across all segments.
    """
    df_metric = df_yoy[df_yoy['metric'] == metric_name]
    yoy_cols = [col for col in df_yoy.columns if 'YoY %' in col]

    if not yoy_cols:
        return None
    
    df_melt = df_metric.melt(id_vars=['segment'], value_vars=yoy_cols, var_name='Year-over-Year Period', value_name='Percentage Change')

    fig = px.bar(df_melt, x='segment', y='Percentage Change', color='Year-over-Year Period',
                 barmode='group',
                 title=f'Year-over-Year % Change for: {metric_name}',
                 labels={'segment': 'Donor Segment', 'Percentage Change': 'YoY % Change'})
    
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(yaxis_ticksuffix='%', legend_title_text='YoY Period')
    return fig


#################################
# STEP 10: LLM/AI COMMENTARY FUNS #
#################################

# MODIFIED FOR OPENAI
def get_llm_commentary_openai(df: pd.DataFrame) -> str:
    """
    Generates natural language commentary on the analysis results using the OpenAI API.
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Convert dataframe to a string format that's easy for the LLM to parse
    data_string = df.to_csv(index=False)
    
    # Get the most recent fiscal year and the rolling period columns for focus
    fy_cols = sorted([col for col in df.columns if col.startswith('FY')])
    most_recent_fy = fy_cols[-1] if fy_cols else "the most recent fiscal year"
    
    # Create the system prompt to set the persona and instructions
    system_prompt = f"""
    You are an expert fundraising analyst for a non-profit organization. Your task is to provide a concise, insightful, and actionable analysis of the provided donor file health data. The data is presented in CSV format.

    The 'segment' column describes donor cohorts:
    - New Donors: Gave for the first time in that fiscal year.
    - Consecutive: Gave in the previous two fiscal years.
    - Lapsed: Skipped the previous fiscal year but gave before that.
    - Reinstated Last Year: Gave last year after previously skipping a year.
    - All Donors: The entire active donor file.
    
    The 'metric' column contains key performance indicators. The columns 'FY...' represent fiscal years, and 'Rolling' represents a projection for the current year-end.

    Please structure your analysis into three sections with clear markdown headings:

    ### Executive Summary
    A brief, high-level overview of the most critical findings. Focus on the overall health of the donor file based on the data in `{most_recent_fy}` and the 'Rolling' period.

    ### Key Observations & Trends
    - Analyze the performance of the key donor segments (New, Consecutive, Lapsed) in `{most_recent_fy}`. Are they growing or shrinking? What is their value?
    - Identify the most significant Year-over-Year (YoY) changes. Mention specific metrics and segments (e.g., "a 20% decline in Consecutive donors" or "a 15% increase in the average gift from New Donors").
    - Discuss the implications of these trends. For example, what does a decline in lapsed donor reinstatement mean for future revenue?

    ### Actionable Recommendations
    - Based on your analysis, provide 2-3 specific, actionable recommendations.
    - For each recommendation, state who it affects (e.g., "Target Lapsed donors...") and what the desired outcome is (e.g., "...to improve retention and long-term value.").
    - Examples: "Launch a targeted win-back campaign for Lapsed donors with a special offer." or "Develop a 'second gift' welcome series for New Donors to improve their retention into the next year."
    """

    # Create the user prompt which contains the actual data
    user_prompt = f"""
    Here is the donor file health data for my organization. Please analyze it according to the instructions.
    ---
    {data_string}
    ---
    """

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",  # Or "gpt-3.5-turbo", "gpt-4-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


##########################
# STREAMLIT FRONT END    #
##########################

st.set_page_config(layout="centered")
col1, col2, col3, col4, col5 = st.columns(5)
col3.image("IS-Logo_RGB_Vertical-onWhite.png", width=112)
st.title("Donor File Health Analysis Portal")

# 1) File Upload
uploaded_file = st.file_uploader("Upload donor transaction file and run analysis.", type=["csv"])

df_uploaded = None
if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.warning("Please upload a file to begin. (CSV Only)")

if df_uploaded is not None:
    # 2) Column Mapping Step
    st.subheader("Map Your Columns")
    columns_in_file = df_uploaded.columns.tolist()
    
    col_map_cols = st.columns(3)
    with col_map_cols[0]:
        id_col = st.selectbox("Select the column for Donor ID", columns_in_file, index=0)
    with col_map_cols[1]:
        amount_col = st.selectbox("Select the column for Amount", columns_in_file, index=1 if len(columns_in_file) > 1 else 0)
    with col_map_cols[2]:
        date_col = st.selectbox("Select the column for Date", columns_in_file, index=2 if len(columns_in_file) > 2 else 0)

    column_map = {'id': id_col, 'amount': amount_col, 'date': date_col}
    if len(set(column_map.values())) < 3:
        st.error("Each mapping (ID, Amount, Date) must be a unique column.")
        st.stop()

    # 3) Parameter Inputs
    st.subheader("Analysis Parameters")
    param_cols = st.columns(4)
    with param_cols[0]:
        fy_format_options = {"January–December": 1, "April–March": 4, "July–June": 7, "October–September": 10}
        fy_choice = st.selectbox("Fiscal Year Format", list(fy_format_options.keys()))
        fy_start_month = fy_format_options[fy_choice]
    with param_cols[1]:
        years_available = list(range(2018, datetime.now().year + 2))
        default_years = [y for y in [current_year - 3, current_year - 2, current_year - 1, current_year] if y in years_available]
        selected_fys = st.multiselect("FYs to Include", years_available, default=default_years)
    with param_cols[2]:
        analysis_start_date = st.date_input("Start Date (optional)", value=None)
        analysis_end_date = st.date_input("End Date (optional)", value=None)
    with param_cols[3]:
        max_donation_cutoff = st.number_input("Max Donation Cutoff", value=10000, step=1000)

    run_analysis_button = st.button("Run Analysis", type="primary")

    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'final_df' not in st.session_state:
        st.session_state.final_df = pd.DataFrame()

    if run_analysis_button:
        with st.spinner("Processing your file... This may take a moment."):
            try:
                trx = clean_data(
                    df_uploaded, column_map,
                    max_date=(pd.to_datetime(analysis_end_date) if analysis_end_date else None),
                    min_date=(pd.to_datetime(analysis_start_date) if analysis_start_date else None),
                    max_amount=max_donation_cutoff, fy_start_month=fy_start_month
                )
                if not selected_fys:
                    st.error("Please select at least one Fiscal Year.")
                    st.stop()

                results = runAll(trx, fy_list=sorted(selected_fys))
                last_fy = sorted(selected_fys)[-1]
                rolled_data = toRolling(trx)
                rolling_results = runAll(rolled_data, [last_fy])
                
                st.session_state.final_df = combine_results(results, rolling_results, last_fy)
                st.session_state.analysis_complete = True
                st.session_state.commentary = "" # Reset commentary on new run
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.session_state.analysis_complete = False

if st.session_state.get('analysis_complete', False):
    st.success("Analysis complete!")
    
    final_df = st.session_state.final_df
    final_csv = final_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Combined Analysis (CSV)",
        data=final_csv,
        file_name="analysis_results.csv",
        mime="text/csv"
    )

    st.subheader("Combined Table Preview")
    st.dataframe(final_df)
    
    st.subheader("Analysis Charts")
    yoy_df = calculate_yoy(final_df)
    metrics_to_chart = final_df['metric'].unique()
    tab1, tab2 = st.tabs(["Metric Trends", "Year-over-Year % Change"])

    with tab1:
        st.header("Metric Trends Over Time")
        for metric in metrics_to_chart:
            st.markdown(f"#### {metric}")
            fig = create_metric_trend_chart(final_df, metric)
            st.plotly_chart(fig, use_container_width=True)
            img_bytes = fig.to_image(format="png", scale=2)
            st.download_button(f"Download '{metric}' Trend Chart", img_bytes, f"trend_{metric}.png", "image/png")
            st.markdown("---")

    with tab2:
        st.header("Year-over-Year Percentage Change")
        if yoy_df.empty:
            st.warning("Not enough fiscal years selected for YoY changes.")
        else:
            for metric in metrics_to_chart:
                st.markdown(f"#### {metric}")
                fig = create_yoy_chart(yoy_df, metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    img_bytes = fig.to_image(format="png", scale=2)
                    st.download_button(f"Download '{metric}' YoY Chart", img_bytes, f"yoy_{metric}.png", "image/png")
                    st.markdown("---")
    
    # MODIFIED FOR OPENAI
    with st.expander("Generate AI Commentary", expanded=True):
        st.write("""
        **Setup required:** This feature uses the OpenAI API. To use it, you must have an OpenAI API key stored in Streamlit secrets.
        1. Create a file named `secrets.toml` in a `.streamlit` directory in your app's root folder.
        2. Add your API key: `OPENAI_API_KEY = "sk-..."`
        """)
        
        if st.button("Generate Commentary", type="secondary"):
            if 'OPENAI_API_KEY' not in st.secrets or not st.secrets.get('OPENAI_API_KEY'):
                st.error("OpenAI API key not found. Please follow the setup instructions.")
            else:
                try:
                    with st.spinner("Generating AI analysis with OpenAI..."):
                        commentary = get_llm_commentary_openai(final_df)
                        st.session_state.commentary = commentary
                except Exception as e:
                    st.error(f"An error occurred while generating commentary: {e}")

        if st.session_state.get('commentary'):
            st.subheader("AI-Generated Analysis")
            st.markdown(st.session_state.commentary)