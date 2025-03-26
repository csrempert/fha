import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add the directory containing this file (app.py) to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

######################
# CONFIG & CONSTANTS #
######################

# A simple list of labels used for final CSV rows
ROW_LABELS = ['All Donors', 'New Donors', 'New Last Year', 'Consecutive', 'Lapsed', 'Reinstated Last Year']


##########################################
# STEP 1: READ & CLEAN THE TRANSACTIONS  #
##########################################

def assign_fiscal_year(row, start_month: int) -> int:
    """
    Assigns the fiscal year based on the user’s 'start_month'.
    This version uses the START of the fiscal year for labeling.
    
    For example, if the user selects July–June (start_month=7),
    then a transaction on 2022-07-01 => FY=2022 (the year it starts).
    That means all transactions from 2022-07-01 through 2023-06-30
    will be labeled '2022'.
    """
    y = row['YEAR']
    m = row['MONTH']

    # If the user selected January–December, just use the calendar year:
    if start_month == 1:
        return y
    
    # Otherwise, for Apr–Mar, Jul–Jun, or Oct–Sep, if the transaction date
    # is on or after 'start_month', assign it the current year;
    # otherwise assign it to the previous year.
    if m >= start_month:
        return y
    else:
        return y - 1


def clean_data(
    df: pd.DataFrame,
    column_map: dict,
    max_date: pd.Timestamp = None,
    min_date: pd.Timestamp = None,
    max_amount: float = None,
    fy_start_month: int = 1
) -> pd.DataFrame:
    """
    Reads transaction data from an already-loaded DataFrame,
    uses 'column_map' to rename columns to a standard set:
       - 'ID', 'AMOUNT', 'DATE'
    Then applies:
      - date parsing
      - optional filtering by date range
      - optional filtering out large gifts
      - assignment of YEAR, MONTH, DAY
      - assignment of a dynamic FY based on the user-selected start month
      - dropping rows missing critical fields
    """

    # 1) Rename user-selected columns to a standard set
    rename_dict = {
        column_map['id']: 'ID',
        column_map['amount']: 'AMOUNT',
        column_map['date']: 'DATE'
    }
    df = df.rename(columns=rename_dict)

    # 2) Ensure required columns exist
    required_cols = ['ID', 'AMOUNT', 'DATE']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing: {missing}.\n"
            f"Please re-check your column_map: {column_map}\n"
            f"DataFrame columns are: {list(df.columns)}"
        )

    # 3) Parse date and amount
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['AMOUNT'] = (
        df['AMOUNT']
        .astype(str)
        .str.replace('[$,]', '', regex=True)
        .astype(float)
    )

    # 4) Drop rows missing ID, DATE, or AMOUNT
    df = df.dropna(subset=['ID', 'DATE', 'AMOUNT']).reset_index(drop=True)

    # 5) Filter out by optional date range
    if min_date is not None:
        df = df[df['DATE'] >= min_date]
    if max_date is not None:
        df = df[df['DATE'] <= max_date]

    # 6) Filter out donations above the user-defined max_amount
    if max_amount is not None:
        df = df[df['AMOUNT'] <= max_amount]

    # 7) Create YEAR, MONTH, DAY columns
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day

    # 8) Compute FY based on user-selected fiscal year start
    df['FY'] = df.apply(lambda r: assign_fiscal_year(r, fy_start_month), axis=1)

    return df


##########################
# STEP 2: HELPER METHODS #
##########################

def create_output(name: str, data: pd.DataFrame) -> dict:
    """
    Example output function (group by YEAR/MONTH, etc.).
    In your final use-case, tailor as needed.
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
    Used later to identify donors who might have lapsed, etc.
    """
    data = data.sort_values(['ID','DATE'])
    data['Diff'] = data.groupby('ID')['DATE'].diff().dt.days / 365.0
    return data


#########################
# STEP 3: CATEGORY FUNS #
#########################

def getNewSingle(df: pd.DataFrame, data_only: bool=False) -> pd.DataFrame or dict:
    """
    Return each donor's very first gift record (by date).
    """
    name = 'new'
    new_data = df.sort_values(by=['ID','DATE']).groupby('ID', group_keys=False).head(1)

    if data_only:
        return new_data
    else:
        return create_output(name, new_data)

def getLapsedSingle(df: pd.DataFrame, data_only: bool=False) -> pd.DataFrame or dict:
    """
    Example 'lapsed' logic: if time between consecutive gifts is > 1 year,
    consider that donor record lapsed at that point.
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
    Returns data for donors whose first-ever gift is in the specified FY.
    """
    new_ids = getNewSingle(df, data_only=True)
    new_ids = new_ids[new_ids['FY'] == fy]['ID'].unique()
    return df[df['ID'].isin(new_ids)].reset_index(drop=True)

def getNewDonorsYearPrevious(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Returns data for donors whose first-ever gift was in the prior FY.
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
    Example logic: Donors who gave before (fy-1), but not in (fy-1).
    """
    new_df = df.copy().sort_values(by=['ID','DATE'])
 
    def condition(col):
        return (not (col == fy - 1).any()) and ((col < (fy - 1)).any())
 
    mask = new_df.groupby('ID')['FY'].transform(condition)
    return new_df[mask].reset_index(drop=True)

def getReinstatedLastYear(df: pd.DataFrame, fy: int) -> pd.DataFrame:
    """
    Example logic: Donors who gave in (fy-1), did NOT give in (fy-2),
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
    # Move everything so it aligns at a year earlier
    new_end_date = pd.to_datetime(f"{max_date.year - 1}-12-31")
    diff = max_date - new_end_date
 
    data_rolled = data.copy()
    data_rolled['DATE'] = data_rolled['DATE'] - diff
 
    # Recompute YEAR, MONTH, DAY, FY
    data_rolled['YEAR'] = data_rolled['DATE'].dt.year
    data_rolled['MONTH'] = data_rolled['DATE'].dt.month
    data_rolled['DAY'] = data_rolled['DATE'].dt.day
 
    # For demonstration, label the rolling window as a calendar FY
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


####################
# EXECUTION EXAMPLE
####################

if __name__ == "__main__":
    print("fha.py: Corrected version with revised fiscal year logic.")

##########################
# STREAMLIT FRONT END
##########################

 # Display logo
with open("is_logo_dark.svg", "r") as logo_file:
     logo_svg = logo_file.read()
st.markdown(logo_svg, unsafe_allow_html=True)
st.title("Donor File Health Analysis Portal")
st.write("Upload donor transaction file and run analysis.")

# 1) File Upload
uploaded_file = st.file_uploader("Upload Transaction File", type=["csv"])

# We'll store a placeholder for the DataFrame once uploaded
df_uploaded = None
column_map = {}

if uploaded_file is not None:
    # Read file into a DataFrame (detect if CSV vs Excel)
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        df_uploaded = pd.read_csv(uploaded_file)
    else:
        df_uploaded = pd.read_excel(uploaded_file)

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # 2) Column Mapping Step
    st.subheader("Map Your Columns")

    columns_in_file = df_uploaded.columns.tolist()

    # Required columns
    id_col = st.selectbox("Select the column for Donor ID", columns_in_file)
    amount_col = st.selectbox("Select the column for Amount", columns_in_file)
    date_col = st.selectbox("Select the column for Date", columns_in_file)

    # Build the column_map based on user selections
    column_map = {
        'id': id_col,
        'amount': amount_col,
        'date': date_col
    }

    # Prevent user from selecting the same column for multiple required fields
    if (id_col == amount_col) or (id_col == date_col) or (amount_col == date_col):
        st.error("You have selected the same column for multiple required fields (ID, Amount, Date).")
        st.stop()

    st.write("Column mapping complete. Proceed with analysis parameters below.")

else:
    st.warning("Please upload a file (CSV only) to begin.")


# 3) Parameter Inputs
st.subheader("Analysis Parameters")

# Fiscal Year Format (dropdown)
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

# Let users pick which FYs they want to include
years_available = list(range(2019, 2031))  # Example range
selected_fys = st.multiselect(
    "Select which Fiscal Years to include in the output",
    years_available,
    default=[2021, 2022, 2023]  # or any sensible default
)

# Date range
analysis_start_date = st.date_input("Filter Start Date (optional)", value=None)
analysis_end_date = st.date_input("Filter End Date (optional)", value=None)

# Max donation cutoff
max_donation_cutoff = st.number_input(
    "Max Donation Amount (exclude donations above this amount)",
    value=25000,
    step=1000
)

# Button to run
run_analysis_button = st.button("Run Analysis")


##########################
# RUN ANALYSIS LOGIC
##########################

if run_analysis_button:
    if df_uploaded is None:
        st.warning("No file uploaded. Please upload a file first.")
    else:
        # Convert from Streamlit date to pd.Timestamp if user selected something
        start_date = pd.to_datetime(analysis_start_date) if analysis_start_date else None
        end_date = pd.to_datetime(analysis_end_date) if analysis_end_date else None

        st.write("Processing your file. Sit tight! This may take a while.")

        try:
            # 1) Clean & standardize the DataFrame
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

        # 2) Run analysis
        results = runAll(trx, fy_list=sorted(selected_fys))

        # Generate CSVs in-memory for download
        csv_files = {}
        for fy, df_out in results.items():
            csv_files[f"analysis_{fy}.csv"] = df_out.to_csv(index=False)

        # Rolling analysis
        rolled_data = toRolling(trx)
        last_fy = sorted(selected_fys)[-1]
        rolling_results = runAll(rolled_data, [last_fy])
        for fy, df_out in rolling_results.items():
            csv_files[f"analysis_{fy}_rolling.csv"] = df_out.to_csv(index=False)

        import io, zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_name, csv_content in csv_files.items():
                zip_file.writestr(file_name, csv_content)
        zip_buffer.seek(0)

        st.download_button(
            label="Download Analysis Results (Zip)",
            data=zip_buffer,
            file_name="analysis_results.zip",
            mime="application/zip"
        )
        st.success("Analysis complete! Download the results using the button above.")

        # 5) Display a summary in-app
        st.subheader("Analysis Summary")
        st.write("Here are the breakdowns for each FY you selected:")
        for fy_str, df_out in results.items():
            st.markdown(f"**Fiscal Year: {fy_str}**")
            st.dataframe(df_out)

        st.info("Done! Download your analysis results using the button above.")