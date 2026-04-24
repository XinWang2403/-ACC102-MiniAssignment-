#!/usr/bin/env python
# coding: utf-8

# # Interactive Sales and Profit Analysis Tool
# 
# **Course:** ACC102  
# **Project type:** Business data analysis notebook  
# **Objective:** To analyse sales and profit performance in a retail dataset and present an interactive, academically structured decision-support notebook.
# 
# This notebook develops a complete workflow that moves from data acquisition and cleaning to feature engineering, visual interpretation, and managerial insights. The emphasis is on clarity, correctness, and business interpretation rather than advanced modelling.

# ## 1. Problem Definition
# 
# Retail organisations must understand where revenue is generated, which segments create profit, and whether discounting practices support or damage financial performance. In this assignment, the business problem is framed as follows:
# 
# **How can sales transactions be analysed to identify trends, regional differences, product performance, and the effect of discounting on profitability?**
# 
# The analysis is designed to support management decisions in four areas:
# 
# - monitoring sales and profit performance over time;
# - comparing regional business outcomes;
# - identifying strong and weak product categories;
# - evaluating whether discounting is associated with lower profit outcomes.

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
pio.renderers.default = "iframe"
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
px.defaults.template = 'plotly_white'

OUTPUT_DIR = Path('outputs') / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_plotly_image(fig, file_name, width=1400, height=800, scale=2):
    # Plotly image export requires the kaleido package in the notebook environment.
    output_path = OUTPUT_DIR / file_name
    pio.write_image(fig, output_path, format='png', width=width, height=height, scale=scale)
    print(f'Saved image to: {output_path.resolve()}')


# ## 2. Data Source Description
# 
# The dataset used in this project is the Superstore retail transactions dataset provided in the assignment brief.
# 
# - **Primary source URL:** `https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv`
# - **Fallback mirror used for reproducibility if the branch path changes:** `https://raw.githubusercontent.com/leonism/sample-superstore/refs/heads/master/data/superstore.csv`
# - **Source type:** Online CSV file
# - **Access date for this assignment:** **April 2026**
# - **Unit of analysis:** Individual sales transactions
# 
# The dataset contains transaction-level information such as order dates, customer segments, regions, product categories, sales values, discount rates, and profit outcomes. It is therefore appropriate for descriptive business analytics and for building a simple interactive decision-support notebook.

# In[2]:


DATA_URLS = [
    'https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv',
    'https://raw.githubusercontent.com/plotly/datasets/main/superstore.csv',
    'https://raw.githubusercontent.com/leonism/sample-superstore/refs/heads/master/data/superstore.csv',
]

# The notebook tries the assignment URL first and then a same-repository fallback
# in case the branch path changes over time.
df = None
selected_url = None
last_error = None

for url in DATA_URLS:
    try:
        df = pd.read_csv(url, encoding='latin1')
        selected_url = url
        break
    except Exception as exc:
        last_error = exc

if df is None:
    raise RuntimeError(
        'The dataset could not be loaded from the configured online source URLs. '
        f'Last error: {last_error}'
    )

print(f'Dataset loaded from: {selected_url}')
print(f'Shape: {df.shape[0]:,} rows x {df.shape[1]} columns')

df.head()


# The dataset is loaded directly from the online source, which supports reproducibility and demonstrates a realistic analytics workflow. The initial preview allows the analyst to confirm that the table contains retail transaction variables suitable for trend, profitability, and discount analysis.

# In[3]:


# Create a working copy before cleaning.
df = df.copy()

# Standardise column names and remove surrounding white space in text fields.
df.columns = df.columns.str.strip()
text_columns = df.select_dtypes(include='object').columns
for column in text_columns:
    df[column] = df[column].str.strip()

# Convert date fields to datetime.
for column in ['Order Date', 'Ship Date']:
    df[column] = pd.to_datetime(df[column], format='%m/%d/%Y', errors='coerce')

# Convert key numeric fields to numeric dtype.
for column in ['Sales', 'Quantity', 'Discount', 'Profit', 'Postal Code']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

missing_before = df.isna().sum().rename('Missing Before')
duplicate_rows = int(df.duplicated().sum())

# Remove exact duplicate rows and drop records with missing critical business fields.
df = df.drop_duplicates().reset_index(drop=True)
critical_columns = ['Order Date', 'Sales', 'Profit', 'Discount', 'Region', 'Category', 'Sub-Category']
df = df.dropna(subset=critical_columns).reset_index(drop=True)

missing_after = df.isna().sum().rename('Missing After')
cleaning_report = pd.concat([missing_before, missing_after], axis=1).fillna(0).astype(int)

print(f'Duplicate rows removed: {duplicate_rows}')
print(f'Rows remaining after cleaning: {len(df):,}')

cleaning_report.sort_values('Missing Before', ascending=False)


# ## 3. Data Cleaning Discussion
# 
# The cleaning stage converts date variables into a usable datetime format and ensures that financial variables are stored numerically. Duplicate rows are removed to avoid overstating business activity, while records missing essential analytical fields are excluded because they would distort trend and profitability calculations.
# 
# For this dataset, the cleaning process is intentionally simple and transparent. 

# In[4]:


# Derive time and profitability features for later analysis.
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Month Name'] = df['Order Date'].dt.strftime('%b')
df['Year-Month'] = df['Order Date'].dt.to_period('M').astype(str)

# Profit margin is defined as profit divided by sales.
df['Profit Margin'] = np.where(df['Sales'] != 0, df['Profit'] / df['Sales'], np.nan)
df['Profit Margin %'] = df['Profit Margin'] * 100

# Create discount bands to make interpretation easier.
discount_labels = ['No Discount', 'Low', 'Medium', 'High', 'Very High']
df['Discount Band'] = pd.cut(
    df['Discount'],
    bins=[-0.001, 0.00, 0.10, 0.20, 0.30, 1.00],
    labels=discount_labels,
    include_lowest=True,
)
df['Discount Band'] = df['Discount Band'].cat.add_categories(['Unknown']).fillna('Unknown')

feature_preview = df[
    ['Order Date', 'Sales', 'Profit', 'Discount', 'Year', 'Month', 'Profit Margin %', 'Discount Band']
].head()

feature_preview


# ## 4. Feature Engineering Discussion
# 
# Three groups of derived variables are created for the analysis.
# 
# - **Time features (`Year`, `Month`, `Year-Month`)** support trend analysis.
# - **Profit Margin** standardises profit relative to sales and therefore improves comparability across transactions.
# - **Discount Band** converts a continuous discount rate into interpretable business categories.
# 
# These features make the notebook more useful as a management tool because they translate raw transaction data into business-oriented analytical dimensions.

# In[5]:


summary_metrics = pd.DataFrame({
    'Metric': [
        'Number of transactions',
        'Total sales',
        'Total profit',
        'Average discount',
        'Average profit margin',
    ],
    'Value': [
        len(df),
        df['Sales'].sum(),
        df['Profit'].sum(),
        df['Discount'].mean(),
        df['Profit Margin'].mean(),
    ],
})

summary_metrics


# The summary table provides a concise business snapshot before deeper investigation. It gives an initial sense of scale and helps frame the later sections, which explain how total performance is distributed across time, regions, product lines, and discount levels.

# In[6]:


monthly_trends = (
    df.groupby(df['Order Date'].dt.to_period('M'))
      .agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum'))
      .reset_index()
)
monthly_trends['Order Month'] = monthly_trends['Order Date'].dt.to_timestamp()

fig_trend = make_subplots(specs=[[{'secondary_y': True}]])
fig_trend.add_trace(
    go.Scatter(
        x=monthly_trends['Order Month'],
        y=monthly_trends['Sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#1f77b4', width=3),
    ),
    secondary_y=False,
)
fig_trend.add_trace(
    go.Scatter(
        x=monthly_trends['Order Month'],
        y=monthly_trends['Profit'],
        mode='lines+markers',
        name='Profit',
        line=dict(color='#d62728', width=3),
    ),
    secondary_y=True,
)

fig_trend.update_layout(
    title='Sales and Profit Trends Over Time',
    xaxis_title='Order Month',
    hovermode='x unified',
    legend_title='Metric',
    height=500,
)
fig_trend.update_yaxes(title_text='Sales', secondary_y=False)
fig_trend.update_yaxes(title_text='Profit', secondary_y=True)
save_plotly_image(fig_trend, '01_sales_profit_trends.png', width=1400, height=700)
fig_trend.show()


# ## 5. Interpretation of Time Trends
# 
# The time-series visualisation compares sales and profit movements across months. This comparison is important because revenue growth does not necessarily imply profit growth. If sales rise while profit remains flat or declines, the business may be relying on low-margin transactions or excessive discounting.
# 
# In academic terms, this section evaluates both **scale** and **quality** of performance over time.

# In[7]:


regional_summary = (
    df.groupby('Region')
      .agg(
          Sales=('Sales', 'sum'),
          Profit=('Profit', 'sum'),
          Orders=('Order ID', 'nunique'),
          Avg_Profit_Margin=('Profit Margin', 'mean'),
      )
      .reset_index()
)
regional_summary['Avg Profit Margin %'] = regional_summary['Avg_Profit_Margin'] * 100
regional_summary = regional_summary.sort_values('Sales', ascending=False)

fig_region = px.bar(
    regional_summary,
    x='Region',
    y=['Sales', 'Profit'],
    barmode='group',
    title='Regional Comparison of Sales and Profit',
    color_discrete_sequence=['#4c78a8', '#f58518'],
)
fig_region.update_layout(yaxis_title='US Dollars', height=500)
save_plotly_image(fig_region, '02_regional_comparison.png', width=1200, height=700)
fig_region.show()

regional_summary


# ## 6. Interpretation of Regional Comparison
# 
# Regional comparison highlights whether strong sales performance is evenly translated into profit. A region with high sales but relatively weak profit may indicate pricing pressure, poor product mix, or discount overuse. Conversely, a region with moderate sales and strong profit suggests more efficient commercial performance.
# 
# This section is valuable for managerial benchmarking because it connects volume-based performance with financial outcomes.

# In[8]:


category_summary = (
    df.groupby('Category')
      .agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum'), Avg_Profit_Margin=('Profit Margin', 'mean'))
      .reset_index()
)
category_summary['Avg Profit Margin %'] = category_summary['Avg_Profit_Margin'] * 100

fig_category = px.bar(
    category_summary.sort_values('Sales', ascending=False),
    x='Category',
    y=['Sales', 'Profit'],
    barmode='group',
    title='Category-Level Performance',
    color_discrete_sequence=['#2a9d8f', '#e76f51'],
)
fig_category.update_layout(yaxis_title='US Dollars', height=500)
save_plotly_image(fig_category, '03_category_performance.png', width=1200, height=700)
fig_category.show()

sub_category_summary = (
    df.groupby(['Category', 'Sub-Category'])
      .agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum'), Avg_Profit_Margin=('Profit Margin', 'mean'))
      .reset_index()
)

best_subcategories = sub_category_summary.sort_values('Profit', ascending=False).head(10).assign(Group='Top 10 by Profit')
worst_subcategories = sub_category_summary.sort_values('Profit', ascending=True).head(10).assign(Group='Bottom 10 by Profit')
sub_category_view = pd.concat([best_subcategories, worst_subcategories], ignore_index=True)

fig_subcategory = px.bar(
    sub_category_view,
    x='Profit',
    y='Sub-Category',
    color='Group',
    facet_col='Group',
    orientation='h',
    title='Most and Least Profitable Sub-Categories',
    color_discrete_map={'Top 10 by Profit': '#3a86ff', 'Bottom 10 by Profit': '#ef476f'},
)
fig_subcategory.update_layout(height=550)
fig_subcategory.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
save_plotly_image(fig_subcategory, '04_subcategory_profitability.png', width=1400, height=800)
fig_subcategory.show()

sub_category_summary.sort_values('Profit', ascending=False).head(10)


# ## 7. Interpretation of Category and Sub-Category Performance
# 
# Category analysis shows how overall product families contribute to business results, while sub-category analysis identifies the specific lines that create or destroy value. This distinction is important because broad categories may appear successful even when some individual sub-categories perform poorly.
# 
# The sub-category view is especially useful for inventory, pricing, and promotion decisions because it reveals where profit concentration is strongest and where commercial risk may exist.

# In[9]:


discount_profit_summary = (
    df.groupby('Discount Band')
      .agg(
          Records=('Discount', 'size'),
          Avg_Sales=('Sales', 'mean'),
          Avg_Profit=('Profit', 'mean'),
          Avg_Profit_Margin=('Profit Margin', 'mean'),
      )
      .reset_index()
)

discount_order = ['No Discount', 'Low', 'Medium', 'High', 'Very High', 'Unknown']
discount_profit_summary['Discount Band'] = pd.Categorical(
    discount_profit_summary['Discount Band'],
    categories=discount_order,
    ordered=True,
)
discount_profit_summary = discount_profit_summary.sort_values('Discount Band')
discount_profit_summary['Avg Profit Margin %'] = discount_profit_summary['Avg_Profit_Margin'] * 100

fig_discount_scatter = px.scatter(
    df,
    x='Discount',
    y='Profit',
    color='Category',
    hover_data=['Sub-Category', 'Region', 'Sales'],
    opacity=0.55,
    title='Discount Versus Profit at Transaction Level',
)
fig_discount_scatter.add_hline(y=0, line_dash='dash', line_color='black')
fig_discount_scatter.update_layout(height=550)
save_plotly_image(fig_discount_scatter, '05_discount_vs_profit_scatter.png', width=1400, height=800)
fig_discount_scatter.show()

fig_discount_band = px.bar(
    discount_profit_summary,
    x='Discount Band',
    y='Avg Profit Margin %',
    color='Discount Band',
    title='Average Profit Margin by Discount Band',
    category_orders={'Discount Band': discount_order},
)
fig_discount_band.update_layout(showlegend=False, height=500)
save_plotly_image(fig_discount_band, '06_discount_band_profit_margin.png', width=1200, height=700)
fig_discount_band.show()

discount_profit_summary


# ## 8. Interpretation of the Discount-Profit Relationship
# 
# The discount analysis examines whether heavier price reductions are associated with weaker profitability. The scatter plot shows transaction-level variation, while the discount-band summary simplifies the pattern for managerial interpretation.
# 
# From a business perspective, the key question is whether discounting stimulates profitable sales or merely increases volume at the expense of margin. This is an important issue in retail performance management.

# In[10]:


interactive_view = (
    df.groupby(['Region', 'Category', 'Sub-Category'])
      .agg(Sales=('Sales', 'sum'), Profit=('Profit', 'sum'))
      .reset_index()
)

fig_treemap = px.treemap(
    interactive_view,
    path=[px.Constant('Superstore'), 'Region', 'Category', 'Sub-Category'],
    values='Sales',
    color='Profit',
    color_continuous_scale='RdYlGn',
    title='Interactive Sales and Profit Exploration Tree Map',
)
fig_treemap.update_layout(height=650)
save_plotly_image(fig_treemap, '07_interactive_treemap.png', width=1400, height=900)
fig_treemap.show()


# ## 9. Interactive Product View
# 
# The tree map acts as a simple interactive data product. Users can hover to inspect detailed values, click into a region or category, and visually compare how sales volume and profit are distributed across the portfolio. This supports exploratory analysis without introducing unnecessary technical complexity.

# ## 10. Final Summary of Key Insights
# 
# The notebook provides a complete descriptive analysis of Superstore sales and profitability. The main insights to emphasise in a managerial conclusion are the following:
# 
# - sales and profit should be interpreted together because high revenue does not always imply strong profitability;
# - regional analysis can reveal whether some markets convert sales into profit more efficiently than others;
# - category totals are informative, but sub-category analysis is necessary to identify precise sources of strength and weakness;
# - discounting requires careful control because larger discounts may reduce profit margins and increase loss-making transactions;
# - interactive visualisations improve managerial understanding by allowing faster exploration of performance drivers.
# 
# Overall, the project demonstrates how a structured Python notebook can support evidence-based retail decision-making through transparent data preparation, focused visual analysis, and academically written interpretation.
