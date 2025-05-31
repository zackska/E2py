import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import requests
from dateutil.relativedelta import relativedelta
import aiohttp
import asyncio
from anytree import Node, RenderTree
import csv

## Output functions

def remove_all_files_in_directory(directory):
    """
    Remove all files in the specified directory except for .csv and .py files.

    Args:
        directory (str): The path to the directory to clean.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.csv') and not file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

def get_top_terms_per_cluster(X, labels, vectorizer, top_n=3):
    """
    Get the top N terms for each cluster from a TF-IDF matrix and KMeans labels.

    Args:
        X (scipy.sparse matrix): TF-IDF feature matrix.
        labels (array-like): Cluster labels for each row in X.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        top_n (int): Number of top terms to return per cluster.

    Returns:
        dict: Mapping from cluster id to a string of top terms.
    """
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for cluster_id in set(labels):
        cluster_docs = X[labels == cluster_id]
        mean_tfidf = cluster_docs.mean(axis=0).A1
        top_terms = [terms[i] for i in mean_tfidf.argsort()[-top_n:][::-1]]
        cluster_terms[cluster_id] = " ".join(top_terms)
    return cluster_terms    

def pivot_heatmap(df, sumrows, numberfmt, save_path, name, title, ylabel):
    """
    Plot a heatmap from a DataFrame, optionally adding sum rows/columns, and save as an image.

    Args:
        df (pd.DataFrame): Data to plot.
        sumrows (bool): Whether to add sum rows/columns.
        numberfmt (str): Format string for numbers (e.g., '.0f').
        save_path (str): Directory to save the image.
        name (str): Filename (without extension) for the image.
        title (str): Title for the heatmap.
        ylabel (str): Label for the y-axis.
    """
    # Replace sum rows and columns with neutral values for heatmap
    if df.values.any():
        if sumrows:
            # Add a new row for the sum of each column
            df.loc['Category Sum'] = df.sum(axis=0)

            # Add a new column for the sum of each row
            df['Size Sum'] = df.sum(axis=1)

            # Move the 'Row Sum' column to the first position
            df.insert(0, 'Size Sum', df.pop('Size Sum'))

        heatmap_data = df.copy()

        if sumrows:
            heatmap_data.iloc[-1, :] = 0  # Replace column sums with 0 for no color
            heatmap_data.iloc[:, 0] = 0  # Replace row sums with 0 for no color

        # Create the custom colormap
        custom_cmap = plt.get_cmap('plasma')

        # Plot the updated heatmap after ensuring mutual exclusivity
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(heatmap_data, annot=df, cmap=custom_cmap, center=0, fmt=numberfmt, cbar=True, cbar_kws={'shrink': 0.5})
        plt.title(f"{title}")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(save_path + name + '.jpg')
        plt.close()

def plot_top_parts(data, sales_col, save_path):
    """
    Plot a bar chart of the top parts by a sales column and save as an image.

    Args:
        data (pd.DataFrame): Data containing part information.
        sales_col (str): Column name for sales values.
        save_path (str): Directory to save the image.
    """
    # Aggregate total sales by part
    # aggregated = data.groupby(part_col)[sales_col].sum().reset_index(drop=True)
    # Sort by sales and take the top N
     # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(data.index, data[sales_col])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {len(data)} Parts by {sales_col}')
    plt.xlabel('Part Name')
    plt.ylabel(sales_col)
    plt.tight_layout()
    plt.grid(True)
    plt.yticks(np.arange(round((min(data[sales_col])-50)/ 500)  * 500, round((max(data[sales_col])+50)/ 500)* 500, step=250))
    plt.savefig(save_path + f'top{len(data)}{sales_col}.jpg')
    plt.close()

def breakdownPlots(df, statistic, save_path):
    """
    Generate and save various heatmaps and distribution plots for part categories, motor sizes, and customer ratios.

    Args:
        df (pd.DataFrame): Data containing part, sales, and customer info.
        statistic (str): Column name for the statistic to analyze (e.g., 'quantityShipped').
        save_path (str): Directory to save plots.
    """

    numberfmt = '.0f'

    part_categories = df['Category'].unique()

    # Pivot Table on motor size and category
    df=df[~df['Category'].str.contains('none', case = False)]
  
    # Pivot Table over ratio to mandrels
    # first, filter out the customers that never buy mandrels
    mandrel_customers = df[df['Category'].str.contains('Mandrels')]['customerCode'].unique()
    mandrel_df = df[df['customerCode'].isin(mandrel_customers)]
    
    # Replacement parts to mandrels for all orders
    sum_df = mandrel_df.pivot_table(index='MotorSize', columns='Category', values=statistic, aggfunc='sum', fill_value=0)
    
    # Normalize the pivot table by the mandrel quantity
    ratiotable = sum_df.div(sum_df['Mandrels'], axis=0).replace([np.inf, -np.inf], 0).fillna(0)
    ratiotable.pop('Mandrels')
    pivot_heatmap(ratiotable, False, numberfmt, save_path, 'MandrelRatio', 'MandrelRatio', 'MotorSize')

    # Overall Quantity Sums heatmap
    sum_df = df.pivot_table(index='MotorSize', columns='Category', values=statistic, aggfunc='sum', fill_value=0)

    # Convert to percentages of total sales
    if 'Total' in statistic:
        # Convert to percentage of total sales
        total_sum = df[statistic].sum()
        sum_df = sum_df*100/total_sum

        # performance stats
        print(sum_df.values.sum())
    
    #Plot heatmaps by customer
    #simultaneously build graph for ratio of mandrels to other parts by customer
    df = df[~df['customerCode'].isna()]
    custratio_df = pd.DataFrame(columns=df['Category'].unique())
        
    for i in df['customerCode'].unique():
        customer_df = df[df['customerCode'].str.contains(i)]
        #ratio rows for later
        
        sum_df = customer_df.pivot_table(index='MotorSize', columns='Category', values=statistic, aggfunc='sum', fill_value=0)
        ratiosum_df = customer_df.pivot_table(index='customerCode', columns='Category', values=statistic, aggfunc='sum', fill_value=0)
        
        if 'Mandrels' in ratiosum_df.columns: 
            if ratiosum_df.loc[i]['Mandrels']>0:
                custratio_df = pd.concat([custratio_df, ratiosum_df/ratiosum_df.loc[i]['Mandrels']])
    
    # plot the ratio map by customer
    custratio_df.pop('Mandrels')
    # Calculate the average of 'Inner Race' for 'ALTITIUDE', 'Discovery', and 'Total'
    avg_altitude = custratio_df[custratio_df.index.str.contains('ALTITUDE', case=False)].mean()
    avg_discovery = custratio_df[custratio_df.index.str.contains('DISCOVERY', case=False)].mean()
    avg_total = custratio_df[custratio_df.index.str.contains('TOTAL', case=False)].mean()
    custratio_df = custratio_df[~custratio_df.index.str.contains('TOTAL|ALTITUDE|DISCOVERY', case=False)]

    custratio_df.loc['ALTITUDE'] = avg_altitude
    custratio_df.loc['DISCOVERY'] = avg_discovery
    custratio_df.loc['TOTAL'] = avg_total


    # Plot the distribution of 'Inner Race'
    inner_race_data = custratio_df['Inner Race']
    plt.figure(figsize=(10, 6))
    sns.histplot(inner_race_data, bins=20, kde=True)
    plt.title('Distribution of Inner Race')
    plt.xlabel('Inner Race Ratio')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path + 'inner_race_distribution.jpg')
    plt.close()

    

    # Update the 'Inner Race' values with the calculated averages

    # Plot the heatmap for the updated customer ratio data
    pivot_heatmap(custratio_df, False, numberfmt, save_path, 'CustomerRatio', 'Ratio of Parts to Mandrels', 'Customer')

    # Divide each row by the row with index 'ALTITUDE'
    custratio_df = custratio_df.div(custratio_df.loc['ALTITUDE'], axis=1)
    custratio_df = custratio_df.drop('ALTITUDE')
    # Remove 'Compression Tools' and 'Jaw Clutch Center' columns
    custratio_df = custratio_df.drop(columns=['Compression Tools'])
    custratio_df = custratio_df.drop(columns=['Jaw Clutch Center'])
    custratio_df = custratio_df.drop(columns=['Adjustable Bent Housings'])
    pivot_heatmap(custratio_df, False, '0.2f', save_path, 'CustomerRatio', 'Ratio of Parts to Mandrels (compared to Altitude)', 'Customer')

    # # Also plot a breakdown of the variants of each part category
    # for i in part_categories.keys():
    #     category_df = df[df['Category'].str.contains(i)]
    #     partNumbers = list(category_df.index.values)
    #     category_df['Variant'] = [re.sub(r'^\D+', '', j)[2:] for j in partNumbers]

    #     variant_df = category_df.pivot_table(index='MotorSize', columns='Variant', values=statistic, aggfunc='sum', fill_value=0)
    #     pivot_heatmap(variant_df, False, numberfmt, save_path + 'Variants/', i, i, 'MotorSize')

def writeExcel(df, quarter_df, materials_df, statistic):
    """
    Write sales, inventory, and materials data to an Excel file with multiple sheets by motor size and a materials sheet.

    Args:
        df (pd.DataFrame): Main sales/inventory data.
        quarter_df (pd.DataFrame): Quarterly sales data.
        materials_df (pd.DataFrame): Materials requirements data.
        statistic (str): Column name for the main statistic (e.g., 'quantityShipped').
    """

    # Merge the sales data with the quarter sales data, change the column names
    excel_df = df.merge(quarter_df[['partNumber', statistic + '_quarter']], on='partNumber', how='left').fillna(0)
    excel_df = excel_df[['partNumber', 'partDescription', statistic, statistic + '_quarter', 'quantityToMake', 'quantityOnHand', 'MotorSize', 'Category', 'openCustomerOrders']]
    excel_df = excel_df.rename(columns={'quantityToMake':'InProcess', 'quantityOnHand':'Inventory', statistic:'Shipped 12Mo', statistic+'_quarter':'Shipped 3Mo'})
    excel_df['Inventory+InProcess'] = excel_df['Inventory']+excel_df['InProcess']

    excel_df = excel_df[[type(i)==str for i in excel_df['Category']]]
    filename = 'sizeForecasts_' + pd.Timestamp.now().strftime('%y-%m-%d') + '.xlsx'
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for i in sorted(excel_df['MotorSize'].unique()):
            size_df = excel_df[excel_df['MotorSize'] == i].drop(['MotorSize', 'Category'], axis=1)
            if not size_df.empty:
                size_df.to_excel(writer, sheet_name=i, index=False)
                worksheet = writer.sheets[i]
                for column in size_df:
                    column_length = max(size_df[column].astype(str).map(len).max(), len(column))
                    col_idx = size_df.columns.get_loc(column)
                    worksheet.set_column(col_idx, col_idx, column_length)
        # Define fill colors
   
        # blanks_df.to_excel(writer, sheet_name='BKKM', index=False)
        # worksheet = writer.sheets['BKKM']
        # for column in blanks_df:
        #     column_length = max(blanks_df[column].astype(str).map(len).max(), len(column))
        #     col_idx = blanks_df.columns.get_loc(column)
        #     worksheet.set_column(col_idx, col_idx, column_length)

        materials_df.to_excel(writer, sheet_name='Materials', index=False)
        worksheet = writer.sheets['Materials']
        for column in materials_df:
            column_length = max(materials_df[column].astype(str).map(len).max(), len(column))
            col_idx = materials_df.columns.get_loc(column)
            worksheet.set_column(col_idx, col_idx, column_length)
        # Send the email with the Excel file as an attachment

## Data processing functions 
def aggregateStatByPart(df, statistic):
    """
    Aggregate a DataFrame by part number, summing the specified statistic and keeping the first value for other columns.

    Args:
        df (pd.DataFrame): Data to aggregate.
        statistic (str): Column name to sum.

    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    # Aggregate the sales data by part number
    agg_rules = {col: 'first' for col in df.columns if col != statistic}
    agg_rules[statistic] = 'sum'
    df = df.groupby('partNumber').agg(agg_rules).reset_index(drop=True)
    return df


## JSON request functions using E2 API    
def longSearchQuery(url, headers, fields, searchField, search):
    """
    Perform a long search query to the E2 API, batching requests if the search list is large.

    Args:
        url (str): Base URL for the API endpoint.
        headers (dict): HTTP headers for authentication.
        fields (str): Comma-separated list of fields to retrieve.
        searchField (str): Field to search on (e.g., 'partNumber').
        search (list): List of values to search for.

    Returns:
        list: List of results from the API.
    """
    start = 0
    take = 300 #url can only be so long
    if isinstance(search[0], (int, float)):
        search = list(map(str, search))
    searchstr = '|'.join(search[start:start+take])
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(get_all_data(url + f'?{searchField}[in]={searchstr}', headers, fields))
    start += take
    while start < len(search):
        searchstr = '|'.join(search[start:start+take])
        output.extend(loop.run_until_complete(get_all_data(url + f'?{searchField}[in]={searchstr}', headers, fields)))
        start+=take

    return output

async def fetch(session, url, headers, fields, skip):
    """
    Asynchronously fetch a batch of data from the E2 API with pagination.

    Args:
        session (aiohttp.ClientSession): Active aiohttp session.
        url (str): API endpoint URL.
        headers (dict): HTTP headers for authentication.
        fields (str): Comma-separated list of fields to retrieve.
        skip (int): Number of records to skip (for pagination).

    Returns:
        dict: JSON response from the API.
    """
    async with session.get(url + f'&take=200&skip={str(skip)}', headers=headers, params={'fields': fields}) as response:
        return await response.json()

async def get_all_data(url, headers, fields):
    """
    Asynchronously retrieve all data from the E2 API, handling pagination.

    Args:
        url (str): API endpoint URL.
        headers (dict): HTTP headers for authentication.
        fields (str): Comma-separated list of fields to retrieve.

    Returns:
        list: List of all results from the API.
    """
    output = []
    skip = 0
    async with aiohttp.ClientSession() as session:
        while True:
            data = await fetch(session, url, headers, fields, skip)
            out = data['Data']
            if not out:
                break
            output.extend(out)
            skip += 200
    return output

def authHeader(url, api_key, password, user):
    """
    Authenticate with the E2 API and return a headers dictionary with a bearer token.

    Args:
        url (str): Base URL for the API.
        api_key (str): API key for authentication.
        password (str): User password.
        user (str): Username.

    Returns:
        dict: HTTP headers with Bearer token for API requests.
    """
    # API endpoint for token generation
    registrationURL = url+'register?apiKey='+ api_key + '&username=' + user +'&password=' + password
    
    response = requests.post(registrationURL)
    bearerpw = response.json()['result']

    data = {
    'apiKey': api_key,
    'userName': user,
    'password': bearerpw,
    }

    response = requests.post(url + 'Login', data=data)
    token = response.json()['result']
    headers = {'Authorization': 'Bearer ' +token}
    return headers

def getdB(url, header, db_name, fields, root='', search={}, filterString = ''):
    """
    Retrieve a database from the E2 API as a DataFrame, optionally saving to CSV and supporting search/filtering.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        db_name (str): Name of the database/table to retrieve.
        fields (str): Comma-separated list of fields to retrieve.
        root (str): Optional directory to save CSV.
        search (dict): Optional search dictionary {field: values}.
        filterString (str): Optional filter string for the query.

    Returns:
        pd.DataFrame: Retrieved data as a DataFrame.
    """
    # This is the wrapper function for retrieving individual databases from the E2 API. fields is a list of strings that are the fields to be retrieved,
    #  db_name is the name of the database to be retrieved, and search is a dictionary with the field to be searched as the key and the search term as the value.
    # If root is specified, the data will be saved to a csv file in the root directory. If filterString is specified, it will be appended to the end of the query string.
    loop = asyncio.get_event_loop()
    
    # In case there are spaces after the commas between fields
    fields = fields.replace(' ', '')
    
    csv_filename = root + f"{db_name}.csv"

    if root and os.path.exists(csv_filename):
        db = pd.read_csv(csv_filename)
    else:
        if search:
            db = longSearchQuery(url + db_name, header, fields, list(search.keys())[0], list(search.values())[0])
        else:
            query = url + db_name + '?'
            if filterString:
                query += filterString
            
            db = loop.run_until_complete(get_all_data(query, header, fields))
        
        db = pd.DataFrame(db)
        if root:
            db.to_csv(csv_filename, index=False)

    return db


def getOpenSales(url, header, df):
    """
    Retrieve and merge open sales order quantities for parts in the DataFrame, splitting in-process and customer orders.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        df (pd.DataFrame): DataFrame with part numbers to match.

    Returns:
        pd.DataFrame: DataFrame with open order quantities merged.
    """

    fields = 'partNumber,quantityToMake,status, orderNumber'
    openOrderQty = getdB(url, header, 'order-line-items', fields, filterString='status=Open')
    openOrderQty = openOrderQty[openOrderQty['partNumber'].isin(df['partNumber'])]
    # Aggregate all open orders
    # Get customerCode for each partNumber by requesting the orders table
    orders = getdB(url, header, 'orders', 'orderNumber,customerCode', search={'orderNumber': openOrderQty['orderNumber'].unique()})
    openOrderQty = openOrderQty.merge(orders[['orderNumber', 'customerCode']], how='left', on='orderNumber')

    # Aggregate open orders to TOMAHAWKDH only
    inProcessQty = openOrderQty[openOrderQty['customerCode'] == 'TOMAHAWKDH']
    inProcessQty = inProcessQty.groupby('partNumber').agg({'quantityToMake': 'sum'}).reset_index()
    df = df.merge(inProcessQty, how='left', on='partNumber')
    
    openOrderQty = openOrderQty[openOrderQty['customerCode'] != 'TOMAHAWKDH']
    openOrderQty = openOrderQty.groupby('partNumber').agg({'quantityToMake': 'sum'}).reset_index()
    openOrderQty.rename(columns={'quantityToMake': 'openCustomerOrders'}, inplace=True)
    df = df.merge(openOrderQty, how='left', on='partNumber')
    return df


def getOpenPurchases(url, header, df):
    """
    Retrieve and merge open purchase order quantities for parts in the DataFrame.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        df (pd.DataFrame): DataFrame with part numbers to match.

    Returns:
        pd.DataFrame: DataFrame with open purchase order quantities merged.
    """
    openOrderQty = getdB(url, header, 'purchase-order-line-items', 'partNumber,quantityOrdered,quantityReceived,partDescription', filterString='status=Open')
    openOrderQty = openOrderQty.fillna(0)
    openOrderQty['quantityOrdered'] = openOrderQty['quantityOrdered']-openOrderQty['quantityReceived']
    if 'partDescription' in df.columns:
        openOrderQty.drop('partDescription', axis = 1)
    openOrderQty = openOrderQty.groupby('partNumber').agg({'quantityOrdered':'sum', 'partDescription':'first'})
    df = df.merge(openOrderQty, how='left', on='partNumber')
    return df

def getLeadTimes(purchases):
    """
    Calculate recent lead times for each part number from purchase order data.

    Args:
        purchases (pd.DataFrame): DataFrame with purchase order data, including 'dateEntered' and 'dateFinished'.

    Returns:
        pd.DataFrame: DataFrame with most recent lead time for each part number.
    """
    dateStats = ['dateEntered', 'dateFinished']
    purchases = purchases[dateStats + ['partNumber', 'vendorCode']]
    purchases = purchases.dropna()
    for i in dateStats:
        purchases[i] = [j[0] for j in purchases[i].str.split('T')]
        purchases[i] = pd.to_datetime(purchases[i])

    purchases['leadTime'] = purchases[dateStats[1]] - purchases[dateStats[0]]
    # the last date will be enetered into the excel, so clean up the format
    purchases[dateStats[1]] = purchases[dateStats[1]].dt.strftime('%y-%m-%d')
    # Get the most recent lead time for each partNumber based on the 'dateFinished'
    # average_lead_times = purchases.groupby('vendorCode')['leadTime'].mean().reset_index()
    # average_lead_times['leadTime'] = average_lead_times['leadTime'].dt.days  # Convert to days

    recent_lead_times = purchases.sort_values(dateStats[1]).groupby('partNumber').tail(1)[['partNumber', 'leadTime', dateStats[1]]]
    recent_lead_times.rename(columns={dateStats[1]:'lastDateReceived'}, inplace=True)
   
    return recent_lead_times

def getEmployees(url, header):
    """
    Retrieve employee data from the E2 API as a DataFrame.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.

    Returns:
        pd.DataFrame: DataFrame of employee data.
    """
    loop = asyncio.get_event_loop()
    fields = 'employeeCode,employeeName,active,departmentNumber'
    employees = pd.DataFrame(loop.run_until_complete(get_all_data(url + f'employees', header, fields)))
    return employees

# functions that acquire multiple databases to form a composite dataframe
def getSales(url, header, startDate, root):
    """
    Retrieve sales and invoice detail data from the E2 API and merge into a single DataFrame.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        startDate (str): Start date for filtering sales (YYYY-MM-DD).
        root (str): Optional directory to save CSVs.

    Returns:
        pd.DataFrame: DataFrame of merged sales and invoice details.
    """

    fields_dict = {
        'ar-invoices': 'invoiceNumber,invoiceDate,customerCode',
        'ar-invoice-details': 'partNumber,partDescription,quantityShipped,unitPrice,invoiceNumber,jobNumber,discountPercent'
    }

    sales = getdB(url, header, 'ar-invoices', fields_dict['ar-invoices'], filterString='invoiceDate[gt]=' + startDate, root=root)
    salesLineItems = getdB(url, header, 'ar-invoice-details', fields_dict['ar-invoice-details'], search={'invoiceNumber': sales['invoiceNumber'].unique()}, root=root)

    salesFinal_df = salesLineItems.merge(sales, how='left', on='invoiceNumber')

    return salesFinal_df

def getPurchases(url, header, startDate, root):
    """
    Retrieve purchase order line items and purchase orders from the E2 API and merge into a single DataFrame.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        startDate (str): Start date for filtering purchases (YYYY-MM-DD).
        root (str): Optional directory to save CSVs.

    Returns:
        pd.DataFrame: DataFrame of merged purchase order line items and purchase orders.
    """

    fields_dict = {
        'purchase-order-line-items': 'dateFinished,dueDate,itemNumber,lastModDate,outsideService,partDescription,partNumber,purchaseOrderNumber,quantityOrdered,quantityReceived,status,unit,unitCost',
        'purchase-orders': 'dateEntered,PONumber,status,vendorCode,vendorDescription',
        # 'purchase-order-releases': 'dateReceived,jobNumber,partNumber,PONumber,quantity'   
    }
    purchaseLineItems = getdB(url, header, 'purchase-order-line-items', fields_dict['purchase-order-line-items'], filterString='dateFinished[gt]=' + startDate, root=root)
    purchaseLineItems.rename(columns={'purchaseOrderNumber': 'PONumber'}, inplace=True)
    purchases = getdB(url, header, 'purchase-orders', fields_dict['purchase-orders'], search={'PONumber': purchaseLineItems['PONumber'].unique()}, root=root)
    # purchaseReleases = getdB(url, header, 'purchase-order-releases', fields_dict['purchase-order-releases'], df=purchases, searchfield='PONumber', root=root)
    
    # Merge the dataframes
    purchasesFinaldf = purchaseLineItems.merge(purchases, how='left', on='PONumber')
    # purchasesFinaldf = purchasesFinaldf.merge(purchaseReleases, how='left', on=['PONumber', 'partNumber'])

    
    return purchasesFinaldf

def getorderItems(url, header, startDate, root):
    """
    Retrieve order line items and related order data from the E2 API and merge into a single DataFrame.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        startDate (str): Start date for filtering orders (YYYY-MM-DD).
        root (str): Optional directory to save CSVs.

    Returns:
        pd.DataFrame: DataFrame of merged order line items and order data.
    """
    
    fields_dict = {
        'order-line-items': 'actualEndDate, actualStartDate, dateFinished, dueDate, estimatedEndDate, estimatedStartDate, itemNumber, jobNotes, jobNumber, masterJobNumber, orderNumber, partDescription, partNumber, pricingUnit, productCode, quantityCanceled, quantityOrdered, quantityShippedToCustomer, quantityShippedToStock, quantityToMake, quantityToStock, revision, status, totalActualHours, totalEstimatedHours, uniqueID, unitPrice, unitPriceForeign, workCode',
        'orders': 'customerCode, customerDescription, dateEntered, lastModDate, orderNumber, orderTotal, orderTotalForeign, PONumber, salesID, status, termsCode, territory, uniqueID'
    }

    orderItems_df = getdB(url, header, 'order-line-items', fields_dict['order-line-items'], filterString='dateFinished[gt]=' + startDate, root=root)
    orders_df = getdB(url, header, 'orders', fields_dict['orders'], search={'orderNumber': orderItems_df['orderNumber'].unique()}, root=root)
    # # sometimes someone is goofing around and entering a non-numeric order number
    # orders_df['orderNumber'] = pd.to_numeric(orders_df['orderNumber'], errors='coerce')
    orders_df = orders_df.merge(orderItems_df, how='left', on='orderNumber')

    return orders_df

def getJobs(url, header, startDate, root, orderItems_df):
    """
    Retrieve job routings, job materials, and estimates from the E2 API and merge them for job analysis.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        startDate (str): Start date for filtering jobs (YYYY-MM-DD).
        root (str): Optional directory to save CSVs.
        orderItems_df (pd.DataFrame): DataFrame of order items to match jobs.

    Returns:
        tuple: (jobs_df, materials_df, estimateMaterials_df)
    """

    fields_dict = {
        'order-routings': 'actualEndDate, actualStartDate, actualPiecesGood, burdenRate, description, employeeCode, estimatedQuantity, jobNumber, laborRate, operationCode, orderNumber,machinesRun, partNumber, status, stepNumber, timeUnit, totalActualHours, totalEstimatedHours, vendorCode, workCenter, workCenterOrVendor',
        'routings': 'partNumber, stepNumber, workCenter, vendorCode, description, setupTime, timeUnit, cycleTime, cycleUnit, burdenRate, laborRate, lastModDate',
        'job-materials': 'datePosted, description, jobNumber, orderNumber, outsideService, partNumber, postedFromStock, quantityPosted1, stepNumber, stockingCost, stockUnit',
        'materials': 'partNumber, subPartNumber, description, quantity, unit, unitPrice, unitCost'
    }

    # jobs data
    materials_df = getdB(url, header, 'job-materials', fields_dict['job-materials'], filterString='datePosted[gt]='+(startDate-relativedelta(years=1)).strftime('%Y-%m-%d'), root=root)
    orderRoutings_df = getdB(url, header, 'order-routings', fields_dict['order-routings'], search={'orderNumber': list(orderItems_df['orderNumber'].unique())}, root=root)

    # Estimates
    estimateMaterials_df = getdB(url, header, 'materials', fields_dict['materials'], root=root)
    estimateRoutings_df = getdB(url, header, 'routings', fields_dict['routings'], root=root)
    # Convert all times to hours
    estimateRoutings_df.rename(columns={'timeUnit': 'setupUnit'}, inplace=True)
    time_conversion = {'M': 60, 'S': 3600}
    for col in ['setupTime', 'cycleTime']:
        for unit, factor in time_conversion.items():
            estimateRoutings_df[col] = np.where(estimateRoutings_df[f'{col[:-4]}Unit'] == unit, estimateRoutings_df[col] / factor, estimateRoutings_df[col])
    estimateRoutings_df.drop(columns=['setupUnit', 'cycleUnit'], inplace=True)
    
    # Merge the dataframes
    jobs_df = orderRoutings_df.merge(estimateRoutings_df, on=['partNumber', 'stepNumber'], how='outer', suffixes=('', '_estimate'))

    return jobs_df, materials_df, estimateMaterials_df

## Tree functions
def makeTree(url, header, df):
    """
    Build a part tree from a DataFrame using E2 API data, returning the root node.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        df (pd.DataFrame): DataFrame with part numbers and descriptions.

    Returns:
        Node: Root node of the constructed tree.
    """
    # Create a root node
    root = Node("Root")

    # Create a dictionary to store nodes by part number for quick lookup
    nodes = [root]

    # Iterate through the dataframe and add all parts to the tree
    for part in df['partNumber'].unique():
        # Create nodes for each part and store them in the dictionary
        newNode = Node(part, parent=root)
        newNode.quantity = 1
        newNode.description = df[df['partNumber']== part]['partDescription'].values[0]
        nodes.append(newNode)

    nextlayer = grabTreeLayer(url, header, df)
    while not nextlayer.empty:
        nextlayer = nextlayer[nextlayer['Assembly'].notna()]  # create new nodes for parts that have a parent, delete the rest
        # Iterate through the dataframe and add nodes to the tree
        for assembly in nextlayer['Assembly'].unique():
            childParts = nextlayer[nextlayer['Assembly'] == assembly].drop_duplicates(subset='partNumber')
            # Get all occurrences of assembly in nodes_dict
            parent_nodes = [node for node in root.descendants if node.name == assembly]
            for parent_node in parent_nodes:
                # Check if the child node already exists
                for _, childRow in childParts.iterrows():
                    if childRow['partNumber'] not in [node.name for node in parent_node.children]:
                        # Create a new child node and add it to the dictionary
                        child_node = Node(childRow['partNumber'], parent=parent_node)
                        child_node.quantity = childRow['quantity']
                        child_node.description = childRow['partDescription']
                        nodes.append(child_node)

        nextlayer = grabTreeLayer(url, header, nextlayer)
            
        
    # root = pruneTree(root)

    return root

def pruneTree(tree):
    """
    Prune the tree by removing duplicate nodes, keeping only the deepest instance of each part.

    Args:
        tree (Node): Root node of the tree to prune.

    Returns:
        Node: Pruned tree root node.
    """
    nodeNames = set([node.name for node in tree.descendants])
    for node in nodeNames:
        nodeCopies = [j for j in tree.descendants if j.name == node]
        nodeDepths = [j.depth for j in nodeCopies]
        deepestNode = nodeCopies[nodeDepths.index(max(nodeDepths))]
        for scanNode in nodeCopies:
            if scanNode is not deepestNode:
                scanNode.parent = None

    return tree

def grabTreeLayer(url, header, df):
    """
    Retrieve the next layer of materials for the given parts from the E2 API and return as a DataFrame.

    Args:
        url (str): Base URL for the API.
        header (dict): HTTP headers for authentication.
        df (pd.DataFrame): DataFrame with part numbers to query.

    Returns:
        pd.DataFrame: DataFrame of the next layer of materials.
    """
    fields = 'partNumber,subPartNumber,description,quantity,unit,unitPrice'
    allMaterials = longSearchQuery(url + f'materials', header, fields, 'partNumber', list(df['partNumber'].unique()))
    df = pd.DataFrame(allMaterials)
    df = df.rename(columns={'description':'partDescription', 'partNumber':'Assembly', 'subPartNumber':'partNumber'})
    return df

def getTreeMaterials(tree, df, statistic):
    """
    Traverse the tree and collect all required materials for the given parts and statistic.

    Args:
        tree (Node): Root node of the part tree.
        df (pd.DataFrame): DataFrame with part numbers and statistics.
        statistic (str): Column name for the statistic to collect (e.g., 'quantityShipped').

    Returns:
        pd.DataFrame: DataFrame of required materials.
    """
    materials = pd.DataFrame(columns=['partNumber', statistic, 'partDescription'])
    moldBlankStrings = ['BK', 'KM', 'BM', 'BNK', 'BVM', 'JM', 'XM']
    for _, row in df.iterrows():
        part = row['partNumber']
        quantity = row[statistic]
        nodes = [node for node in tree.descendants if node.name == part]
        if nodes:
            node = nodes[0]
            getMaterials(node, materials, moldBlankStrings, quantity)
               
    return materials
    
def getMaterials(node, materials, moldBlankStrings, parentQuantity):
    """
    Recursively collect materials from a tree node, considering mold/blank nodes and parent quantity.

    Args:
        node (Node): Current node in the tree.
        materials (pd.DataFrame): DataFrame to append material info to.
        moldBlankStrings (list): List of strings identifying mold/blank nodes.
        parentQuantity (float): Quantity multiplier from parent node.
    """
    if not node:
        return

    if not node.children or any(s in node.name for s in moldBlankStrings): # record if there are no children or this node is a mold or blank
        materials.loc[len(materials)] = [node.name, node.quantity*parentQuantity, node.description]

    for child in node.children:
        getMaterials(child, materials, moldBlankStrings, node.quantity*parentQuantity)

def getTreeComponents(tree, df, statistic):
    """
    Traverse the tree and collect all components for the given parts and statistic.

    Args:
        tree (Node): Root node of the part tree.
        df (pd.DataFrame): DataFrame with part numbers and statistics.
        statistic (str): Column name for the statistic to collect.

    Returns:
        pd.DataFrame: DataFrame of required components.
    """
    # df = aggregateStatByPart(df, statistic)
    componentsdf = pd.DataFrame(columns=['partNumber', statistic, 'partDescription'])
    moldBlankStrings = ['BK', 'KM', 'BM', 'BNK', 'BVM', 'JM', 'XM']
    nodes = [node for node in tree.descendants if node.name in df['partNumber'].values]
    for _, row in df.iterrows():
        part = row['partNumber']
        quantity = row[statistic]
        nodes = [node for node in tree.descendants if node.name == part]
        node_parts = df[df['partNumber'].isin([node.name for node in nodes])]
        if nodes:
            node = min(nodes, key=lambda x: x.quantity)
            getComponents(node, componentsdf, moldBlankStrings, quantity)
    
    # componentsdf = aggregateStatByPart(componentsdf, statistic) # we aggregate before and after to get the correct quantities
    return componentsdf

def getComponents(node, components, moldBlankStrings, parentQuantity):
    """
    Recursively collect components from a tree node, considering mold/blank nodes and parent quantity.

    Args:
        node (Node): Current node in the tree.
        components (pd.DataFrame): DataFrame to append component info to.
        moldBlankStrings (list): List of strings identifying mold/blank nodes.
        parentQuantity (float): Quantity multiplier from parent node.
    """
    
    moldBlank = any(s in node.name for s in moldBlankStrings)
    moldBlankChildren = [child for child in node.children if any(s in child.name for s in moldBlankStrings)]
    Xraces = any(x in node.name for x in ['15X', '16X', '15EX', '16EX'])
    if (moldBlank or node.height<1) and not Xraces: # if there are no children, or this node is a mold or blank -- stop condition
        return   
    
    if node.height == 1 or moldBlankChildren or Xraces: # only record "parts" -- record condition 
        components.loc[len(components)] = [node.name, node.quantity*parentQuantity, node.description]

    for child in node.children:
        getComponents(child, components, moldBlankStrings, node.quantity*parentQuantity)

def plotTree(tree):
    """
    Print a visual representation of the tree structure to the console.

    Args:
        tree (Node): Root node of the tree to print.
    """
    for pre, _, node in RenderTree(tree):
        print(f"{pre}{node.name}")


def save_anytree_to_csv(root, filename):
    """
    Save an anytree structure to a CSV file, including name, description, and quantity for each node.

    Args:
        root (Node): Root node of the anytree structure.
        filename (str): Output CSV file name.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Node", "Parent", "Depth", "Description", "Quantity"])  # Column Headers

        for _, _, node in RenderTree(root):
            parent = node.parent.name if node.parent else "ROOT"
            description = getattr(node, "description", "N/A")  # Handle missing attributes
            quantity = getattr(node, "quantity", 0)  # Default to 0 if not set
            writer.writerow([node.name, parent, node.depth, description, quantity])

    print(f"Tree saved to {filename}")

def load_anytree_from_csv(filename):
    """
    Load an anytree structure from a CSV file and reconstruct the tree.

    Args:
        filename (str): Path to the CSV file to load.

    Returns:
        Node: Root node of the reconstructed tree.
    """
    nodes = {}  # Dictionary to store Node objects
    root = None  # Store the root node

    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        # First pass: Create all nodes
        for row in reader:
            node_name, parent_name, depth, description, quantity = row
            quantity = float(quantity)  # Convert quantity to integer

            # Create node with attributes
            node = Node(node_name, description=description, quantity=quantity)
            nodes[node_name] = node
            if parent_name == "ROOT":
                root = node  # Identify root node

    # Second pass: Assign parent-child relationships
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header

        for row in reader:
            node_name, parent_name, depth, description, quantity = row
            if parent_name != "ROOT":
                nodes[node_name].parent = nodes[parent_name]  # Link parent

    return root


## Umbrella functions

def cleanSales(df, statistic, datestat, includeStock=[]):
    """
    Clean and filter sales data, removing unwanted records and converting columns to appropriate types.

    Args:
        df (pd.DataFrame): Raw sales data.
        statistic (str): Column name for the sales statistic (e.g., 'quantityShipped').
        datestat (str): Column name for the date field (e.g., 'invoiceDate').
        includeStock (list): If not empty, include stock sales; otherwise, exclude.

    Returns:
        pd.DataFrame: Cleaned sales data.
    """
    df.dropna(subset=[datestat], inplace=True)

    # Clean up the data: Remove '$' signs and convert key columns to numeric
    df = df.fillna(0)
    df = df[[type(i) is str for i in df['partNumber']]]
    df = df[[type(i) is str for i in df['partDescription']]]

    if not includeStock:
        df = df[~df['customerCode'].str.contains('TOMAHAWK')]
    
    df = df[~df['partDescription'].str.contains(r'scrap|return|inspection|labor|freight|black|replace|maintenance|reweld|re-weld|bit ball|screw|bit|balls|set', case=False)]
    df = df[~df['partNumber'].str.contains(r'scrap|return|inspection|labor|freight|black|replace|balls|bit ball|set|screw|rw|skim', case=False)]

    #df = swapPartNumbers(df, 'Tomahawk') # categorization of part numbers requires replacement of some E2 pns
    df.reset_index(drop=True)

    df[statistic] = df[statistic].replace('[\$,%]', '', regex=True).astype(float)
    temp = [i.split('T')[0] for i in df[datestat]] 
    df[datestat] = pd.to_datetime(temp, errors='coerce')

   
    return df

def getMaterialsWInv(df, tree, inventory_df, statistic):
    """
    Calculate material requirements for parts, subtracting inventory and in-process quantities, and merge with inventory data.

    Args:
        df (pd.DataFrame): DataFrame with part demand and inventory info.
        tree (Node): Root node of the part tree.
        inventory_df (pd.DataFrame): DataFrame with inventory quantities.
        statistic (str): Column name for the demand statistic.

    Returns:
        pd.DataFrame: DataFrame of material requirements with inventory considered.
    """
    # subtract inventory and inProcess 
    df[statistic] = df[statistic]-(df['quantityOnHand']-df['quantityToMake'])
    df = df[df[statistic]>0]
    
    # Get materials from tree based on remaining parts not covered by inventory
    materials_df = getTreeMaterials(tree, df, statistic)
    materials_df = aggregateStatByPart(materials_df, statistic)
    
    # add inventory to the materials demand for comparison
    materialsWInv_df = materials_df.merge(inventory_df, how='left', on='partNumber')
    materialsWInv_df = materialsWInv_df[['partNumber', statistic, 'quantityOnHand', 'quantityOrdered']]
    materialsWInv_df.rename(columns={statistic: statistic + '_12mo'}, inplace=True)

    return materialsWInv_df

