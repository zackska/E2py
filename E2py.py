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
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for cluster_id in set(labels):
        cluster_docs = X[labels == cluster_id]
        mean_tfidf = cluster_docs.mean(axis=0).A1
        top_terms = [terms[i] for i in mean_tfidf.argsort()[-top_n:][::-1]]
        cluster_terms[cluster_id] = " ".join(top_terms)
    return cluster_terms    

def pivot_heatmap(df, sumrows, numberfmt, save_path, name, title, ylabel):
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
    # Aggregate the sales data by part number
    agg_rules = {col: 'first' for col in df.columns if col != statistic}
    agg_rules[statistic] = 'sum'
    df = df.groupby('partNumber').agg(agg_rules).reset_index(drop=True)
    return df

def swapPartNumbers(df, mode):
    
     # replace partnumbers for better parsing of motor size and category
    replacements =  {
        '10XD-PDM-LRJ-001': 'LM10017K',
        '550XD-PDM-LRB-001' : 'LM55005K', 
        '550XD-PDM-LRJ-001': 'LM55017K',
        '675XD-PDM-LRJ-005' : 'LM67043K',
        '550XD-PDM-URB-001': 'LM55045K',
        '500XD-PDM-URB-003' : 'LM50045K',
        '775XD-PDM-URB-001' : 'LM77045K',
        '70725XD-PDM-LRJ-001': 'LM70017K',
        '70725XD-PDM-LRB-001': 'LM70005K', 
        '775XD-PDM-LRJ-006': 'LM77043K',
        '775XD-PDM-LRLB-006': 'LM77005K',
        '850XD-PDM-LRB-001':'LM85005K',
        '725XD-PDM-UOCS-001B': 'LM72045K',
        '775XD-PDM-MRB-001': 'LM77045MK',
        '10564653' : 'BH31005K',
        '10605208': 'BH31005K-U035',
        '10552419': 'BH96005K',
        '10570432': 'BH72005K',
        '10570429': 'BH71044-ND20-K',
        'BH28605K': 'BH28005K',
        'BH28645K': 'BH28045K',
        'BH28644K': 'BH28044K',
        '10505932': 'BH33005K', 
        '10564701': 'BH31043K',
        '10290502': 'BH47045K',
        '10524838':'BH28043K',
        '10528579': 'BH51043K-D',
        '10528581': 'BH51044K-B',
        '10528582-B': 'BH51045K-B',
        '10528582': 'BH51045K',
        '10552416': 'BH96045K',
        '10552417': 'BH96044K',
        '10564701': 'BH31043K',
        '10564702': 'BH31045K',
        '10564703': 'BH31044K',
        '10605209': 'BH31043K-OS44',
        '10644502': 'BH80045-ND6-K',
        '10644503':'BH80044-ND6-K',
        '10543689': 'BH67044-ND7X-K',
        '10523381': 'BH67044-ND7X-K',
        'M8-5002-01A': 'GD80043K',
        'M8-4506-01': 'GD67043K',
        'M8-5154': 'GD80044K',
        'M8-6208-01':'GD96115K',
        'T-SNB-287-010': 'IV28044K', 
        'T-SNB-287-009': 'IV28045K',
        'SH70080RS': 'SH70080DS',
        'T96080X':'SH96080',
        'SH57005BK': 'SH57005KBK',
        '10541142': 'SH55005K',
    }

    if mode=='E2':
            for category, keyword in replacements.items():
                df.loc[df['partNumber'].str.contains(keyword), 'partNumber'] = category
    elif mode=='Tomahawk':
            for category, keyword in replacements.items():
                df.loc[df['partNumber'].str.contains(category), 'partNumber'] = keyword
    
   
    return df

def correctPN(df):
    corrections = {
        'SH47080V': 'SH47080',
        'SH52BAX-S-V-V-F': 'SH52BA-S-V-V-F'
    }
    for category, keyword in corrections.items():
        df.loc[df['partNumber'].str.contains(keyword), 'partNumber'] = category
    return df

def addMotorSize(df, motor_sizes):

    # Create a regex pattern from the list
    pattern = r'(' + '|[a-zA-Z]{2}'.join(motor_sizes) + r')'
    df['MotorSize'] = df['partNumber'].str.extract(pattern, expand=False)
    df['MotorSize'] = df['MotorSize'].fillna('00')
    df['MotorSize'] = df['MotorSize'].str.extract(r'(\d+)', expand=False)
    df.loc[[i == '7' for i in df['MotorSize']], 'MotorSize'] = 'X7'

    # # second pass to catch the old Tomahawk Parts
    # pattern = r'(' + '|[a-zA-Z]{1}'.join(motor_sizes) + r')'
    # T_df = df[]
    # df['MotorSize'] = df['partNumber'].str.extract(pattern, expand=False)
    # df['MotorSize'] = df['MotorSize'].fillna('00')
    # df['MotorSize'] = df['MotorSize'].str.extract(r'(\d+)', expand=False)
    # df.loc[[i == '7' for i in df['MotorSize']], 'MotorSize'] = 'X7'

    return df

def categorize(df, cluster_names=[]):
    
    if 'Category' not in df.columns:
        df['Category'] = 'none'

    if cluster_names:
        for category, keyword in cluster_names.items():
            # Filter parts that match the current category and have not been assigned yet
            if len(keyword) == 2:
                matches = df['partDescription'].str.contains(keyword[0], case=False, regex=True) &  df['partNumber'].str.replace(r'^\D+', '', regex = True).str.contains(keyword[1], case=False, regex=True) 
            elif len(keyword) == 3:
                matches = df['partDescription'].str.contains(keyword[0], case=False, regex=True) &  df['partNumber'].str.replace(r'^\D+', '', regex = True).str.contains(keyword[1], case=False, regex=True) &  ~df['partNumber'].str.contains(keyword[2], case=False, regex=True) 
            else: 
                matches = df['partDescription'].str.contains(keyword, case=False, regex=True)
                
            df.loc[matches & df['Category'].str.contains('none', case=False), 'Category'] = category

            #print(f"parts in {category} ...{matched_parts['partDescription']}")   
    else:
        # Use K-means and TFIDF on the remainder 
        # Step 1: Vectorize the partDescriptions
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['partDescription'])

        # Step 2: Cluster the partDescriptions
        num_clusters = int(np.ceil(len(df)/20)) # Specify the number of categories
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X)

        # Example usage with clusters
        labels = np.array(df['Cluster'])
        cluster_names = get_top_terms_per_cluster(X, labels, vectorizer, top_n=3)
        df['Category'] = df['Cluster'].map(cluster_names)

    return df

def checkCategories(df):
    # categorize the leftovers with k-means to evalute completeness
    leftovers_df = categorize(df[df['Category'].str.contains('none', case=False)])

    # print the clusters to terminal for manual checking
    for i in range(max(leftovers_df['Cluster'])):
        print(leftovers_df[leftovers_df['Cluster'] == i]['partDescription'].unique())
        print(leftovers_df[leftovers_df['Cluster'] == i]['partNumber'].unique())

def getBlanks(inventory_df, quartersales_df, statistic): 

    equivalentBlankMolds = {
        'SH96115KM': 'SH96005K',
        'SH67017KM': ['SH67017K', 'SS67017K'],
        'BK-SH5752017KBK': ['SH57017K', 'SH55017K', 'SS50017K', 'SH52017K'],
        'BK-SH7267017K': ['SH72017K', 'SS70017K', 'SSX70017K', 'SS67017K', 'SH67017K'],
        'BK-SH8077017KBK': ['SH80017K', 'SH77017K'],
        'BK-SH47001-F': ['SH47001', 'SH50001'],
        'BK-SH47001-FP': ['SH47001P', 'SH50001P'],
        'BKSH57001F': ['SH57001', 'SH55001', 'SH52001'],
        'BK-SH67001-F': ['SH67001', 'SS67001'],
        'BK-SH77001-F': ['SH77001', 'SH80001'],
        'BK-SH72001F': ['SH72001', 'SS72001'],
        'BK-SH67001-FP': ['SH67001P', 'SS67001P']
    }

     # create a dataframe for blanks and molds
    blanks_df = inventory_df[(inventory_df['Category'] == 'Blanks') | (inventory_df['Category'] == 'Molds')][['partNumber', 'finalPart', 'quantityOnHand', 'quantityOrdered']]
    blanks_df = blanks_df.merge(quartersales_df, how='left', left_on='finalPart', right_on='partNumber', suffixes=('', '_x'))
    blanks_df.drop('partNumber_x', axis=1, inplace=True)
    
    # Adjust inventory quantities for parts with 2X or 3X in their name
    for multiplier in ['2X', '3X']:
        multiplier_value = int(multiplier[0])
        # Filter parts with the multiplier in their name
        multiplier_parts = blanks_df[blanks_df['partNumber'].str.contains(multiplier)]
        for index, row in multiplier_parts.iterrows():
            base_part_number = row['partNumber'].replace(multiplier, '')
            if base_part_number in blanks_df['partNumber'].values:
                blanks_df.loc[blanks_df['partNumber'] == base_part_number, 'quantityOnHand'] += row['quantityOnHand'] * multiplier_value
                blanks_df.loc[blanks_df['partNumber'] == base_part_number, 'quantityOrdered'] += row['quantityOrdered'] * multiplier_value
            else:
            # If the base part is not found, add it to the blanks_df
                new_row = {'partNumber': base_part_number, 'quantityOnHand': row['quantityOnHand'] * multiplier_value, 'quantityOrdered': row['quantityOrdered'] * multiplier_value, 'finalPart': base_part_number.replace('KM', 'K')}
                blanks_df = pd.concat([blanks_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Remove multiplier parts from blanks_df
    blanks_df = blanks_df[~blanks_df['finalPart'].str.contains(r'2X|3X', regex=True)]
    
    for key, values in equivalentBlankMolds.items():
        if isinstance(values, list):
            total_quantity = quartersales_df[quartersales_df.index.isin(values)][statistic + '_quarter'].sum()
        else:
            total_quantity = quartersales_df[quartersales_df.index == values][statistic + '_quarter'].sum()
        if key in blanks_df['partNumber'].values:
            blanks_df.loc[blanks_df['partNumber'] == key, statistic + '_quarter'] = total_quantity
            if isinstance(values, list):
                blanks_df.loc[blanks_df['partNumber'] == key, 'finalPart'] = ','.join(values)
            else:
                blanks_df.loc[blanks_df['partNumber'] == key, 'finalPart'] = values

    # Only take real part numbers
    blanks_df = blanks_df[blanks_df['partNumber'].str.contains(r'SH|SS')]
    blanks_df.fillna(0, inplace=True)

    return blanks_df


## JSON request functions using E2 API    
def longSearchQuery(url, headers, fields, searchField, search):
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
    async with session.get(url + f'&take=200&skip={str(skip)}', headers=headers, params={'fields': fields}) as response:
        return await response.json()

async def get_all_data(url, headers, fields):
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
    openOrderQty = getdB(url, header, 'purchase-order-line-items', 'partNumber,quantityOrdered,quantityReceived,partDescription', filterString='status=Open')
    openOrderQty = openOrderQty.fillna(0)
    openOrderQty['quantityOrdered'] = openOrderQty['quantityOrdered']-openOrderQty['quantityReceived']
    if 'partDescription' in df.columns:
        openOrderQty.drop('partDescription', axis = 1)
    openOrderQty = openOrderQty.groupby('partNumber').agg({'quantityOrdered':'sum', 'partDescription':'first'})
    df = df.merge(openOrderQty, how='left', on='partNumber')
    return df

def getLeadTimes(purchases):
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
    loop = asyncio.get_event_loop()
    fields = 'employeeCode,employeeName,active,departmentNumber'
    employees = pd.DataFrame(loop.run_until_complete(get_all_data(url + f'employees', header, fields)))
    return employees

# functions that acquire multiple databases to form a composite dataframe
def getSales(url, header, startDate, root):
    fields_dict = {
        'ar-invoices': 'invoiceNumber,invoiceDate,customerCode',
        'ar-invoice-details': 'partNumber,partDescription,quantityShipped,unitPrice,invoiceNumber,jobNumber,discountPercent'
    }

    sales = getdB(url, header, 'ar-invoices', fields_dict['ar-invoices'], filterString='invoiceDate[gt]=' + startDate, root=root)
    salesLineItems = getdB(url, header, 'ar-invoice-details', fields_dict['ar-invoice-details'], search={'invoiceNumber': sales['invoiceNumber'].unique()}, root=root)

    salesFinal_df = salesLineItems.merge(sales, how='left', on='invoiceNumber')

    return salesFinal_df

def getPurchases(url, header, startDate, root):

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
    fields = 'partNumber,subPartNumber,description,quantity,unit,unitPrice'
    allMaterials = longSearchQuery(url + f'materials', header, fields, 'partNumber', list(df['partNumber'].unique()))
    df = pd.DataFrame(allMaterials)
    df = df.rename(columns={'description':'partDescription', 'partNumber':'Assembly', 'subPartNumber':'partNumber'})
    return df

def getTreeMaterials(tree, df, statistic):
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
    if not node:
        return

    if not node.children or any(s in node.name for s in moldBlankStrings): # record if there are no children or this node is a mold or blank
        materials.loc[len(materials)] = [node.name, node.quantity*parentQuantity, node.description]

    for child in node.children:
        getMaterials(child, materials, moldBlankStrings, node.quantity*parentQuantity)

def getTreeComponents(tree, df, statistic):
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
    for pre, _, node in RenderTree(tree):
        print(f"{pre}{node.name}")


def save_anytree_to_csv(root, filename):
    """
    Save an anytree structure to a CSV file, including name, description, and quantity.
    
    :param root: The root node of the anytree structure.
    :param filename: The output CSV file name.
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
def addSizeAndCategory(df, check=True):
  
      # Define part categories and remainder categories
    part_categories = {
        "Assemblies": [r"assembly|transmission", r'BA|080|094|021'],
        "Mandrels": [r"mandrel", '001'],
        "Inner Race": [r"race", r'015'],
        "Outer Race": [r"race", r'016'],
        "Lower Inner Radial": [r"bearing|radial|plain|journal|restrictor|end nut|tile|LBH", r'017|043'],
        "Lower Outer Radial": [r"bearing|radial|plain|journal|restrictor|end nut|tile|LBH", r'005|115'],
        "Upper Inner Radial": [r"bearing|radial|plain|journal|restrictor|diverter", r'044|002S|002J|002'],
        "Upper Outer Radial": [r"bearing|radial|plain|journal|restrictor", r'045|004'],
        "Driver": [r"driver|jaw clutch|coupling|lower|upper", r'018', r'RW'],
        "Jaws 3PC": [r"driven|jaw clutch|coupling", r'019C', r'RW'],
        "Jaw Clutch Center": [r"driven|jaw clutch|coupling|center|shaft", r'020', r'RW'],
        "Driven": [r"driven|jaw clutch|coupling", r'019', r'RW'],
        "Spacers": [r"spacer", r'029|028'],
        "Alignment Ring/WearPad": [r"alignment|wear pad", r'039'],
        "Bearing Housings": [r"bearing housing|slick|bearing", '003'],
        "Ball Seat": [r"ball seat|seat, ball", r'022'],
        "Ball Catch": [r"ball catch|catch, ball", r'023'],
        "Crossover or Combo Subs": [r"crossover|combo|cross", r'137|136|116'],
        'Compression Subs': [r"compression", r'097'],
        'Compression Tools': [r"compression", r'096'],
        "Flex Shafts": [r"flexshaft|flex-shaft", r'021'],
        "Flex Shaft Adapters": [r"flexshaft|flex-shaft", r'018T'],
        "Flow Diverters": [r"flow diverter", '002'],
        "Fixed Bend Housings": [r"fixed bend", '082'],
        "Adjustable Bent Housings": [r"abh|adjustable", r'030|031|033|039'],
        "Stabilizers": [r"stabilizer|wear pad", r'071|062|063'],
        "Extension Housings": [r"extension|straight", r'084|081'],
        "Rotor Catches": [r"catch", '109'],
        'Catch Rings': [r"ring", r'004|009'],
        "Top Subs": [r"top", '006'],
        "Steel": [r"x", r'TMSS0|1026|HH11', r'SH|IN|RW'],
        "Titanium": [r"TI6", r'TI6|Ti'],
        "Carbide": [r"carbide", r'TMSS'], 
    }

    # Categorize the remainders using only the descriptions and remove them from the dataframe
    remainder_categories = {
        "Purchased Components": r"balls|set screw",
        "Reworks and Maintenance":  r"rework|reworked|skim|remove|labor",
        "Gages" : r"gage|tpf|gauge|stub",
        # "Extension Housings": r"stator connection",
        "Power Sections": r"rotor|stator",
        "Inserts": r"mandrel insert",
        "Bit Balls": r"bit ball",
        "Screws": r"screw",
    }

    
    # Add motorsizes to the dataframe
    motor_sizes = ['16', '21', '23', '28', '31', '33', '35', '37', '47', '50', '51',
                '52', '55', '57', '62', '65', '66', '67', '70', '71', '72', '77', 
                '80', '82', '83', '87', '90', '96', 'X7']
    
    df = swapPartNumbers(df, 'Tomahawk') # this is only necessary for categorization and motor size 
    df = categorize(df, part_categories)
    # Remove parts that are not value-added    
    df = categorize(df, remainder_categories)
    df = df[~df['Category'].isin(remainder_categories.keys())] # remove the remainder categories
    
    if check:
        checkCategories(df)

    df = addMotorSize(df, motor_sizes)
    
    df = swapPartNumbers(df, 'E2') # this is only necessary for categorization and motor size 
    
    return df

def cleanSales(df, statistic, datestat, includeStock=[]):
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

