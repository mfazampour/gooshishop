from typing import List

import jdatetime
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager
import numpy as np
import matplotlib.colors as mcolors
import os, pickle

from SalesPlots import SalesPlotter

# File path for the cache (you can adjust this as needed)
CACHE_FILE_PATH = 'data_cache.pkl'


def cache_data(data, cache_file=CACHE_FILE_PATH):
    """Save data to cache using pickle."""
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


def load_cached_data(cache_file=CACHE_FILE_PATH):
    """Load data from cache if available."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


def read_data(file_path, sheet_name: str | list = None):
    """Reads the Excel file and returns the specified sheet or all sheets as a dictionary."""
    # First, try to load from cache
    cached_data = load_cached_data()
    if cached_data is not None:
        print("Loaded data from cache.")
        return cached_data

    try:
        if sheet_name is str:
            sheets_dict = pd.read_excel(file_path, sheet_name=sheet_name)
            sheets_dict = {sheet_name: sheets_dict}  # Wrap in a dictionary for consistency
        elif sheet_name is not None:
            sheets_dict = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            sheets_dict = pd.read_excel(file_path, sheet_name=None)
            sheets_dict = change_sheet_name_to_english(sheets_dict)

        # Cache the data after reading it the first time
        cache_data(sheets_dict)
        return sheets_dict

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None


def change_sheet_name_to_english(sheets_dict):
    translation_map = {
        'اسفند': 'esfand',
        'فروردین': 'farvardin',
        'اردیبهشت': 'ordibehesht',
        'خرداد': 'khordad',
        'تیر': 'tir',
        'مرداد': 'mordad',
        'شهریور': 'shahrivar',
        'مهر': 'mehr',
        'آبان': 'aban',
        'آذر': 'azar',
        'دی': 'dey',
        'بهمن': 'bahman',
        'تیر 1403': 'tir_1403'
    }

    translated_sheets_dict = {}

    for sheet_name, df in sheets_dict.items():
        cleaned_name = sheet_name.strip()
        for farsi_name, english_name in translation_map.items():
            if farsi_name in cleaned_name:
                cleaned_name = cleaned_name.replace(farsi_name, english_name)
        translated_sheets_dict[cleaned_name] = df

    return translated_sheets_dict


def process_price_data(price_data_df, bot_workspace_df):
    if 'دسته گوشی شاپ' not in price_data_df.columns or 'کانال فروش' not in price_data_df.columns:
        print("Error: Required columns are missing from the PriceData sheet.")
        return

    print("Columns in 'PriceData':")
    print(price_data_df.columns)

    unique_values = price_data_df['دسته گوشی شاپ'].unique()
    print("\nUnique values in 'دسته گوشی شاپ':")
    print(unique_values)

    filtered_df = price_data_df[price_data_df['دسته گوشی شاپ'] == 'گوشی موبایل']
    print("\nFiltered DataFrame:")
    print(filtered_df)

    sales_plotter = SalesPlotter(filtered_df, bot_workspace_df)
    sales_plotter.plot_sales_distribution(show_least_selling=False, number_of_products_to_plot=10)
    sales_plotter.plot_sales_distribution(show_least_selling=True)


    plot_channel_distribution(filtered_df.copy())

    plots_for_digikala_channel(filtered_df, bot_workspace_df)

    plots_gooshi_shop_channel(filtered_df, bot_workspace_df)


def plot_channel_distribution(filtered_df):
    channel_counts = filtered_df['کانال فروش'].value_counts()
    # Reshape and fix the bidi order of the title
    reshaped_text = arabic_reshaper.reshape('توزیع کانال فروش برای گوشی موبایل')
    bidi_text = get_display(reshaped_text)
    plt.figure(figsize=(8, 8))
    plt.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', startangle=140)
    # Set the title with the correct Farsi text
    plt.title(bidi_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
    plt.show()


def join_filtered_with_bot_workspace(filtered_df, bot_workspace_df):
    """
    Join filtered_df with BotWorkSpace to have 'کد محصول' for each 'کد تنوع' in filtered_df.

    Parameters:
    filtered_df (DataFrame): DataFrame containing filtered data.
    bot_workspace_df (DataFrame): DataFrame containing the BotWorkSpace data with 'کد محصول' and 'کد تنوع'.

    Returns:
    DataFrame: The merged DataFrame with 'کد محصول' added to filtered_df.
    """
    # Perform a left join to add 'کد محصول' from bot_workspace_df based on 'کد تنوع'
    merged_df = filtered_df.merge(bot_workspace_df[['کد تنوع', 'کد محصول']], on='کد تنوع', how='left')
    return merged_df


def filter_and_return_product_ids_by_torob_ranking(merged_df, torob_ranking_filter):
    """
    Filter the merged DataFrame by 'رتبه ترب', return product IDs with more than 20 occurrences
    and filtered DataFrame sorted by frequency.

    Parameters:
    merged_df (DataFrame): DataFrame containing the filtered and joined data.
    torob_ranking_filter (str or list of str): The رتبه ترب value(s) to filter by.

    Returns:
    frequent_prod_ids_in_ranking (Series): Product IDs with more than 20 occurrences, sorted by frequency.
    ranking_filtered_df (DataFrame): Filtered DataFrame based on the provided 'رتبه ترب' filter.
    """
    # Ensure torob_ranking_filter is a list
    if isinstance(torob_ranking_filter, str):
        torob_ranking_filter = [torob_ranking_filter]

    # Filter the DataFrame based on the 'رتبه ترب' column
    ranking_filtered_df = merged_df[merged_df['رتبه ترب'].isin(torob_ranking_filter)]

    # Count the occurrences of each 'کد محصول'
    frequent_prod_ids_in_ranking = ranking_filtered_df['کد محصول'].value_counts()

    # Filter to include only products with more than 20 occurrences
    frequent_prod_ids_in_ranking = frequent_prod_ids_in_ranking[frequent_prod_ids_in_ranking > 20]

    return frequent_prod_ids_in_ranking, ranking_filtered_df


def convert_jalali_to_gregorian(jalali_datetime_str):
    """
    Convert Jalali datetime string to Gregorian datetime.

    Parameters:
    jalali_datetime_str (str): Jalali datetime string in the format 'YYYY.MM.DD HH:MM'.

    Returns:
    datetime: Corresponding Gregorian datetime object.
    """
    return jdatetime.datetime.strptime(jalali_datetime_str, '%Y.%m.%d %H:%M').togregorian()


def calculate_speed_of_occurrences(frequent_prod_ids_in_ranking, ranking_filtered_df, torob_ranking_filter):
    """
    Calculate the speed of occurrences of a specific ranking for each frequent product ID
    at the time of each occurrence in 3-hour time spans.

    Parameters:
    frequent_prod_ids_in_ranking (Series): Product IDs with more than 20 occurrences, sorted by frequency.
    ranking_filtered_df (DataFrame): Filtered DataFrame based on the provided 'رتبه ترب' filter.
    torob_ranking_filter (str or list of str): The ranking to calculate speed for (e.g., '1', '2', '3').

    Returns:
    speed_of_occurrences_df (DataFrame): DataFrame containing the product ID, timestamp, and speed of occurrences.
    """
    # Ensure torob_ranking_filter is a list
    if isinstance(torob_ranking_filter, str):
        torob_ranking_filter = [torob_ranking_filter]

    # Prepare the DataFrame to store the speed of occurrences
    speed_of_occurrences_data = []

    # Convert 'زمان' from Jalali to Gregorian
    ranking_filtered_df['زمان_میلادی'] = ranking_filtered_df['زمان'].apply(convert_jalali_to_gregorian)

    # Iterate over each frequent product ID
    for product_id in frequent_prod_ids_in_ranking.index:
        # Filter the DataFrame for the specific product ID
        product_df = ranking_filtered_df[ranking_filtered_df['کد محصول'] == product_id]

        # Further filter based on the 'رتبه ترب' values
        product_ranking_df = product_df[product_df['رتبه ترب'].isin(torob_ranking_filter)]

        # Sort the DataFrame by 'زمان_میلادی'
        product_ranking_df = product_ranking_df.sort_values(by='زمان_میلادی')

        # Iterate through the 'زمان_میلادی' column to calculate speed at each occurrence
        times = product_ranking_df['زمان_میلادی'].values
        for i, current_time in enumerate(times):
            # Initialize count of occurrences within the past 3-hour window
            occurrence_count = 0

            threshold = 5  # hours
            # Check for occurrences before the current time and within 3 hours
            for j in range(i):
                previous_time = times[j]
                if (current_time - previous_time) <= pd.Timedelta(hours=threshold):
                    occurrence_count += 1

            # If there is at least one occurrence in the past 3 hours, calculate the speed
            if occurrence_count > 3:
                # # Calculate the time span in hours from the first occurrence to the current occurrence
                # total_time_span_hours = (current_time - times[0]).total_seconds() / 3600

                # Calculate the rate of occurrence over 3-hour periods
                rate_of_occurrence = occurrence_count / threshold

                # Append the result to the list
                speed_of_occurrences_data.append([product_id, current_time, rate_of_occurrence])

    # Convert the result to a DataFrame
    speed_of_occurrences_df = pd.DataFrame(speed_of_occurrences_data,
                                           columns=['کد محصول', 'زمان_میلادی', 'Rate of Occurrence'])

    return speed_of_occurrences_df


def filter_and_print_product_ids_by_torob_rank(merged_df, rank_filter):
    """
    Filter the merged DataFrame by 'رتبه ترب', print product IDs for the matching records sorted by frequency.

    Parameters:
    merged_df (DataFrame): DataFrame containing the filtered and joined data.
    رتبه_ترب_filter (str or list of str): The رتبه ترب value(s) to filter by.

    Returns:
    None: Prints the sorted list of product IDs by frequency.
    """
    # Ensure رتبه ترب filter is a list
    if isinstance(rank_filter, str):
        rank_filter = [rank_filter]

    # Filter the DataFrame based on the 'رتبه ترب' column
    filtered_df = merged_df[merged_df['رتبه ترب'].isin(rank_filter)]

    # Count the occurrences of each 'کد محصول'
    product_counts = filtered_df['کد محصول'].value_counts()

    # Print the product IDs sorted by their frequency
    print("Product IDs sorted by frequency:")
    frequent_prod_ids = []
    for product_id, count in product_counts.items():
        if count > 20:
            frequent_prod_ids.append(product_id)
        print(f'کد محصول: {product_id}, Frequency: {count}')

    return frequent_prod_ids, filtered_df


def plots_gooshi_shop_channel(filtered_df, bot_workspace_df):
    # Filter data for the 'گوشی شاپ' channel
    goshi_shop_df = filtered_df[filtered_df['کانال فروش'] == 'gooshishop']

    sales_plotter = SalesPlotter(goshi_shop_df, bot_workspace_df, sale_channel='گوشی‌شاپ')
    sales_plotter.plot_sales_distribution(show_least_selling=False, number_of_products_to_plot=10)

    # Ensure the necessary column exists
    plot_gs_torob_pie_chart(goshi_shop_df)
    # Ensure the necessary columns exist
    goshi_shop_df_clean = plot_gs_torob_comparison(goshi_shop_df)

    plot_gs_torob_2d_hist(goshi_shop_df, goshi_shop_df_clean)

    gs_df_merged = join_filtered_with_bot_workspace(goshi_shop_df_clean, bot_workspace_df=bot_workspace_df)
    # frequent_prod_ids_in_ranking, ranking_filtered_df = filter_and_print_product_ids_by_torob_rank(merged_df=gs_df_merged, rank_filter=['1', '2', '3'])

    # Step 2: Get frequent product IDs and ranking-filtered DataFrame
    frequent_prod_ids_in_ranking, ranking_filtered_df = filter_and_return_product_ids_by_torob_ranking(gs_df_merged,
                                                                                                       ['1', '2', '3'])

    # Step 3: Calculate the speed of occurrences for each frequent product
    speed_of_occurrences_df = calculate_speed_of_occurrences(frequent_prod_ids_in_ranking, ranking_filtered_df,
                                                             ['1', '2', '3'])

    # Display the speed of occurrences DataFrame
    print(speed_of_occurrences_df)


def plot_gs_torob_pie_chart(goshi_shop_df):
    if 'رتبه ترب' in goshi_shop_df.columns:
        # Convert values to numeric where possible and group values larger than 10
        roteba_torob_counts = goshi_shop_df['رتبه ترب'].apply(pd.to_numeric, errors='coerce')

        # Group values greater than 10 into a single category
        roteba_torob_counts = roteba_torob_counts.apply(lambda x: '10+' if pd.notna(x) and x >= 10 else x)

        # Convert NaNs back to original values
        roteba_torob_counts = roteba_torob_counts.fillna(goshi_shop_df['رتبه ترب'])

        # Get the value counts
        roteba_torob_counts = roteba_torob_counts.value_counts()

        # Define autopct function to hide percentages less than 3%
        def autopct_func(pct):
            return ('%1.1f%%' % pct) if pct >= 3 else ''

        # Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(roteba_torob_counts, labels=roteba_torob_counts.index, autopct=autopct_func, startangle=140)

        # Set the title with the correct Farsi text
        plt.title(get_display(arabic_reshaper.reshape('توزیع رتبه ترب برای گوشی شاپ')),
                  fontdict={'fontname': 'XB Zar', 'fontsize': 16})

        plt.show()
    else:
        print("Error: The column 'رتبه ترب' is missing from the گوشی شاپ channel data.")


def plot_gs_torob_2d_hist(goshi_shop_df: pd.DataFrame, goshi_shop_df_clean):
    # Step 1: Preprocess the `roteba_torob_counts` values
    # Create a new column in goshi_shop_df_clean with specific replacements
    goshi_shop_df = goshi_shop_df.copy()
    # Replace +adv with -1
    goshi_shop_df['roteba_torob'] = goshi_shop_df['رتبه ترب'].replace({'adv': -1, '+30': 11})
    goshi_shop_df.loc[:, 'roteba_torob'] = goshi_shop_df['roteba_torob'].apply(pd.to_numeric, errors='coerce')
    # Cap values equal to or greater than 10 at 10
    goshi_shop_df.loc[:, 'roteba_torob'] = goshi_shop_df['roteba_torob'].apply(lambda x: 10 if x >= 10 else x)
    goshi_shop_df_clean = goshi_shop_df_clean.merge(goshi_shop_df[['#', 'roteba_torob']], on='#', how='left')
    # Remove any rows where `roteba_torob` is NaN
    goshi_shop_df_clean = goshi_shop_df_clean.dropna(subset=['roteba_torob'])
    # Cap the percentage differences to the range -20% to 20%
    goshi_shop_df_clean['perc_diff_قیمت_فروش_قیمت_اول_ترب'] = goshi_shop_df_clean[
        'perc_diff_قیمت_فروش_قیمت_اول_ترب'].apply(lambda x: max(min(x, 20), -20))
    # Step 2: Create the 2D histogram
    plt.figure(figsize=(10, 8))
    hist, xedges, yedges, _ = plt.hist2d(goshi_shop_df_clean['roteba_torob'],
                                         goshi_shop_df_clean['perc_diff_قیمت_فروش_قیمت_اول_ترب'], bins=[11, 50],
                                         cmap='plasma', norm=mcolors.Normalize(vmin=0, vmax=35))
    # Add colorbar for reference
    plt.colorbar(label=get_display(arabic_reshaper.reshape('تعداد')))
    # Set titles and labels with Farsi text
    plt.title(get_display(arabic_reshaper.reshape('ارتباط بین رتبه ترب و درصد اختلاف قیمت')),
              fontdict={'fontname': 'XB Zar', 'fontsize': 16})
    plt.xlabel(get_display(arabic_reshaper.reshape('رتبه ترب')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})
    plt.ylabel(get_display(arabic_reshaper.reshape('درصد اختلاف قیمت')),
               fontdict={'fontname': 'XB Zar', 'fontsize': 14})
    # Customize the X-axis labels to replace -1 with (+adv)
    x_ticks_labels = [get_display(arabic_reshaper.reshape('(+adv)')) if x == -1 else int(x) for x in xedges[:-1]]
    plt.xticks(xedges[:-1], x_ticks_labels)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_gs_torob_comparison(goshi_shop_df):
    # Ensure the necessary columns exist
    goshi_shop_df = goshi_shop_df.copy()

    if 'قیمت فروش' in goshi_shop_df.columns and 'قیمت اول ترب' in goshi_shop_df.columns:
        # Calculate the percentage difference
        goshi_shop_df.loc[:, 'perc_diff_قیمت_فروش_قیمت_اول_ترب'] = 100 * (
                goshi_shop_df['قیمت فروش'] - goshi_shop_df['قیمت اول ترب']) / goshi_shop_df['قیمت اول ترب']

        # Drop rows with infinite or NaN percentage differences
        goshi_shop_df_clean = goshi_shop_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=['perc_diff_قیمت_فروش_قیمت_اول_ترب'])

        # Remove outliers where the percentage difference is more than 100%
        goshi_shop_df_clean = goshi_shop_df_clean[goshi_shop_df_clean['perc_diff_قیمت_فروش_قیمت_اول_ترب'].abs() <= 100]

        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(goshi_shop_df_clean['perc_diff_قیمت_فروش_قیمت_اول_ترب'], bins=50, color='purple', edgecolor='black')

        # Set titles and labels with Farsi text
        title_text = get_display(arabic_reshaper.reshape('درصد اختلاف قیمت فروش و قیمت اول ترب'))
        xlabel_text = get_display(arabic_reshaper.reshape('درصد اختلاف قیمت'))
        ylabel_text = get_display(arabic_reshaper.reshape('تعداد'))

        plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel(xlabel_text, fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel(ylabel_text, fontdict={'fontname': 'XB Zar', 'fontsize': 14})

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("Error: The necessary columns 'قیمت فروش' and 'قیمت اول ترب' are missing from the گوشی شاپ channel data.")
    return goshi_shop_df_clean


def plots_for_digikala_channel(filtered_df, bot_workspace_df):
    # Filter data for the Digikala channel
    digikala_df = filtered_df[filtered_df['کانال فروش'] == 'digikala'].copy()

    sales_plotter = SalesPlotter(digikala_df, bot_workspace_df, sale_channel='دیجی‌کالا')
    sales_plotter.plot_sales_distribution(show_least_selling=False, number_of_products_to_plot=10)

    # Ensure the necessary columns exist
    if 'قیمت فروش' in digikala_df.columns and 'قیمت دیجی' in digikala_df.columns and 'قیمت دیجی اکسترا' in digikala_df.columns:
        # Calculate the percentage differences
        digikala_df['perc_diff_قیمت_فروش_دیجی'] = 100 * (digikala_df['قیمت فروش'] - digikala_df['قیمت دیجی']) / \
                                                  digikala_df['قیمت دیجی']
        digikala_df['perc_diff_قیمت_فروش_دیجی_اکسترا'] = 100 * (
                digikala_df['قیمت فروش'] - digikala_df['قیمت دیجی اکسترا']) / digikala_df['قیمت دیجی اکسترا']

        # Create histograms
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(digikala_df['perc_diff_قیمت_فروش_دیجی'], bins=20, color='blue', edgecolor='black')
        plt.title(get_display(arabic_reshaper.reshape('درصد اختلاف قیمت فروش و قیمت دیجی')),
                  fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.xlabel(get_display(arabic_reshaper.reshape('درصد اختلاف قیمت')))
        plt.ylabel(get_display(arabic_reshaper.reshape('تعداد')))

        plt.subplot(1, 2, 2)
        plt.hist(digikala_df['perc_diff_قیمت_فروش_دیجی_اکسترا'], bins=20, color='green', edgecolor='black')
        plt.title(get_display(arabic_reshaper.reshape('درصد اختلاف قیمت فروش و قیمت دیجی اکسترا')),
                  fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.xlabel(get_display(arabic_reshaper.reshape('درصد اختلاف قیمت')))
        plt.ylabel(get_display(arabic_reshaper.reshape('تعداد')))

        plt.tight_layout()
        plt.show()
    else:
        print("Error: Required columns for Digikala price comparisons are missing.")


# def save_filtered_data(filtered_df, file_name):
#     try:
#         filtered_df.to_excel(file_name, index=False)
#         print(f"Filtered data saved to {file_name}")
#     except Exception as e:
#         print(f"Error saving filtered data: {e}")

if __name__ == '__main__':
    file_path = 'ثبت روزانه سبدهای پیشنهادی تامین کنندگان-3.xlsx'
    sheets_dict = read_data(file_path, ['PriceData', 'BotWorkSpace', 'SellData'])

    if sheets_dict:
        if 'PriceData' in sheets_dict and 'BotWorkSpace' in sheets_dict:
            price_data_df = sheets_dict['PriceData']
            process_price_data(price_data_df, bot_workspace_df=sheets_dict['BotWorkSpace'])
        else:
            print("Error: 'PriceData' sheet not found in the Excel file.")
