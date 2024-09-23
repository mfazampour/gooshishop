from typing import List

import jdatetime
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager
import numpy as np
import matplotlib.colors as mcolors
from SalesPlots import SalesPlotter

def read_data(file_path, sheet_name: str | List[str] =None):
    """Reads the Excel file and returns the specified sheet or all sheets as a dictionary."""
    try:
        if sheet_name is str:
            sheets_dict = pd.read_excel(file_path, sheet_name=sheet_name)
            sheets_dict = {sheet_name: sheets_dict}  # Wrap in a dictionary for consistency
        elif sheet_name is not None:
            sheets_dict = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            sheets_dict = pd.read_excel(file_path, sheet_name=None)
            sheets_dict = change_sheet_name_to_english(sheets_dict)
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

    # plot_sales_distribution(filtered_df.copy(), bot_workspace_df.copy(), number_of_products_to_plot=10)
    # plot_sales_distribution(filtered_df.copy(), bot_workspace_df.copy(), show_least_selling=True)

    plot_channel_distribution(filtered_df.copy())

    plots_for_digikala_channel(filtered_df)

    plots_gooshi_shop_channel(filtered_df)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import jdatetime
# import arabic_reshaper
# from bidi.algorithm import get_display

# def plot_overall_sales_per_hour(df, window_size=3):
#     plt.figure(figsize=(12, 8))
#
#     # Ensure 'زمان_میلادی' is in datetime format
#     df['زمان_میلادی'] = pd.to_datetime(df['زمان_میلادی'])
#
#     # Extract date and hour from 'زمان_میلادی'
#     df['date'] = df['زمان_میلادی'].dt.normalize()
#     df['hour'] = df['زمان_میلادی'].dt.hour
#
#     # Get the full date range
#     min_date = df['date'].min()
#     max_date = df['date'].max()
#     full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
#
#     # Create a DataFrame with all combinations of date and hour
#     all_hours = pd.DataFrame({'hour': range(24)})
#     date_hour_df = pd.MultiIndex.from_product(
#         [full_date_range, all_hours['hour']], names=['date', 'hour']
#     ).to_frame(index=False)
#
#     # Ensure 'date' columns are of datetime64[ns] type
#     df['date'] = pd.to_datetime(df['date'])
#     date_hour_df['date'] = pd.to_datetime(date_hour_df['date'])
#
#     # Group df by date and hour and count sales
#     hourly_sales_counts = df.groupby(['date', 'hour']).size().reset_index(name='sales_count')
#
#     # Merge date_hour_df with hourly_sales_counts
#     date_hour_df = date_hour_df.merge(hourly_sales_counts, on=['date', 'hour'], how='left')
#     date_hour_df['sales_count'] = date_hour_df['sales_count'].fillna(0)
#
#     # Group by 'hour' and compute mean and std
#     stats_by_hour = date_hour_df.groupby('hour')['sales_count'].agg(['mean', 'std'])
#     stats_by_hour = stats_by_hour.reindex(range(24), fill_value=0)
#
#     # Circularly extend the data for moving average
#     mean_extended = np.concatenate((
#         stats_by_hour['mean'][-(window_size//2):],
#         stats_by_hour['mean'],
#         stats_by_hour['mean'][:(window_size//2)]
#     ))
#     std_extended = np.concatenate((
#         stats_by_hour['std'][-(window_size//2):],
#         stats_by_hour['std'],
#         stats_by_hour['std'][:(window_size//2)]
#     ))
#
#     # Compute moving average on extended data
#     mean_moving_avg = pd.Series(mean_extended).rolling(window=window_size, center=True).mean()
#     std_moving_avg = pd.Series(std_extended).rolling(window=window_size, center=True).mean()
#
#     # Extract the central part corresponding to original data
#     start_idx = window_size//2
#     end_idx = start_idx + len(stats_by_hour)
#     mean_moving_avg = mean_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
#     std_moving_avg = std_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
#
#     # Plot the moving average of the mean
#     line, = plt.plot(stats_by_hour.index, mean_moving_avg, label='Overall Sales')
#
#     # Get the color of the line
#     color = line.get_color()
#
#     # Compute upper and lower bounds
#     upper_bound = mean_moving_avg + std_moving_avg
#     lower_bound = mean_moving_avg - std_moving_avg
#
#     # Fill between the upper and lower bounds with a light color
#     plt.fill_between(stats_by_hour.index, lower_bound, upper_bound, color=color, alpha=0.2)
#
#     # Set titles and labels with Farsi text
#     plt.title(get_display(arabic_reshaper.reshape('میانگین فروش ساعتی کل محصولات')), fontdict={'fontname': 'XB Zar', 'fontsize': 16})
#     plt.xlabel(get_display(arabic_reshaper.reshape('ساعت روز')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#     plt.ylabel(get_display(arabic_reshaper.reshape('میانگین فروش در ساعت')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#
#     plt.xticks(range(24))
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
# def plot_overall_sales_per_day_of_week(df, window_size=3):
#     plt.figure(figsize=(12, 8))
#
#     # Ensure 'زمان_میلادی' is in datetime format
#     df['زمان_میلادی'] = pd.to_datetime(df['زمان_میلادی'])
#
#     # Extract date and day of the week from 'زمان_میلادی'
#     df['date'] = df['زمان_میلادی'].dt.normalize()
#     df['day_of_week'] = df['زمان_میلادی'].dt.dayofweek  # 0=Monday, ..., 6=Sunday
#
#     # Get the full date range
#     min_date = df['date'].min()
#     max_date = df['date'].max()
#     full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
#
#     # Create a DataFrame with all combinations of date and day_of_week
#     date_df = pd.DataFrame({'date': full_date_range})
#     date_df['day_of_week'] = date_df['date'].dt.dayofweek
#
#     # Group df by date and count sales
#     daily_sales_counts = df.groupby('date').size().reset_index(name='sales_count')
#
#     # Merge date_df with daily_sales_counts
#     date_df = date_df.merge(daily_sales_counts, on='date', how='left')
#     date_df['sales_count'] = date_df['sales_count'].fillna(0)
#
#     # Group by 'day_of_week' and compute mean and std
#     stats_by_day = date_df.groupby('day_of_week')['sales_count'].agg(['mean', 'std'])
#     stats_by_day = stats_by_day.reindex(range(7), fill_value=0)
#
#     # Circularly extend the data for moving average
#     mean_extended = np.concatenate((
#         stats_by_day['mean'][-(window_size//2):],
#         stats_by_day['mean'],
#         stats_by_day['mean'][:(window_size//2)]
#     ))
#     std_extended = np.concatenate((
#         stats_by_day['std'][-(window_size//2):],
#         stats_by_day['std'],
#         stats_by_day['std'][:(window_size//2)]
#     ))
#
#     # Compute moving average on extended data
#     mean_moving_avg = pd.Series(mean_extended).rolling(window=window_size, center=True).mean()
#     std_moving_avg = pd.Series(std_extended).rolling(window=window_size, center=True).mean()
#
#     # Extract the central part corresponding to original data
#     start_idx = window_size//2
#     end_idx = start_idx + len(stats_by_day)
#     mean_moving_avg = mean_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
#     std_moving_avg = std_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
#
#     # Plot the moving average of the mean
#     line, = plt.plot(stats_by_day.index, mean_moving_avg, label='Overall Sales')
#
#     # Get the color of the line
#     color = line.get_color()
#
#     # Compute upper and lower bounds
#     upper_bound = mean_moving_avg + std_moving_avg
#     lower_bound = mean_moving_avg - std_moving_avg
#
#     # Fill between the upper and lower bounds with a light color
#     plt.fill_between(stats_by_day.index, lower_bound, upper_bound, color=color, alpha=0.2)
#
#     # Set titles and labels with Farsi text
#     plt.title(get_display(arabic_reshaper.reshape('میانگین فروش روزانه کل محصولات')), fontdict={'fontname': 'XB Zar', 'fontsize': 16})
#     plt.xlabel(get_display(arabic_reshaper.reshape('روز هفته')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#     plt.ylabel(get_display(arabic_reshaper.reshape('میانگین فروش در روز')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#
#     # Replace x-ticks with day names in Persian and reshape for RTL
#     days_in_persian = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه', 'شنبه', 'یکشنبه']
#     days_in_persian_reshaped = [get_display(arabic_reshaper.reshape(day)) for day in days_in_persian]
#     plt.xticks(range(7), days_in_persian_reshaped)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


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


# def plot_sales_distribution(filtered_df, bot_workspace_df, number_of_products_to_plot=6, show_least_selling=False):
#     # Step 1: Find the top 'کد تنوع' based on occurrences
#     if show_least_selling:
#         top_kod_tanavo = filtered_df['کد تنوع'].value_counts().nsmallest(number_of_products_to_plot * 2).index
#     else:
#         top_kod_tanavo = filtered_df['کد تنوع'].value_counts().nlargest(number_of_products_to_plot * 2).index
#
#     # Step 2: Convert 'زمان' from Jalali to Gregorian and then to datetime
#     filtered_df['زمان_میلادی'] = filtered_df['زمان'].apply(
#         lambda x: jdatetime.datetime.strptime(x, '%Y.%m.%d %H:%M').togregorian()
#     )
#
#     # Step 3: Function to get the product name based on 'کد محصول'
#     def get_product_name(df, kod_mahsul):
#         product_name = df[df['کد محصول'] == kod_mahsul]['نام محصول'].iloc[0]
#         reshaped_name = arabic_reshaper.reshape(product_name)
#         bidi_name = get_display(reshaped_name)
#         return bidi_name
#
#     # Step 4: Function to map 'کد تنوع' to 'کد محصول'
#     def get_kod_mahsul_for_kod_tanavo(site_prices_df, kod_tanavo):
#         kod_mahsul = site_prices_df[site_prices_df['کد تنوع'] == kod_tanavo]['کد محصول'].iloc[0]
#         return kod_mahsul
#
#     # Step 5: Function to get all 'کد تنوع' for a 'کد محصول'
#     def get_all_kod_tanavo_for_kod_mahsul(site_prices_df, kod_mahsul):
#         return site_prices_df[site_prices_df['کد محصول'] == kod_mahsul]['کد تنوع'].dropna().unique()
#
#     # Step 6: Function to plot moving averages for the summed sales of all 'کد تنوع' for each 'کد محصول'
#     def plot_moving_average_all(df, bot_workspace_df, top_kod_tanavo, window_size=5, show_least_selling=False):
#         plt.figure(figsize=(12, 8))
#
#         for kod_tanavo in top_kod_tanavo:
#             # Get the corresponding 'کد محصول'
#             kod_mahsul = get_kod_mahsul_for_kod_tanavo(bot_workspace_df, kod_tanavo)
#
#             # Get all 'کد تنوع' for this 'کد محصول'
#             all_kod_tanavo = get_all_kod_tanavo_for_kod_mahsul(bot_workspace_df, kod_mahsul)
#
#             # Filter the DataFrame for all 'کد تنوع' related to this 'کد محصول'
#             df_kod = df[df['کد تنوع'].isin(all_kod_tanavo)]
#
#             # Resample by day and calculate the count
#             daily_counts = df_kod.resample('D', on='زمان_میلادی').size()
#
#             # Get the first and last dates
#             start_date = df['زمان_میلادی'].min().date()
#             end_date = df['زمان_میلادی'].max().date()
#
#             # Create a full date range from start_date to end_date
#             full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
#
#             # Reindex the daily_counts DataFrame to include all dates in the range and fill missing values with zero
#             daily_counts = daily_counts.reindex(full_date_range, fill_value=0)
#
#             # Calculate the moving average
#             moving_avg = daily_counts.rolling(window=window_size).mean()
#
#             # Get the product name for the legend
#             product_name = get_product_name(bot_workspace_df, kod_mahsul)
#
#             # Plot with a label for the legend
#             plt.plot(moving_avg, label=f'{product_name}, {kod_mahsul}')
#
#         # Set titles and labels with Farsi text
#         title_text = 'Moving Average of Sales for Top Products' if not show_least_selling else 'Moving Average of Sales for Worst Products'
#         plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
#         plt.xlabel('Time', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#         plt.ylabel('Moving Average of Sales', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#
#         plt.grid(True, linestyle='--', alpha=0.6)
#         plt.legend(title='Product Name and ID')
#         plt.tight_layout()
#         plt.show()
#
#     # Step 7: Function to plot sales per hour for each product
#     def plot_sales_per_hour(df, bot_workspace_df, top_kod_tanavo, window_size=5, show_least_selling=False):
#         plt.figure(figsize=(12, 8))
#
#         for kod_tanavo in top_kod_tanavo:
#             # Get the corresponding 'کد محصول'
#             kod_mahsul = get_kod_mahsul_for_kod_tanavo(bot_workspace_df, kod_tanavo)
#
#             # Get all 'کد تنوع' for this 'کد محصول'
#             all_kod_tanavo = get_all_kod_tanavo_for_kod_mahsul(bot_workspace_df, kod_mahsul)
#
#             # Filter the DataFrame for all 'کد تنوع' related to this 'کد محصول'
#             df_kod = df[df['کد تنوع'].isin(all_kod_tanavo)].copy()
#
#             # Extract date and hour from 'زمان_میلادی'
#             # Use dt.normalize() to keep 'date' as datetime64[ns] at midnight
#             df_kod['date'] = df_kod['زمان_میلادی'].dt.normalize()
#             df_kod['hour'] = df_kod['زمان_میلادی'].dt.hour
#
#             # Get the full date range
#             min_date = df_kod['date'].min()
#             max_date = df_kod['date'].max()
#             full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
#
#             # Create a DataFrame with all combinations of date and hour
#             all_hours = pd.DataFrame({'hour': range(24)})
#             date_hour_df = pd.MultiIndex.from_product(
#                 [full_date_range, all_hours['hour']], names=['date', 'hour']
#             ).to_frame(index=False)
#
#             # Ensure 'date' columns are of datetime64[ns] type in both DataFrames
#             df_kod['date'] = pd.to_datetime(df_kod['date'])
#             date_hour_df['date'] = pd.to_datetime(date_hour_df['date'])
#
#             # Group df_kod by date and hour and count sales
#             hourly_sales_counts = df_kod.groupby(['date', 'hour']).size().reset_index(name='sales_count')
#
#             # Ensure 'date' in hourly_sales_counts is datetime64[ns]
#             hourly_sales_counts['date'] = pd.to_datetime(hourly_sales_counts['date'])
#
#             # Merge date_hour_df with hourly_sales_counts
#             date_hour_df = date_hour_df.merge(hourly_sales_counts, on=['date', 'hour'], how='left')
#             date_hour_df['sales_count'] = date_hour_df['sales_count'].fillna(0)
#
#             # Compute mean and std per hour, ensuring all hours are included
#             stats_by_hour = date_hour_df.groupby('hour')['sales_count'].agg(['mean', 'std']).reindex(range(24),
#                                                                                                      fill_value=0)
#
#             # Circularly extend the data for moving average
#             mean_extended = np.concatenate((
#                 stats_by_hour['mean'][-(window_size // 2):],  # Last half-window
#                 stats_by_hour['mean'],
#                 stats_by_hour['mean'][:(window_size // 2)]  # First half-window
#             ))
#             std_extended = np.concatenate((
#                 stats_by_hour['std'][-(window_size // 2):],
#                 stats_by_hour['std'],
#                 stats_by_hour['std'][:(window_size // 2)]
#             ))
#
#             # Compute moving average on extended data
#             mean_moving_avg = pd.Series(mean_extended).rolling(window=window_size, center=True).mean()
#             std_moving_avg = pd.Series(std_extended).rolling(window=window_size, center=True).mean()
#
#             # Extract the central part corresponding to original data
#             start_idx = window_size // 2
#             end_idx = start_idx + len(stats_by_hour)
#             mean_moving_avg = mean_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
#             std_moving_avg = std_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
#
#             # Get the product name for the legend
#             product_name = get_product_name(bot_workspace_df, kod_mahsul)
#
#             # Plot the moving average of the mean
#             line, = plt.plot(stats_by_hour.index, mean_moving_avg, label=f'{product_name}, {kod_mahsul}')
#
#             # Get the color of the line
#             color = line.get_color()
#
#             # Compute upper and lower bounds
#             upper_bound = mean_moving_avg + std_moving_avg / 4
#             lower_bound = mean_moving_avg - std_moving_avg / 4
#
#             # Fill between the upper and lower bounds with a light color
#             plt.fill_between(stats_by_hour.index, lower_bound, upper_bound, color=color, alpha=0.2)
#
#             # # Plot the mean line and get the line object
#             # line, = plt.plot(stats_by_hour.index, stats_by_hour['mean'], label=f'{product_name}, {kod_mahsul}')
#             #
#             # # Get the color of the line
#             # color = line.get_color()
#             #
#             # # Compute upper and lower bounds
#             # upper_bound = stats_by_hour['mean'] + stats_by_hour['std'] / 4
#             # lower_bound = stats_by_hour['mean'] - stats_by_hour['std'] / 4
#             #
#             # # Fill between the upper and lower bounds with a light color
#             # plt.fill_between(stats_by_hour.index, lower_bound, upper_bound, color=color, alpha=0.2)
#
#             # # Extract hour from 'زمان_میلادی'
#             # df_kod['hour'] = df_kod['زمان_میلادی'].dt.hour
#             #
#             # # Group by hour and count
#             # hourly_counts = df_kod.groupby('hour').size()
#             #
#             # # Reindex to include all hours (0-23)
#             # hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
#             #
#             # # **Compute the number of days in df_kod**
#             # min_date = df_kod['زمان_میلادی'].dt.date.min()
#             # max_date = df_kod['زمان_میلادی'].dt.date.max()
#             # num_days = (max_date - min_date).days + 1
#             #
#             # # **Divide hourly_counts by num_days to get average sales per hour per day**
#             # hourly_counts = hourly_counts / num_days
#             #
#             # # Calculate the moving average
#             # moving_avg = hourly_counts.rolling(window=window_size, center=True, min_periods=1).mean()
#             #
#             # # Get the product name for the legend
#             # product_name = get_product_name(bot_workspace_df, kod_mahsul)
#             #
#             # # Plot with a label for the legend
#             # plt.plot(hourly_counts.index, moving_avg, label=f'{product_name}, {kod_mahsul}')
#
#         # Set titles and labels with Farsi text
#         title_text = 'Average Sales per Hour per Day for Top Products with Std.' if not show_least_selling else 'Average Sales per Hour per Day for Worst Products'
#         plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
#         plt.xlabel('Hour of Day', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#         plt.ylabel('Average Sales per Day', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#
#         plt.xticks(range(24))
#         plt.grid(True, linestyle='--', alpha=0.6)
#         plt.legend(title='Product Name and ID')
#         plt.tight_layout()
#         plt.show()
#
#     # Step 8: Function to plot sales per day of the week for each product
#     def plot_sales_per_day_of_week(df, bot_workspace_df, top_kod_tanavo, show_least_selling=False):
#         plt.figure(figsize=(12, 8))
#
#         for kod_tanavo in top_kod_tanavo:
#             # Get the corresponding 'کد محصول'
#             kod_mahsul = get_kod_mahsul_for_kod_tanavo(bot_workspace_df, kod_tanavo)
#
#             # Get all 'کد تنوع' for this 'کد محصول'
#             all_kod_tanavo = get_all_kod_tanavo_for_kod_mahsul(bot_workspace_df, kod_mahsul)
#
#             # Filter the DataFrame for all 'کد تنوع' related to this 'کد محصول'
#             df_kod = df[df['کد تنوع'].isin(all_kod_tanavo)].copy()
#
#             # Extract date and day of the week from 'زمان_میلادی'
#             df_kod['date'] = df_kod['زمان_میلادی'].dt.date
#             df_kod['day_of_week'] = df_kod['زمان_میلادی'].dt.dayofweek  # 0=Monday, ..., 6=Sunday
#
#             # Get the full date range
#             min_date = df_kod['date'].min()
#             max_date = df_kod['date'].max()
#             full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
#
#             # Create a DataFrame with 'date' and 'day_of_week'
#             date_df = pd.DataFrame({'date': full_date_range})
#             date_df['day_of_week'] = date_df['date'].dt.dayofweek
#
#             # **Ensure 'date' columns are of datetime type in both DataFrames**
#             date_df['date'] = pd.to_datetime(date_df['date'])
#             df_kod['date'] = pd.to_datetime(df_kod['date'])
#
#             # Group df_kod by date and count sales
#             daily_sales_counts = df_kod.groupby('date').size().reset_index(name='sales_count')
#
#             # **Ensure 'date' column in daily_sales_counts is of datetime type**
#             daily_sales_counts['date'] = pd.to_datetime(daily_sales_counts['date'])
#
#             # Merge date_df with daily_sales_counts
#             date_df = date_df.merge(daily_sales_counts, on='date', how='left')
#             date_df['sales_count'] = date_df['sales_count'].fillna(0)
#
#             # Now, group by 'day_of_week' and compute mean and std
#             stats_by_day = date_df.groupby('day_of_week')['sales_count'].agg(['mean', 'std'])
#
#             # Get the product name for the legend
#             product_name = get_product_name(bot_workspace_df, kod_mahsul)
#
#             # Plot the mean line and get the line object
#             line, = plt.plot(stats_by_day.index, stats_by_day['mean'], label=f'{product_name}, {kod_mahsul}')
#
#             # Get the color of the line
#             color = line.get_color()
#
#             # Compute upper and lower bounds
#             upper_bound = stats_by_day['mean'] + stats_by_day['std'] / 4
#             lower_bound = stats_by_day['mean'] - stats_by_day['std'] / 4
#
#             # Fill between the upper and lower bounds with a light color
#             plt.fill_between(stats_by_day.index, lower_bound, upper_bound, color=color, alpha=0.2)
#
#             # Plot the mean with error bars (standard deviation)
#             # plt.errorbar(stats_by_day.index, stats_by_day['mean'], yerr=stats_by_day['std'],
#             #              label=f'{product_name}, {kod_mahsul}', capsize=5)
#
#             # # Group by day of the week and count sales
#             # dayofweek_counts = df_kod.groupby('day_of_week').size()
#             #
#             # # Reindex to include all days of the week (0-6)
#             # dayofweek_counts = dayofweek_counts.reindex(range(7), fill_value=0)
#             #
#             # # **Compute the total number of occurrences of each day of week in the date range**
#             # min_date = df_kod['زمان_میلادی'].dt.date.min()
#             # max_date = df_kod['زمان_میلادی'].dt.date.max()
#             # full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
#             # # day_of_week_counts_in_range = full_date_range.dayofweek.value_counts().sort_index()
#             # # day_of_week_counts_in_range = day_of_week_counts_in_range.reindex(range(7), fill_value=0)
#             #
#             # # **Divide dayofweek_counts by number of weeks to get average sales per day**
#             # average_sales = dayofweek_counts / (full_date_range / 7)
#             #
#             # # Get the product name for the legend
#             # product_name = get_product_name(bot_workspace_df, kod_mahsul)
#             #
#             # # Plot with a label for the legend
#             # plt.plot(average_sales.index, average_sales.values, label=f'{product_name}, {kod_mahsul}')
#
#         # Set titles and labels with Farsi text
#         title_text = 'Average Sales per Day of Week for Top Products with Std.' if not show_least_selling else 'Average Sales per Day of Week for Worst Products'
#         plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
#         plt.xlabel('Day of Week', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#         plt.ylabel('Average Sales per Day', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
#
#         # Replace x-ticks with day names in Persian
#         days_in_persian = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه', 'شنبه', 'یکشنبه']
#         # Reshape and reorder the day names for correct RTL display
#         days_in_persian_reshaped = [get_display(arabic_reshaper.reshape(day)) for day in days_in_persian]
#         plt.xticks(range(7), days_in_persian_reshaped)
#         plt.grid(True, linestyle='--', alpha=0.6)
#         plt.legend(title='Product Name and ID')
#         plt.tight_layout()
#         plt.show()
#
#     prod_id_list = []
#     filtered_top_kod_tanavo = []
#     for kod_tanavo in top_kod_tanavo:
#         kod_mahsul = get_kod_mahsul_for_kod_tanavo(bot_workspace_df, kod_tanavo)
#         if kod_mahsul not in prod_id_list:
#             prod_id_list.append(kod_mahsul)
#             filtered_top_kod_tanavo.append(kod_tanavo)
#             if len(filtered_top_kod_tanavo) == number_of_products_to_plot:
#                 break
#         else:
#             # we have this one already
#             continue
#
#     # Call the plotting functions
#     plot_moving_average_all(filtered_df, bot_workspace_df, filtered_top_kod_tanavo, show_least_selling=show_least_selling)
#
#     # only show this for top-selling products
#     if not show_least_selling:
#         plot_overall_sales_per_hour(filtered_df.copy())
#         plot_overall_sales_per_day_of_week(filtered_df.copy())
#         plot_sales_per_hour(filtered_df, bot_workspace_df, filtered_top_kod_tanavo, show_least_selling=show_least_selling)
#         plot_sales_per_day_of_week(filtered_df, bot_workspace_df, filtered_top_kod_tanavo, show_least_selling=show_least_selling)


def plots_gooshi_shop_channel(filtered_df):
    # Filter data for the 'گوشی شاپ' channel
    goshi_shop_df = filtered_df[filtered_df['کانال فروش'] == 'gooshishop']
    # Ensure the necessary column exists
    plot_gs_torob_pie_chart(goshi_shop_df)
    # Ensure the necessary columns exist
    goshi_shop_df_clean = plot_gs_torob_comparison(goshi_shop_df)

    plot_gs_torob_2d_hist(goshi_shop_df, goshi_shop_df_clean)


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


def plots_for_digikala_channel(filtered_df):
    # Filter data for the Digikala channel
    digikala_df = filtered_df[filtered_df['کانال فروش'] == 'digikala'].copy()
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
