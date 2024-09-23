from typing import List

import jdatetime
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager
import numpy as np
import matplotlib.colors as mcolors


class SalesPlotter:
    def __init__(self, filtered_df, bot_workspace_df):
        self.filtered_df = filtered_df.copy()
        self.bot_workspace_df = bot_workspace_df.copy()

        # Convert 'زمان' from Jalali to Gregorian and then to datetime
        self.filtered_df['زمان_میلادی'] = self.filtered_df['زمان'].apply(
            lambda x: jdatetime.datetime.strptime(x, '%Y.%m.%d %H:%M').togregorian()
        )

    def get_top_kod_tanavo(self, number_of_products_to_plot=6, show_least_selling=False):
        if show_least_selling:
            top_kod_tanavo = self.filtered_df['کد تنوع'].value_counts().nsmallest(number_of_products_to_plot * 2).index
        else:
            top_kod_tanavo = self.filtered_df['کد تنوع'].value_counts().nlargest(number_of_products_to_plot * 2).index
        return top_kod_tanavo

    def get_product_name(self, kod_mahsul):
        product_name = self.bot_workspace_df[self.bot_workspace_df['کد محصول'] == kod_mahsul]['نام محصول'].iloc[0]
        reshaped_name = arabic_reshaper.reshape(product_name)
        bidi_name = get_display(reshaped_name)
        return bidi_name

    def get_kod_mahsul_for_kod_tanavo(self, kod_tanavo):
        kod_mahsul = self.bot_workspace_df[self.bot_workspace_df['کد تنوع'] == kod_tanavo]['کد محصول'].iloc[0]
        return kod_mahsul

    def get_all_kod_tanavo_for_kod_mahsul(self, kod_mahsul):
        return self.bot_workspace_df[self.bot_workspace_df['کد محصول'] == kod_mahsul]['کد تنوع'].dropna().unique()

    def filter_top_kod_tanavo(self, top_kod_tanavo, number_of_products_to_plot):
        prod_id_list = []
        filtered_top_kod_tanavo = []
        for kod_tanavo in top_kod_tanavo:
            kod_mahsul = self.get_kod_mahsul_for_kod_tanavo(kod_tanavo)
            if kod_mahsul not in prod_id_list:
                prod_id_list.append(kod_mahsul)
                filtered_top_kod_tanavo.append(kod_tanavo)
                if len(filtered_top_kod_tanavo) == number_of_products_to_plot:
                    break
            else:
                continue
        return filtered_top_kod_tanavo

    def plot_moving_average_all(self, filtered_top_kod_tanavo, window_size=5, show_least_selling=False):
        plt.figure(figsize=(12, 8))

        for kod_tanavo in filtered_top_kod_tanavo:
            # Get the corresponding 'کد محصول'
            kod_mahsul = self.get_kod_mahsul_for_kod_tanavo(kod_tanavo)

            # Get all 'کد تنوع' for this 'کد محصول'
            all_kod_tanavo = self.get_all_kod_tanavo_for_kod_mahsul(kod_mahsul)

            # Filter the DataFrame for all 'کد تنوع' related to this 'کد محصول'
            df_kod = self.filtered_df[self.filtered_df['کد تنوع'].isin(all_kod_tanavo)]

            # Resample by day and calculate the count
            daily_counts = df_kod.resample('D', on='زمان_میلادی').size()

            # Get the first and last dates
            start_date = self.filtered_df['زمان_میلادی'].min().date()
            end_date = self.filtered_df['زمان_میلادی'].max().date()

            # Create a full date range from start_date to end_date
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Reindex the daily_counts DataFrame to include all dates in the range and fill missing values with zero
            daily_counts = daily_counts.reindex(full_date_range, fill_value=0)

            # Calculate the moving average
            moving_avg = daily_counts.rolling(window=window_size).mean()

            # Get the product name for the legend
            product_name = self.get_product_name(kod_mahsul)

            # Plot with a label for the legend
            plt.plot(moving_avg, label=f'{product_name}, {kod_mahsul}')

        # Set titles and labels with Farsi text
        title_text = 'Moving Average of Sales for Top Products' if not show_least_selling else 'Moving Average of Sales for Worst Products'
        plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel('Time', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel('Moving Average of Sales', fontdict={'fontname': 'XB Zar', 'fontsize': 14})

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Product Name and ID')
        plt.tight_layout()
        plt.show()

    def plot_sales_per_hour(self, filtered_top_kod_tanavo, window_size=5, show_least_selling=False):
        plt.figure(figsize=(12, 8))

        for kod_tanavo in filtered_top_kod_tanavo:
            # Get the corresponding 'کد محصول'
            kod_mahsul = self.get_kod_mahsul_for_kod_tanavo(kod_tanavo)

            # Get all 'کد تنوع' for this 'کد محصول'
            all_kod_tanavo = self.get_all_kod_tanavo_for_kod_mahsul(kod_mahsul)

            # Filter the DataFrame for all 'کد تنوع' related to this 'کد محصول'
            df_kod = self.filtered_df[self.filtered_df['کد تنوع'].isin(all_kod_tanavo)].copy()

            # Extract date and hour from 'زمان_میلادی'
            df_kod['date'] = df_kod['زمان_میلادی'].dt.normalize()
            df_kod['hour'] = df_kod['زمان_میلادی'].dt.hour

            # Get the full date range
            min_date = df_kod['date'].min()
            max_date = df_kod['date'].max()
            full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

            # Create a DataFrame with all combinations of date and hour
            all_hours = pd.DataFrame({'hour': range(24)})
            date_hour_df = pd.MultiIndex.from_product(
                [full_date_range, all_hours['hour']], names=['date', 'hour']
            ).to_frame(index=False)

            # Ensure 'date' columns are of datetime64[ns] type in both DataFrames
            df_kod['date'] = pd.to_datetime(df_kod['date'])
            date_hour_df['date'] = pd.to_datetime(date_hour_df['date'])

            # Group df_kod by date and hour and count sales
            hourly_sales_counts = df_kod.groupby(['date', 'hour']).size().reset_index(name='sales_count')

            # Ensure 'date' in hourly_sales_counts is datetime64[ns]
            hourly_sales_counts['date'] = pd.to_datetime(hourly_sales_counts['date'])

            # Merge date_hour_df with hourly_sales_counts
            date_hour_df = date_hour_df.merge(hourly_sales_counts, on=['date', 'hour'], how='left')
            date_hour_df['sales_count'] = date_hour_df['sales_count'].fillna(0)

            # Compute mean and std per hour, ensuring all hours are included
            stats_by_hour = date_hour_df.groupby('hour')['sales_count'].agg(['mean', 'std']).reindex(range(24), fill_value=0)

            # Circularly extend the data for moving average
            mean_extended = np.concatenate((
                stats_by_hour['mean'][-(window_size // 2):],  # Last half-window
                stats_by_hour['mean'],
                stats_by_hour['mean'][:(window_size // 2)]  # First half-window
            ))
            std_extended = np.concatenate((
                stats_by_hour['std'][-(window_size // 2):],
                stats_by_hour['std'],
                stats_by_hour['std'][:(window_size // 2)]
            ))

            # Compute moving average on extended data
            mean_moving_avg = pd.Series(mean_extended).rolling(window=window_size, center=True).mean()
            std_moving_avg = pd.Series(std_extended).rolling(window=window_size, center=True).mean()

            # Extract the central part corresponding to original data
            start_idx = window_size // 2
            end_idx = start_idx + len(stats_by_hour)
            mean_moving_avg = mean_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
            std_moving_avg = std_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)

            # Get the product name for the legend
            product_name = self.get_product_name(kod_mahsul)

            # Plot the moving average of the mean
            line, = plt.plot(stats_by_hour.index, mean_moving_avg, label=f'{product_name}, {kod_mahsul}')

            # Get the color of the line
            color = line.get_color()

            # Compute upper and lower bounds
            upper_bound = mean_moving_avg + std_moving_avg / 4
            lower_bound = mean_moving_avg - std_moving_avg / 4

            # Fill between the upper and lower bounds with a light color
            plt.fill_between(stats_by_hour.index, lower_bound, upper_bound, color=color, alpha=0.2)

        # Set titles and labels with Farsi text
        title_text = 'Average Sales per Hour per Day for Top Products with Std.' if not show_least_selling else 'Average Sales per Hour per Day for Worst Products'
        plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel('Hour of Day', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel('Average Sales per Day', fontdict={'fontname': 'XB Zar', 'fontsize': 14})

        plt.xticks(range(24))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Product Name and ID')
        plt.tight_layout()
        plt.show()

    def plot_sales_per_day_of_week(self, filtered_top_kod_tanavo, show_least_selling=False):
        plt.figure(figsize=(12, 8))

        for kod_tanavo in filtered_top_kod_tanavo:
            # Get the corresponding 'کد محصول'
            kod_mahsul = self.get_kod_mahsul_for_kod_tanavo(kod_tanavo)

            # Get all 'کد تنوع' for this 'کد محصول'
            all_kod_tanavo = self.get_all_kod_tanavo_for_kod_mahsul(kod_mahsul)

            # Filter the DataFrame for all 'کد تنوع' related to this 'کد محصول'
            df_kod = self.filtered_df[self.filtered_df['کد تنوع'].isin(all_kod_tanavo)].copy()

            # Extract date and day of the week from 'زمان_میلادی'
            df_kod['date'] = df_kod['زمان_میلادی'].dt.date
            df_kod['day_of_week'] = df_kod['زمان_میلادی'].dt.dayofweek  # 0=Monday, ..., 6=Sunday

            # Get the full date range
            min_date = df_kod['date'].min()
            max_date = df_kod['date'].max()
            full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

            # Create a DataFrame with 'date' and 'day_of_week'
            date_df = pd.DataFrame({'date': full_date_range})
            date_df['day_of_week'] = date_df['date'].dt.dayofweek

            # Ensure 'date' columns are of datetime type in both DataFrames
            date_df['date'] = pd.to_datetime(date_df['date'])
            df_kod['date'] = pd.to_datetime(df_kod['date'])

            # Group df_kod by date and count sales
            daily_sales_counts = df_kod.groupby('date').size().reset_index(name='sales_count')

            # Ensure 'date' column in daily_sales_counts is of datetime type
            daily_sales_counts['date'] = pd.to_datetime(daily_sales_counts['date'])

            # Merge date_df with daily_sales_counts
            date_df = date_df.merge(daily_sales_counts, on='date', how='left')
            date_df['sales_count'] = date_df['sales_count'].fillna(0)

            # Now, group by 'day_of_week' and compute mean and std
            stats_by_day = date_df.groupby('day_of_week')['sales_count'].agg(['mean', 'std'])

            # Get the product name for the legend
            product_name = self.get_product_name(kod_mahsul)

            # Plot the mean line and get the line object
            line, = plt.plot(stats_by_day.index, stats_by_day['mean'], label=f'{product_name}, {kod_mahsul}')

            # Get the color of the line
            color = line.get_color()

            # Compute upper and lower bounds
            upper_bound = stats_by_day['mean'] + stats_by_day['std'] / 4
            lower_bound = stats_by_day['mean'] - stats_by_day['std'] / 4

            # Fill between the upper and lower bounds with a light color
            plt.fill_between(stats_by_day.index, lower_bound, upper_bound, color=color, alpha=0.2)

        # Set titles and labels with Farsi text
        title_text = 'Average Sales per Day of Week for Top Products with Std.' if not show_least_selling else 'Average Sales per Day of Week for Worst Products'
        plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel('Day of Week', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel('Average Sales per Day', fontdict={'fontname': 'XB Zar', 'fontsize': 14})

        # Replace x-ticks with day names in Persian
        days_in_persian = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه', 'شنبه', 'یکشنبه']
        days_in_persian_reshaped = [get_display(arabic_reshaper.reshape(day)) for day in days_in_persian]
        plt.xticks(range(7), days_in_persian_reshaped)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Product Name and ID')
        plt.tight_layout()
        plt.show()

    def plot_overall_sales_per_hour(self, window_size=3):
        plt.figure(figsize=(12, 8))

        df = self.filtered_df.copy()

        # Extract date and hour from 'زمان_میلادی'
        df['date'] = df['زمان_میلادی'].dt.normalize()
        df['hour'] = df['زمان_میلادی'].dt.hour

        # Get the full date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create a DataFrame with all combinations of date and hour
        all_hours = pd.DataFrame({'hour': range(24)})
        date_hour_df = pd.MultiIndex.from_product(
            [full_date_range, all_hours['hour']], names=['date', 'hour']
        ).to_frame(index=False)

        # Ensure 'date' columns are of datetime64[ns] type
        df['date'] = pd.to_datetime(df['date'])
        date_hour_df['date'] = pd.to_datetime(date_hour_df['date'])

        # Group df by date and hour and count sales
        hourly_sales_counts = df.groupby(['date', 'hour']).size().reset_index(name='sales_count')

        # Merge date_hour_df with hourly_sales_counts
        date_hour_df = date_hour_df.merge(hourly_sales_counts, on=['date', 'hour'], how='left')
        date_hour_df['sales_count'] = date_hour_df['sales_count'].fillna(0)

        # Group by 'hour' and compute mean and std
        stats_by_hour = date_hour_df.groupby('hour')['sales_count'].agg(['mean', 'std']).reindex(range(24), fill_value=0)

        # Circularly extend the data for moving average
        mean_extended = np.concatenate((
            stats_by_hour['mean'][-(window_size//2):],
            stats_by_hour['mean'],
            stats_by_hour['mean'][:(window_size//2)]
        ))
        std_extended = np.concatenate((
            stats_by_hour['std'][-(window_size//2):],
            stats_by_hour['std'],
            stats_by_hour['std'][:(window_size//2)]
        ))

        # Compute moving average on extended data
        mean_moving_avg = pd.Series(mean_extended).rolling(window=window_size, center=True).mean()
        std_moving_avg = pd.Series(std_extended).rolling(window=window_size, center=True).mean()

        # Extract the central part corresponding to original data
        start_idx = window_size//2
        end_idx = start_idx + len(stats_by_hour)
        mean_moving_avg = mean_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
        std_moving_avg = std_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)

        # Plot the moving average of the mean
        line, = plt.plot(stats_by_hour.index, mean_moving_avg, label='Overall Sales')

        # Get the color of the line
        color = line.get_color()

        # Compute upper and lower bounds
        upper_bound = mean_moving_avg + std_moving_avg
        lower_bound = mean_moving_avg - std_moving_avg

        # Fill between the upper and lower bounds with a light color
        plt.fill_between(stats_by_hour.index, lower_bound, upper_bound, color=color, alpha=0.2)

        # Set titles and labels with Farsi text
        plt.title(get_display(arabic_reshaper.reshape('میانگین فروش ساعتی کل محصولات')), fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel(get_display(arabic_reshaper.reshape('ساعت روز')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel(get_display(arabic_reshaper.reshape('میانگین فروش در ساعت')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})

        plt.xticks(range(24))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_overall_sales_per_day_of_week(self, window_size=3):
        plt.figure(figsize=(12, 8))

        df = self.filtered_df.copy()

        # Extract date and day of the week from 'زمان_میلادی'
        df['date'] = df['زمان_میلادی'].dt.normalize()
        df['day_of_week'] = df['زمان_میلادی'].dt.dayofweek  # 0=Monday, ..., 6=Sunday

        # Get the full date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create a DataFrame with 'date' and 'day_of_week'
        date_df = pd.DataFrame({'date': full_date_range})
        date_df['day_of_week'] = date_df['date'].dt.dayofweek

        # Ensure 'date' columns are of datetime type in both DataFrames
        date_df['date'] = pd.to_datetime(date_df['date'])
        df['date'] = pd.to_datetime(df['date'])

        # Group df by date and count sales
        daily_sales_counts = df.groupby('date').size().reset_index(name='sales_count')

        # Merge date_df with daily_sales_counts
        date_df = date_df.merge(daily_sales_counts, on='date', how='left')
        date_df['sales_count'] = date_df['sales_count'].fillna(0)

        # Group by 'day_of_week' and compute mean and std
        stats_by_day = date_df.groupby('day_of_week')['sales_count'].agg(['mean', 'std']).reindex(range(7), fill_value=0)

        # Circularly extend the data for moving average
        mean_extended = np.concatenate((
            stats_by_day['mean'][-(window_size//2):],
            stats_by_day['mean'],
            stats_by_day['mean'][:(window_size//2)]
        ))
        std_extended = np.concatenate((
            stats_by_day['std'][-(window_size//2):],
            stats_by_day['std'],
            stats_by_day['std'][:(window_size//2)]
        ))

        # Compute moving average on extended data
        mean_moving_avg = pd.Series(mean_extended).rolling(window=window_size, center=True).mean()
        std_moving_avg = pd.Series(std_extended).rolling(window=window_size, center=True).mean()

        # Extract the central part corresponding to original data
        start_idx = window_size//2
        end_idx = start_idx + len(stats_by_day)
        mean_moving_avg = mean_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
        std_moving_avg = std_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)

        # Plot the moving average of the mean
        line, = plt.plot(stats_by_day.index, mean_moving_avg, label='Overall Sales')

        # Get the color of the line
        color = line.get_color()

        # Compute upper and lower bounds
        upper_bound = mean_moving_avg + std_moving_avg
        lower_bound = mean_moving_avg - std_moving_avg

        # Fill between the upper and lower bounds with a light color
        plt.fill_between(stats_by_day.index, lower_bound, upper_bound, color=color, alpha=0.2)

        # Set titles and labels with Farsi text
        plt.title(get_display(arabic_reshaper.reshape('میانگین فروش روزانه کل محصولات')), fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel(get_display(arabic_reshaper.reshape('روز هفته')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel(get_display(arabic_reshaper.reshape('میانگین فروش در روز')), fontdict={'fontname': 'XB Zar', 'fontsize': 14})

        # Replace x-ticks with day names in Persian and reshape for RTL
        days_in_persian = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه', 'شنبه', 'یکشنبه']
        days_in_persian_reshaped = [get_display(arabic_reshaper.reshape(day)) for day in days_in_persian]
        plt.xticks(range(7), days_in_persian_reshaped)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_sales_distribution(self, number_of_products_to_plot=6, show_least_selling=False):
        top_kod_tanavo = self.get_top_kod_tanavo(number_of_products_to_plot, show_least_selling)
        filtered_top_kod_tanavo = self.filter_top_kod_tanavo(top_kod_tanavo, number_of_products_to_plot)
        self.plot_moving_average_all(filtered_top_kod_tanavo, show_least_selling=show_least_selling)
        if not show_least_selling:
            self.plot_overall_sales_per_hour()
            self.plot_overall_sales_per_day_of_week()
            self.plot_sales_per_hour(filtered_top_kod_tanavo, show_least_selling=show_least_selling)
            self.plot_sales_per_day_of_week(filtered_top_kod_tanavo, show_least_selling=show_least_selling)

