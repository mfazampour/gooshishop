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
    def __init__(self, filtered_df, bot_workspace_df, sale_channel: str = None):
        self.filtered_df = filtered_df.copy()
        self.bot_workspace_df = bot_workspace_df.copy()
        self.window_size = 5  # Default window size for moving averages
        self.window_size_weekly = 3  # Default window size for moving averages on week days
        self.plot_title_suffix = f" فروش در {sale_channel}" if sale_channel is not None else ''
        self.plot_title_suffix = get_display(arabic_reshaper.reshape(self.plot_title_suffix))

        self._prepare_data()

    def _prepare_data(self):
        # Convert 'زمان' from Jalali to Gregorian and then to datetime
        self.filtered_df['زمان_میلادی'] = self.filtered_df['زمان'].apply(
            lambda x: jdatetime.datetime.strptime(x, '%Y.%m.%d %H:%M').togregorian()
        )
        # Extract date-related features
        self.filtered_df['date'] = self.filtered_df['زمان_میلادی'].dt.normalize()
        self.filtered_df['hour'] = self.filtered_df['زمان_میلادی'].dt.hour
        self.filtered_df['day_of_week'] = self.filtered_df['زمان_میلادی'].dt.dayofweek  # 0=Monday, ..., 6=Sunday

    def get_product_name(self, kod_mahsul):
        product_name = self.bot_workspace_df[self.bot_workspace_df['کد محصول'] == kod_mahsul]['نام محصول'].iloc[0]
        reshaped_name = arabic_reshaper.reshape(product_name)
        bidi_name = get_display(reshaped_name)
        return bidi_name

    def get_kod_mahsul_for_kod_tanavo(self, kod_tanavo):
        kod_mahsul = self.bot_workspace_df[self.bot_workspace_df['کد تنوع'] == kod_tanavo]
        if kod_mahsul.__len__() > 0:
            kod_mahsul = self.bot_workspace_df[self.bot_workspace_df['کد تنوع'] == kod_tanavo]['کد محصول'].iloc[0]
            return kod_mahsul
        else:  # it is strange but there are some that do not exist here
            return None

    def get_all_kod_tanavo_for_kod_mahsul(self, kod_mahsul):
        return self.bot_workspace_df[self.bot_workspace_df['کد محصول'] == kod_mahsul]['کد تنوع'].dropna().unique()

    def get_filtered_top_kod_tanavo(self, number_of_products_to_plot=6, show_least_selling=False):
        # Get top or least selling 'کد تنوع'
        if show_least_selling:
            kod_tanavo_counts = self.filtered_df['کد تنوع'].value_counts().nsmallest(number_of_products_to_plot * 2)
        else:
            kod_tanavo_counts = self.filtered_df['کد تنوع'].value_counts().nlargest(number_of_products_to_plot * 2)

        top_kod_tanavo = kod_tanavo_counts.index
        prod_id_list = []
        filtered_top_kod_tanavo = []
        for kod_tanavo in top_kod_tanavo:
            kod_mahsul = self.get_kod_mahsul_for_kod_tanavo(kod_tanavo)
            if kod_mahsul is None:
                continue
            if kod_mahsul not in prod_id_list:
                prod_id_list.append(kod_mahsul)
                filtered_top_kod_tanavo.append(kod_tanavo)
                if len(filtered_top_kod_tanavo) == number_of_products_to_plot:
                    break
        return filtered_top_kod_tanavo

    def compute_stats_by_time_unit(self, df, time_unit):
        # Compute sales counts grouped by the specified time unit
        if time_unit == 'hour':
            time_values = range(24)
        elif time_unit == 'day_of_week':
            time_values = range(7)
        else:
            raise ValueError("Invalid time_unit. Use 'hour' or 'day_of_week'.")

        # Group by time unit and compute mean and std
        stats = df.groupby(time_unit)['sales_count'].agg(['mean', 'std']).reindex(time_values, fill_value=0)
        return stats

    def compute_moving_average(self, data_series, window_size=None):
        if not window_size:
            window_size = self.window_size
        # Circularly extend the data for moving average
        mean_extended = np.concatenate((
            data_series[-(window_size // 2):],
            data_series,
            data_series[:(window_size // 2)]
        ))
        # Compute moving average on extended data
        mean_moving_avg = pd.Series(mean_extended).rolling(window=window_size, center=True).mean()
        # Extract the central part corresponding to original data
        start_idx = window_size // 2
        end_idx = start_idx + len(data_series)
        mean_moving_avg = mean_moving_avg.iloc[start_idx:end_idx].reset_index(drop=True)
        return mean_moving_avg

    def plot_moving_average_all(self, filtered_top_kod_tanavo, show_least_selling=False):
        plt.figure(figsize=(12, 8))
        for kod_tanavo in filtered_top_kod_tanavo:
            # Get corresponding 'کد محصول' and all 'کد تنوع'
            kod_mahsul = self.get_kod_mahsul_for_kod_tanavo(kod_tanavo)
            if kod_mahsul is None:
                continue
            all_kod_tanavo = self.get_all_kod_tanavo_for_kod_mahsul(kod_mahsul)

            # Filter DataFrame for the product
            df_kod = self.filtered_df[self.filtered_df['کد تنوع'].isin(all_kod_tanavo)]
            # Resample by day and calculate the count
            daily_counts = df_kod.resample('D', on='زمان_میلادی').size()
            # Create a full date range
            full_date_range = pd.date_range(
                start=self.filtered_df['زمان_میلادی'].min().date(),
                end=self.filtered_df['زمان_میلادی'].max().date(),
                freq='D'
            )
            # Reindex and fill missing values with zero
            daily_counts = daily_counts.reindex(full_date_range, fill_value=0)
            # Calculate moving average
            moving_avg = daily_counts.rolling(window=self.window_size).mean()
            product_name = self.get_product_name(kod_mahsul)
            # Plot
            plt.plot(moving_avg, label=f'{product_name}, {kod_mahsul}')

        # Set titles and labels
        title_text = 'Moving Average of Sales for Top Products' if not show_least_selling else 'Moving Average of Sales for Worst Products'
        plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel('Time', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel('Moving Average of Sales', fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Product Name and ID')
        plt.tight_layout()
        plt.show()

    def plot_sales_per_time_unit(self, filtered_top_kod_tanavo, time_unit, show_least_selling=False):
        plt.figure(figsize=(12, 8))
        for kod_tanavo in filtered_top_kod_tanavo:
            kod_mahsul = self.get_kod_mahsul_for_kod_tanavo(kod_tanavo)
            if kod_mahsul is None:
                continue
            all_kod_tanavo = self.get_all_kod_tanavo_for_kod_mahsul(kod_mahsul)

            # Filter DataFrame for the product
            df_kod = self.filtered_df[self.filtered_df['کد تنوع'].isin(all_kod_tanavo)].copy()

            # Define time values based on the time unit
            if time_unit == 'hour':
                time_values = range(24)
                xlabel = get_display(arabic_reshaper.reshape('ساعت روز'))
                title_text = 'Average Sales per Hour per Day for Top Products with Std.' if not show_least_selling else 'Average Sales per Hour per Day for Worst Products'
            elif time_unit == 'day_of_week':
                time_values = range(7)
                xlabel = get_display(arabic_reshaper.reshape('روز هفته'))
                title_text = 'Average Sales per Day of Week for Top Products with Std.' if not show_least_selling else 'Average Sales per Day of Week for Worst Products'
                # title_text = get_display(arabic_reshaper.reshape('میانگین فروش روزانه کل محصولات'))
            else:
                raise ValueError("Invalid time_unit. Use 'hour' or 'day_of_week'.")

            mean_moving_avg, std_moving_avg = self.calculate_stats_to_plot(df_kod, time_unit, time_values)

            # Plot
            product_name = self.get_product_name(kod_mahsul)
            line, = plt.plot(time_values, mean_moving_avg, label=f'{product_name}, {kod_mahsul}')
            color = line.get_color()
            upper_bound = mean_moving_avg + std_moving_avg / 4
            lower_bound = mean_moving_avg - std_moving_avg / 4
            plt.fill_between(time_values, lower_bound, upper_bound, color=color, alpha=0.2)

        legend_title = 'Product Name and ID'

        self.set_title_and_labels(legend_title, time_unit, time_values, title_text, xlabel)

    def set_title_and_labels(self, legend_title, time_unit, time_values, title_text, xlabel):
        title_text += self.plot_title_suffix
        # Set titles and labels
        plt.title(title_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})
        plt.xlabel(xlabel, fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.ylabel(get_display(arabic_reshaper.reshape('میانگین فروش')),
                   fontdict={'fontname': 'XB Zar', 'fontsize': 14})
        plt.grid(True, linestyle='--', alpha=0.6)
        if time_unit == 'day_of_week':
            days_in_persian = ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه', 'شنبه', 'یکشنبه']
            xtick_labels = [get_display(arabic_reshaper.reshape(day)) for day in days_in_persian]
            plt.xticks(time_values, xtick_labels)
        else:
            plt.xticks(time_values)
        plt.legend(title=legend_title)
        plt.tight_layout()
        plt.show()

    def calculate_stats_to_plot(self, df_kod, time_unit, time_values):
        # Create full date range
        min_date = df_kod['date'].min()
        max_date = df_kod['date'].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        # Create a DataFrame with all combinations
        date_time_df = pd.MultiIndex.from_product(
            [full_date_range, time_values], names=['date', time_unit]
        ).to_frame(index=False)
        # Group df_kod by date and time unit and count sales
        df_kod['sales_count'] = 1
        grouped = df_kod.groupby(['date', time_unit])['sales_count'].sum().reset_index()
        # Merge with date_time_df
        if time_unit == 'hour':
            date_time_df = date_time_df.merge(grouped, on=['date', time_unit], how='left').fillna(0)
        else:
            date_time_df = date_time_df.merge(grouped, on=['date', time_unit], how='left').dropna()
        # Compute stats
        stats = self.compute_stats_by_time_unit(date_time_df, time_unit)
        # Compute moving average
        mean_moving_avg = self.compute_moving_average(stats['mean'],
                                                      window_size=self.window_size if time_unit == 'hour' else self.window_size_weekly)
        std_moving_avg = self.compute_moving_average(stats['std'],
                                                     window_size=self.window_size if time_unit == 'hour' else self.window_size_weekly)

        return mean_moving_avg, std_moving_avg

    def plot_overall_sales_per_time_unit(self, time_unit):
        plt.figure(figsize=(12, 8))
        df = self.filtered_df.copy()

        # Define time values and labels based on the time unit
        if time_unit == 'hour':
            time_values = range(24)
            xlabel = get_display(arabic_reshaper.reshape('ساعت روز'))
            title_text = get_display(arabic_reshaper.reshape('میانگین فروش ساعتی کل محصولات'))
        elif time_unit == 'day_of_week':
            time_values = range(7)
            xlabel = get_display(arabic_reshaper.reshape('روز هفته'))
            title_text = get_display(arabic_reshaper.reshape('میانگین فروش روزانه کل محصولات'))
        else:
            raise ValueError("Invalid time_unit. Use 'hour' or 'day_of_week'.")

        mean_moving_avg, std_moving_avg = self.calculate_stats_to_plot(df, time_unit, time_values)

        # Plot
        line, = plt.plot(time_values, mean_moving_avg, label='Overall Sales')
        color = line.get_color()
        upper_bound = mean_moving_avg + std_moving_avg / 4
        lower_bound = mean_moving_avg - std_moving_avg / 4
        plt.fill_between(time_values, lower_bound, upper_bound, color=color, alpha=0.2)

        self.set_title_and_labels('', time_unit, time_values, title_text, xlabel)

    def plot_sales_distribution(self, number_of_products_to_plot=6, show_least_selling=False):
        filtered_top_kod_tanavo = self.get_filtered_top_kod_tanavo(number_of_products_to_plot, show_least_selling)
        self.plot_moving_average_all(filtered_top_kod_tanavo, show_least_selling=show_least_selling)
        if not show_least_selling:
            self.plot_overall_sales_per_time_unit('hour')
            self.plot_overall_sales_per_time_unit('day_of_week')
            self.plot_sales_per_time_unit(filtered_top_kod_tanavo, 'hour', show_least_selling=show_least_selling)
            self.plot_sales_per_time_unit(filtered_top_kod_tanavo, 'day_of_week', show_least_selling=show_least_selling)
