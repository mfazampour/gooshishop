import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager
import numpy as np

def read_data(file_path, sheet_name=None):
    """Reads the Excel file and returns the specified sheet or all sheets as a dictionary."""
    try:
        if sheet_name:
            sheets_dict = pd.read_excel(file_path, sheet_name=sheet_name)
            sheets_dict = {sheet_name: sheets_dict}  # Wrap in a dictionary for consistency
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

def process_price_data(price_data_df):
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

    channel_counts = filtered_df['کانال فروش'].value_counts()

    # Reshape and fix the bidi order of the title
    reshaped_text = arabic_reshaper.reshape('توزیع کانال فروش برای گوشی موبایل')
    bidi_text = get_display(reshaped_text)

    plt.figure(figsize=(8, 8))
    plt.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', startangle=140)

    # Set the title with the correct Farsi text
    plt.title(bidi_text, fontdict={'fontname': 'XB Zar', 'fontsize': 16})

    plt.show()

    # Filter data for the Digikala channel
    digikala_df = filtered_df[filtered_df['کانال فروش'] == 'digikala']

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

        # Filter data for the 'گوشی شاپ' channel
        goshi_shop_df = filtered_df[filtered_df['کانال فروش'] == 'گوشی شاپ']

        # Ensure the necessary column exists
        if 'رتبه ترب' in goshi_shop_df.columns:
            # Count the occurrences of each unique value in the 'رتبه ترب' column
            roteba_torob_counts = goshi_shop_df['رتبه ترب'].value_counts()

            # Create a pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(roteba_torob_counts, labels=roteba_torob_counts.index, autopct='%1.1f%%', startangle=140)

            # Set the title with the correct Farsi text
            plt.title(get_display(arabic_reshaper.reshape('توزیع رتبه ترب برای گوشی شاپ')),
                      fontdict={'fontname': 'XB Zar', 'fontsize': 16})

            plt.show()
        else:
            print("Error: The column 'رتبه ترب' is missing from the گوشی شاپ channel data.")

    # Filter data for the 'گوشی شاپ' channel
    goshi_shop_df = filtered_df[filtered_df['کانال فروش'] == 'gooshishop']

    # Ensure the necessary column exists
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

    # Ensure the necessary columns exist
    # Ensure the necessary columns exist
    if 'قیمت فروش' in goshi_shop_df.columns and 'قیمت اول ترب' in goshi_shop_df.columns:
        # Calculate the percentage difference
        goshi_shop_df['perc_diff_قیمت_فروش_قیمت_اول_ترب'] = 100 * (
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

    # Step 1: Preprocess the `roteba_torob_counts` values
    goshi_shop_df_clean['roteba_torob'] = goshi_shop_df_clean['رتبه ترب'].apply(pd.to_numeric, errors='coerce')

    # Replace +adv with -1
    goshi_shop_df_clean['roteba_torob'] = goshi_shop_df_clean['roteba_torob'].fillna(-1)

    # Cap values equal to or greater than 10 at 10
    goshi_shop_df_clean['roteba_torob'] = goshi_shop_df_clean['roteba_torob'].apply(lambda x: 10 if x >= 10 else x)

    # Remove any rows where `roteba_torob` is NaN
    goshi_shop_df_clean = goshi_shop_df_clean.dropna(subset=['roteba_torob'])

    # Cap the percentage differences to the range -20% to 20%
    goshi_shop_df_clean['perc_diff_قیمت_فروش_قیمت_اول_ترب'] = goshi_shop_df_clean[
        'perc_diff_قیمت_فروش_قیمت_اول_ترب'].apply(lambda x: max(min(x, 20), -20))

    # Step 2: Create the 2D histogram
    plt.figure(figsize=(10, 8))
    hist, xedges, yedges, _ = plt.hist2d(goshi_shop_df_clean['roteba_torob'],
                                         goshi_shop_df_clean['perc_diff_قیمت_فروش_قیمت_اول_ترب'], bins=[10, 50],
                                         cmap='plasma')

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

    save_filtered_data(filtered_df, 'filtered_price_data.xlsx')

def save_filtered_data(filtered_df, file_name):
    try:
        filtered_df.to_excel(file_name, index=False)
        print(f"Filtered data saved to {file_name}")
    except Exception as e:
        print(f"Error saving filtered data: {e}")

if __name__ == '__main__':
    file_path = 'ثبت روزانه سبدهای پیشنهادی تامین کنندگان-3.xlsx'
    sheets_dict = read_data(file_path, 'PriceData')

    if sheets_dict:
        if 'PriceData' in sheets_dict:
            price_data_df = sheets_dict['PriceData']
            process_price_data(price_data_df)
        else:
            print("Error: 'PriceData' sheet not found in the Excel file.")
