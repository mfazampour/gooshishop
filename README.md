## Overview

This project processes and analyzes sales data from an Excel file that contains multiple sheets with product, sales, and channel data. The main objective is to visualize and analyze sales performance, price comparisons, and the ranking of products across different sales channels such as GooshiShop and Digikala.

The program is designed to:

- Read data from an Excel file, cache it for faster subsequent access.
- Process the sales and product data to generate insightful visualizations.
- Filter product sales data based on rankings and sales channels.
- Analyze the distribution of products and their performance across different sales platforms.
- Visualize the speed of product sales occurrences over time.

## Installation

### Requirements

The project requires the following Python packages:
- `pandas`
- `numpy`
- `matplotlib`
- `jdatetime`
- `arabic_reshaper`
- `python-bidi`
- `xlrd` (for reading Excel files)

You can install the required packages using the following command:

```bash
pip install pandas numpy matplotlib jdatetime arabic_reshaper python-bidi xlrd
```

### Fonts

Ensure you have an appropriate Persian font, such as `XB Zar`, installed on your system to render Farsi text correctly in the visualizations.

## File Structure

- `main.py`: The main script that contains functions for reading, processing, and visualizing the sales data.
- `SalesPlots.py`: Contains the `SalesPlotter` class used to generate different sales plots and visualizations.
- `data_cache.pkl`: A cache file to speed up data loading.

## Usage

1. **Read Data**: 
   The script reads the sales data from an Excel file. It supports reading specific sheets, with the option to cache the data for quicker subsequent loads.

   ```python
   sheets_dict = read_data('your_excel_file.xlsx', ['PriceData', 'BotWorkSpace', 'SellData'])
   ```

   The script will also translate Farsi sheet names to English for easier manipulation.

2. **Process and Visualize Data**:
   Once the data is loaded, it processes the data by filtering and performing various analyses. The results include:

   - Plotting the sales distribution for the top products.
   - Generating pie charts of sales channel distribution.
   - Comparing product rankings from platforms like Torob.
   - Analyzing the speed of sales occurrences over time.

   Example:

   ```python
   process_price_data(price_data_df, bot_workspace_df=sheets_dict['BotWorkSpace'])
   ```

3. **Cache Data**:
   The script caches the loaded data using a pickle file (`data_cache.pkl`). This allows future runs to load the data without needing to read from the Excel file each time, which speeds up the process.

4. **Plots and Visualizations**:
   The script provides multiple visualizations such as:
   
   - **Sales Distribution**: Shows the top-selling products in specific channels.
   - **Price Comparisons**: Compares the prices across different sales channels like Digikala and GooshiShop.
   - **Torob Ranking Analysis**: Analyzes how product rankings change over time on platforms like Torob.

5. **Time-based Occurrence Analysis**:
   The script calculates the speed of product occurrences within certain time frames (3-hour windows) and plots the rate of sales occurrences for highly ranked products.

## Customization

- **File Path**: The file path for the Excel file and cache can be customized in the script by modifying the variables `file_path` and `CACHE_FILE_PATH`.
- **Sheet Translation**: If your Excel sheets use different Farsi names, you can update the `translation_map` in the `change_sheet_name_to_english` function to map Farsi sheet names to English.

## Example

To run the script:

```bash
python main.py
```

This will read the sales data from the specified Excel file, process the relevant sheets, and generate the visualizations.

## Dependencies

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib**: For generating plots and visualizations.
- **jdatetime**: To handle Jalali dates and convert them to Gregorian.
- **arabic_reshaper**: To render Farsi text correctly in visualizations.
- **python-bidi**: To display bidirectional text, especially for Farsi labels.
- **pickle**: To cache the data for faster subsequent runs.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

