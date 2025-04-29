from email.utils import make_msgid

import pandas as pd
from io import BytesIO


table_style = """
<style>
    .summary-table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .summary-table th {
        background-color: #f2f2f2;
        color: #333;
        text-align: center;
        padding: 8px;
        border: 1px solid #ddd;
    }
    .summary-table td {
        padding: 8px;
        border: 1px solid #ddd;
        text-align: center;
    }
    .summary-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
</style>
"""

def buildMonthlyTable(res_generation, res_generation_day):
    monthly_mean = res_generation[res_generation.index.month == pd.Timestamp.now().month].mean()
    monthly_q1 = res_generation[res_generation.index.month == pd.Timestamp.now().month].quantile(0.25)
    monthly_q3 = res_generation[res_generation.index.month == pd.Timestamp.now().month].quantile(0.75)

    monthly_mean_last_year = res_generation[
        (res_generation.index.month == pd.Timestamp.now().month) &
        (res_generation.index.year == (pd.Timestamp.now().year - 1))
        ].mean()
    monthly_mean_current_year = res_generation_day[
        (res_generation_day.index.month == pd.Timestamp.now().month) &
        (res_generation_day.index.year == (pd.Timestamp.now().year))
        ].mean()
    monthly_q1_current_year = res_generation_day[
        (res_generation_day.index.month == pd.Timestamp.now().month) &
        (res_generation_day.index.year == (pd.Timestamp.now().year))
        ].quantile(0.25)
    monthly_q3_current_year = res_generation_day[
        (res_generation_day.index.month == pd.Timestamp.now().month) &
        (res_generation_day.index.year == (pd.Timestamp.now().year))
        ].quantile(0.75)

    monthly_stats = pd.DataFrame({
        'Techno': monthly_q1.index,
        'q1 (25%) history': monthly_q1,
        'mean history': monthly_mean,
        'q3 (75%) history': monthly_q3,
        'mean_last_year': monthly_mean_last_year,
        'mean_current_year': monthly_mean_current_year,
        'q1_current_year (25%)': monthly_q1_current_year,
        'q3_current_year (75%)': monthly_q3_current_year})

    return monthly_stats

def setImgEmail(fig):
    img_data = BytesIO()
    fig.savefig(img_data, format='png')
    img_data.seek(0)
    image_cid = make_msgid(domain='xyz.com')[1:-1]
    return image_cid, img_data

def style_html_table(html):
    return html.replace(
        "<table border=\"1\" class=\"dataframe\">",
        """<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px;">"""
    ).replace(
        "<th>",
        """<th style="background-color: #f2f2f2; color: #333; text-align: center; padding: 8px; border: 1px solid #ddd;">"""
    ).replace(
        "<td>",
        """<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">"""
    )