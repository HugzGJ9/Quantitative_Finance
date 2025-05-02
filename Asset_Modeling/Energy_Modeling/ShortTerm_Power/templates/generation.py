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
