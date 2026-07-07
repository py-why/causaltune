from experiment_plots import metric_names


def df_to_latex(df, filename):
    latex = df.to_latex(escape=False, index=False)
    latex = prettify_latex(latex)
    with open(filename, "w") as file:
        file.write(latex)


def prettify_latex(latex: str):
    latex = "\\renewcommand{\\arraystretch}{1.1}\n" + latex.replace("\\toprule", "\\hline").replace(
        "\\midrule", "\\hline"
    ).replace("\\bottomrule", "\\hline")
    return latex


def rename(x):
    if x in metric_names:
        return metric_names[x].replace("\n", " ")
    return x


def create_latex_table(
    data,
    dataset_col,
    metric_col,
    corr_col,
    row_order,
    column_order=None,
    filename=None,
    flag_max=True,
    first_col_precision: bool = False,
):
    """
    Create a LaTeX table from a long-format dataset with customizable column names,
    row and column order, and formatting options.

    Parameters:
        data (pd.DataFrame): A DataFrame containing the data.
        dataset_col (str): Column name for datasets (columns in the output table).
        metric_col (str): Column name for metrics (rows in the output table).
        corr_col (str): Column name for correlation values (cells in the table).
        row_order (list): Ordered list of metric values to arrange the rows in the table.
        column_order (list, optional): Ordered list of dataset values to arrange the columns in the table.
        filename (str, optional): File path to save the LaTeX table. If None, the table is not saved.

    Returns:
        str: A LaTeX table as a string.
    """
    if column_order is None:

        cols = []
        for kind in [
            "RCT",
            "KCKP",
            "KC",
        ]:
            cols += [f"Linear {kind}", f"NonLinear {kind}"]
        column_order = cols

    # Pivot the data into a wide format
    table = data.pivot(index=metric_col, columns=dataset_col, values=corr_col)

    table = table.loc[row_order]
    if column_order:
        table = table[column_order]

    # Reorder rows and columns based on the provided order
    table.columns = [
        c.replace("NonLinear", "Nonlin.")
        .replace("KCKP", "KPTT")
        .replace("KC", "FPTT")
        .replace("Linear", "Lin.")
        for c in table.columns
    ]

    # Function to bold the maximum value in each column
    def bold_max(series):
        if flag_max:
            max_value = series.max()
        else:
            max_value = series.min()
        if series.name == "Linear RCT" and first_col_precision:
            return [f"\\textbf{{{v:.3f}}}" if v == max_value else f"{v:.3f}" for v in series]
        else:
            return [f"\\textbf{{{v:.2f}}}" if v == max_value else f"{v:.2f}" for v in series]

    # Apply bold_max to each column
    formatted_table = table.apply(bold_max, axis=0)

    # Generate the LaTeX table with a vertical line after row names
    latex_table = formatted_table.to_latex(
        escape=False,  # Allow LaTeX formatting
        column_format="|l|" + "r" * len(table.columns) + "|",  # Vertical line after row names
        header=True,  # Include column headers (dataset names)
    )

    # Remove the dataset column name from the LaTeX table
    lines = latex_table.splitlines()
    lines[0] = lines[0].replace(f"{dataset_col} ", "")  # Remove the dataset column header
    lines[1] = lines[1].replace(f"{'-' * len(dataset_col)} ", "")  # Adjust the alignment line
    filtered_lines = [line for line in lines if not line.startswith(metric_col)]
    latex_table = (
        "\n".join(filtered_lines)
        .replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline")
    )

    latex_table = "\\renewcommand{\\arraystretch}{1.1}\n" + latex_table

    # Save the table to a file if a filename is provided
    if filename:
        with open(filename, "w") as file:
            file.write(latex_table)

    return latex_table
