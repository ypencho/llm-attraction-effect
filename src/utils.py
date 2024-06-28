import pandas as pd
import json

def wrap_string(s, n):
    words = s.split()
    lines = []
    current_line = words[0]

    for word in words[1:]:
        if len(current_line) + len(word) + 1 <= n:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return '\n'.join(lines)

def read_results(results):
    if isinstance(results, str):  # file path
        with open(results, 'r') as f:
            exp_results = pd.DataFrame(json.load(f))
    elif isinstance(results, list):  # list of dicts
        exp_results = pd.DataFrame(results)
    elif isinstance(results, pd.DataFrame):  # pandas dataframe
        exp_results = results
    else:
        raise ValueError()
    
    return exp_results

def get_axis_label(job_opening_details, axis='dim_1'):
    dim_data = job_opening_details[axis]
    return f"{dim_data['label']} {dim_data['type']} [{dim_data['unit']}]"

dim_values = lambda dim_range: list(range(dim_range[0], dim_range[1]+1))


def save_latex_table(df, tag, caption, column_format, table_folder="../results/tables/", index=False):
    latex_table_custom = df.to_latex(
        index=index, 
        caption=caption,
        label=f'tab:{tag}', 
        column_format=column_format
        )

    with open(table_folder + f'{tag}_table.tex', 'w') as f:
        f.write(latex_table_custom)
    
    print(latex_table_custom)
    
def save_verbatim_text(text, tag, text_folder="../results/tables/"):
    verbatim_tex = "\\begin{verbatim}\n" + text + "\n\\end{verbatim}\n"

    with open(text_folder + f'text_{tag}.tex', 'w') as file:
        file.write(verbatim_tex)

    print(verbatim_tex)