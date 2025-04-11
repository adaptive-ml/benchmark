import click
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


@click.command()
@click.option("--model", required=True, type=str, help="Name of the benchmarked model")
@click.option("--output-tokens", required=True, type=int, help="Number of output tokens")
@click.option(
    "--input-files",
    required=True,
    type=click.Path(exists=True),
    multiple=True,
    help="Provide 2 results files to plot",
)
@click.option(
    "--provider-suffixes",
    type=str,
    multiple=True,
    help="Suffixes to attach to provider names (must match input file count)",
)
@click.option(
    "--output-file",
    default="results.html",
    type=str,
    show_default=True,
    help="Output HTML file",
)
@click.option(
    "--extra-header",
    type=str,
    help="Add an h1 header to top of page",
)
def main(model, output_tokens, input_files, provider_suffixes, output_file, extra_header):
    if provider_suffixes:
        assert len(input_files) == len(provider_suffixes), \
            "If passing suffixes, you must pass one per input file"
    assert len(input_files) <= 5, "Can only compare 5 result files at a time"

    dfs = []
    for idx, f in enumerate(input_files):
        this_df = pd.read_csv(f)
        if provider_suffixes:
            suffix = provider_suffixes[idx]
            this_df["Provider"] = this_df["Provider"].astype(str) + f"-{suffix}"
        dfs.append(this_df)

    df = pd.concat(dfs, axis=0)

    html_output = ["""
    <html>
    <head>
        <title>Benchmark Results Analysis</title>
        <style>
            .plot-container {
                width: 100%;
                margin: 20px 0;
            }
            h1 {
                text-align: center;
                margin: 40px 0 20px 0;
            }
            h2 {
                text-align: left;
                margin: 40px 0 20px 0;
            }
        </style>
    </head>
    <body>
    """]

    gpu_name = os.environ.get("GPU_NAME", "")
    html_output.append(
        f"<h1>Model:{model}   |   Output tokens:{output_tokens}   |   Time per test (s):60   |   GPU:1 x {gpu_name} </h1>"
    )
    if extra_header:
        html_output.append(f"<h1>{extra_header}</h1>")

    prompt_tokens = sorted(df["Prompt Tokens"].unique())

    line_and_fill_colors = [
        ("blue", "135, 206, 250, 0.4"),
        ("red", "255, 182, 193, 0.4"),
        ("orange", "255, 200, 124, 0.1"),
        ("pink", "255, 192, 203, 0.4"),
        ("green", "144, 238, 144, 0.4"),
    ]

    for token_value in prompt_tokens:
        token_df = df[df["Prompt Tokens"] == token_value]

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "% incomplete requests/total requests vs. Concurrency",
                "P90 Time to First Token vs. Concurrency",
                "P90 Time per Output Token vs. Concurrency",
                "P90 Total Latency vs. Concurrency",
            ),
            vertical_spacing=0.1,
        )

        for idx, provider in enumerate(df["Provider"].unique()):
            provider_df = token_df[token_df["Provider"] == provider]

            ratio = round((provider_df["Incomplete Requests"] / provider_df["Total Requests"]) * 100, 2)
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=ratio,
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Time To First Token"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Latency Per Token"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Total Latency"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f"rgba({line_and_fill_colors[idx][1]})",
                    line=dict(color=line_and_fill_colors[idx][0]),
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

        fig.update_layout(
            height=1000,
            width=1000,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        for i in range(1, 5):
            fig.update_xaxes(title_text="Concurrency (QPS)", row=i, col=1)
        fig.update_yaxes(title_text="% incomplete requests/total requests", row=1, col=1)
        fig.update_yaxes(title_text="TTFT (ms)", row=2, col=1)
        fig.update_yaxes(title_text="TPOT(ms)", row=3, col=1)
        fig.update_yaxes(title_text="Total latency(ms)", row=4, col=1)

        html_output.append(f"<h2>Input Tokens: {int(token_value)}</h2>")
        html_output.append('<div class="plot-container">')
        html_output.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        html_output.append("</div>")

    html_output.append("</body></html>")

    print("Writing HTML")
    with open(output_file, "w") as f:
        f.write("\n".join(html_output))


if __name__ == "__main__":
    main()
