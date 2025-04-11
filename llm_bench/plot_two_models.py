import click
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


@click.command()
@click.argument("csv_file", type=click.Path(exists=True))
@click.option("--output-file", default="benchmark_results.html", help="Output HTML file path")
@click.option("--title", default="Benchmark Results", help="Title for the benchmark report")
def plot_benchmark_results(csv_file, output_file, title):
    """Create simple interactive plots from benchmark results CSV file."""
    df = pd.read_csv(csv_file)

    output_path = Path(output_file).parent
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    html_output = [
        """
    <html>
    <head>
        <title>Benchmark Results</title>
        <style>
            .plot-container { width: 100%; margin: 20px 0; }
            h1 { text-align: center; margin: 30px 0; }
            h2 { margin: 20px 0 10px 0; }
        </style>
    </head>
    <body>
    """
    ]

    html_output.append(f"<h1>{title}</h1>")

    create_performance_plot(df, html_output)

    create_throughput_plot(df, html_output)

    html_output.append("</body></html>")

    with open(output_file, "w") as f:
        f.write("\n".join(html_output))

    print(f"Interactive visualization saved to {output_file}")


def create_performance_plot(df, html_output):
    """Create a simple plot comparing key performance metrics."""
    html_output.append("<h2>Performance Metrics</h2>")
    html_output.append('<div class="plot-container">')

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Time To First Token",
            "Total Latency",
            "Latency Per Token",
            "Completion Rate",
        ),
    )

    colors = {"Model 1": "blue", "Model 2": "red"}

    for model_name in (
        df["ModelName"].unique()
        if "ModelName" in df.columns
        else df["Test"].apply(lambda x: "Model 1" if "model1" in x else "Model 2").unique()
    ):
        if "ModelName" in df.columns:
            model_df = df[df["ModelName"] == model_name]
        else:
            model_df = df[
                df["Test"].apply(lambda x: "Model 1" if "model1" in x else "Model 2") == model_name
            ]

        if "TargetQPS" in df.columns:
            qps_values = model_df["TargetQPS"]
        else:
            qps_values = model_df["Test"].apply(
                lambda x: int(x.split("_")[-1]) if "qps" in x else model_df["Qps"].iloc[0]
            )

        completion_rate = 100 - (model_df["Incomplete Requests"] / model_df["Total Requests"] * 100)

        fig.add_trace(
            go.Scatter(
                x=qps_values,
                y=model_df["Time To First Token"],
                mode="lines+markers",
                name=f"{model_name} - TTFT",
                line=dict(color=colors.get(model_name, "gray")),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=qps_values,
                y=model_df["Total Latency"],
                mode="lines+markers",
                name=f"{model_name} - Total Latency",
                line=dict(color=colors.get(model_name, "gray")),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=qps_values,
                y=model_df["Latency Per Token"],
                mode="lines+markers",
                name=f"{model_name} - Latency Per Token",
                line=dict(color=colors.get(model_name, "gray")),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=qps_values,
                y=completion_rate,
                mode="lines+markers",
                name=f"{model_name} - Completion Rate",
                line=dict(color=colors.get(model_name, "gray")),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=600,
        width=900,
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(l=50, r=50, t=60, b=50),
    )

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="QPS", row=i, col=j)

    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Completion %", row=2, col=2)

    html_output.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_output.append("</div>")


def create_throughput_plot(df, html_output):
    """Create a plot showing achieved QPS vs target QPS."""
    html_output.append("<h2>Achieved QPS vs Target QPS</h2>")
    html_output.append('<div class="plot-container">')

    fig = go.Figure()

    colors = {"Model 1": "blue", "Model 2": "red"}

    max_qps = df["Qps"].max() * 1.2  # Add some margin

    fig.add_trace(
        go.Scatter(
            x=[0, max_qps],
            y=[0, max_qps],
            mode="lines",
            name="Ideal (1:1)",
            line=dict(color="black", width=1, dash="dash"),
            opacity=0.5,
        )
    )

    for model_name in (
        df["ModelName"].unique()
        if "ModelName" in df.columns
        else df["Test"].apply(lambda x: "Model 1" if "model1" in x else "Model 2").unique()
    ):
        if "ModelName" in df.columns:
            model_df = df[df["ModelName"] == model_name]
        else:
            model_df = df[
                df["Test"].apply(lambda x: "Model 1" if "model1" in x else "Model 2") == model_name
            ]

        if "TargetQPS" in df.columns:
            target_qps = model_df["TargetQPS"]
        else:
            target_qps = model_df["Test"].apply(
                lambda x: int(x.split("_")[-1]) if "qps" in x else model_df["Qps"].iloc[0]
            )

        fig.add_trace(
            go.Scatter(
                x=target_qps,
                y=model_df["Qps"],
                mode="lines+markers",
                name=f"{model_name}",
                line=dict(color=colors.get(model_name, "gray")),
                marker=dict(size=8),
            )
        )

    fig.update_layout(height=400, width=700, xaxis_title="Target QPS", yaxis_title="Achieved QPS")

    html_output.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_output.append("</div>")


if __name__ == "__main__":
    plot_benchmark_results()
