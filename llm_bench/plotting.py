import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main(args):
    # Read the CSV data
    df = pd.read_csv(args.input_file)

    # Create the HTML file
    html_output = []
    html_output.append(
        """
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
    """
    )
    html_output.append(
        f"<h1>Model:{args.model}   |   Output tokens:{args.output_tokens}   |   Time per test (s):60   |   GPU:1 x L40S </h1>"
    )
    # html_output.append("")

    # Get unique prompt token values
    prompt_tokens = sorted(df["Prompt Tokens"].unique())

    # Create plots for each prompt token value
    for token_value in prompt_tokens:
        # Filter data for current prompt token value
        token_df = df[df["Prompt Tokens"] == token_value]

        # Create figure with two subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                f"P90 Time to First Token vs. Concurrency",
                f"P90 Time per Output Token vs. Concurrency",
                "P90 Total Latency vs. Concurrency",
            ),
            vertical_spacing=0.1,
        )

        # Plot data for each provider
        for provider in ["vllm", "adaptive"]:
            provider_df = token_df[token_df["Provider"] == provider]

            # First subplot: P90 Time to First Token
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Time To First Token"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f'rgba{(135, 206, 235, 0.4) if provider == "vllm" else (255, 182, 193, 0.4)}',
                    line=dict(color="blue" if provider == "vllm" else "red"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Second subplot: P90 Latency per Token
            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Latency Per Token"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f'rgba{(135, 206, 235, 0.4) if provider == "vllm" else (255, 182, 193, 0.4)}',
                    line=dict(color="blue" if provider == "vllm" else "red"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=provider_df["Concurrency"],
                    y=provider_df["P90 Total Latency"],
                    name=f"{provider}",
                    fill="tozeroy",
                    fillcolor=f'rgba{(135, 206, 235, 0.4) if provider == "vllm" else (255, 182, 193, 0.4)}',
                    line=dict(color="blue" if provider == "vllm" else "red"),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

        # Update layout
        fig.update_layout(
            # title_text=f"Input Tokens: {int(token_value)}",
            height=1000,
            width=1000,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        # Update axes
        fig.update_xaxes(title_text="Concurrency (QPS)", row=1, col=1)
        fig.update_xaxes(title_text="Concurrency (QPS)", row=2, col=1)
        fig.update_xaxes(title_text="Concurrency (QPS)", row=3, col=1)
        fig.update_yaxes(title_text="TTFT (ms)", row=1, col=1)
        fig.update_yaxes(title_text="TPOT(ms)", row=2, col=1)
        fig.update_yaxes(title_text="Total latency(ms)", row=3, col=1)

        # Add to HTML output
        html_output.append(f"<h2>Input Tokens: {int(token_value)}</h2>")
        html_output.append('<div class="plot-container">')
        html_output.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        html_output.append("</div>")

    # Close HTML file
    html_output.append("</body></html>")

    # Write to file
    print("Writing HTML")
    with open(args.output_file, "w") as f:
        f.write("\n".join(html_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--model", type=str, required=True, help="Name of the benchmarked model")
    parser.add_argument("--output-tokens", type=int, required=True, help="Number of output tokens")
    parser.add_argument("--input-file", type=str, required=False, default="test.csv", help="Number of output tokens")
    parser.add_argument(
        "--output-file", type=str, required=False, default="results.html", help="Number of output tokens"
    )
    args = parser.parse_args()

    main(args)
