from typing import Any
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from datetime import timedelta


def main() -> None:
    # build page
    st.set_page_config(layout="wide")
    st.subheader("RAM and VRAM usage")
    c1, c2, c3 = st.columns(3)
    c1.button("Reset")
    smoothing = c2.slider("Smoothing", min_value=1, max_value=50, value=1)
    file = c3.text_input("File", value="ram_log.txt")

    data = read_data(file)
    fig = create_plot(data, smoothing)

    st.plotly_chart(fig, use_container_width=True)

    meta = analyze(data)
    st.subheader("Analysis")
    st.table(meta)

    st.subheader("Predictions")
    c1, c2 = st.columns(2)
    ram_limit = c1.number_input("RAM limit (GB)", value=64, step=1, min_value=1)
    vram_limit = c2.number_input("VRAM limit (GB)", value=8, step=1, min_value=1)
    prediction_table, out_of_ram, out_of_vram = predict_future(data, ram_limit, vram_limit)
    st.text(f"Out of RAM in {out_of_ram}")
    st.text(f"Out of VRAM in {out_of_vram}")
    st.table(prediction_table)


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def predict_future(
    data: dict[str, Any], ram_limit: float, vram_limit: float
) -> tuple[pd.DataFrame, timedelta, timedelta]:
    # predict linearly y=ax+b
    to_predict = {
        "1h": 3600,
        "10h": 36000,
        "24h": 24 * 3600,
        "48h": 48 * 3600,
        "1week": 7 * 24 * 3600,
    }
    predictions: dict[str, dict] = {key: {} for key in to_predict.keys()}
    x = data["Time (sec)"]
    out_of_ram = None
    out_of_vram = None
    for key, y in data.items():
        if key == "Time (sec)":
            continue

        a, b = np.polyfit(x, y, 1)

        # create some predictions
        for predict_key, t in to_predict.items():
            predictions[predict_key][key] = a * t + b

        if key == "Used RAM":
            # y = a*x + b
            out_of_ram = timedelta(seconds=(ram_limit - b) / a)
        if key == "Used VRAM":
            out_of_vram = timedelta((vram_limit - b) / a)
    assert out_of_ram is not None
    assert out_of_vram is not None

    return pd.DataFrame(predictions), out_of_ram, out_of_vram


def read_data(file_name: str) -> dict[str, np.ndarray]:
    file = Path(file_name)
    with file.open("r") as F:
        contents = F.readlines()

    keys = contents[0].split(";")
    values = [line.split(";") for line in contents[1:]]
    data = {
        key: np.array([float(value) for value in values])
        for key, values in zip(keys, zip(*values))
    }
    return data


def create_plot(data: dict[str, np.ndarray], smoothing: int) -> go.Figure:
    fig = go.Figure()
    x = data["Time (sec)"]
    for key, y in data.items():
        if key == "Time (sec)":
            continue

        i_max = np.argmax(y)

        y_smooth = moving_average(y, n=smoothing)

        fig.add_scatter(
            x=x[smoothing - 1 :],
            y=y_smooth,
            name=key,
            showlegend=True,
            mode="lines",
        )
        fig.add_annotation(x=x[i_max], y=y[i_max], text=f"{y[i_max]:.2f}")

    fig.update_layout(xaxis_rangeslider_visible=True)
    return fig


def analyze(data: dict[str, np.ndarray]) -> pd.DataFrame:
    data = {k: v for k, v in data.items() if k != "Time (sec)"}
    maxs = {k: np.max(v) for k, v in data.items()}
    mins = {k: np.min(v) for k, v in data.items()}
    avg = {k: np.mean(v) for k, v in data.items()}
    meta = pd.DataFrame({"max": maxs, "min": mins, "average": avg})
    return meta


# st.line_chart(data, x="Time (sec)")

if __name__ == "__main__":
    main()
