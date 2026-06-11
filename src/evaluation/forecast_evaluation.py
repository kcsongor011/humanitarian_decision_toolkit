"""Small forecast evaluation and display helpers for reader-facing notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def mean_absolute_error(actual, forecast) -> float:
    """Return mean absolute error after aligning inputs by position."""

    errors = pd.Series(actual).reset_index(drop=True) - pd.Series(
        forecast
    ).reset_index(drop=True)
    return float(errors.abs().mean())


def root_mean_squared_error(actual, forecast) -> float:
    """Return root mean squared error after aligning inputs by position."""

    errors = pd.Series(actual).reset_index(drop=True) - pd.Series(
        forecast
    ).reset_index(drop=True)
    return float((errors.pow(2).mean()) ** 0.5)


def make_forecast_frame(
    results,
    test_data: pd.DataFrame,
    forecast_column: str,
    lower_column: str,
    upper_column: str,
    alpha: float = 0.20,
) -> pd.DataFrame:
    forecast_result = results.get_forecast(steps=len(test_data))
    forecast_mean = forecast_result.predicted_mean
    forecast_interval = forecast_result.conf_int(alpha=alpha)

    return pd.DataFrame(
        {
            "date": test_data["date"].to_numpy(),
            "actual_applications": test_data["applications"].to_numpy(),
            forecast_column: forecast_mean.to_numpy(),
            lower_column: forecast_interval.iloc[:, 0].to_numpy(),
            upper_column: forecast_interval.iloc[:, 1].to_numpy(),
        }
    )


def plot_forecast_with_interval(
    forecast_data: pd.DataFrame,
    forecast_column: str,
    lower_column: str,
    upper_column: str,
    title: str,
    forecast_label: str,
    grid_alpha: float = 0.25,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        forecast_data["date"],
        forecast_data["actual_applications"],
        label="Actual test observations",
        linewidth=1.8,
    )
    ax.plot(
        forecast_data["date"],
        forecast_data[forecast_column],
        label=forecast_label,
        linewidth=1.5,
    )
    ax.fill_between(
        forecast_data["date"],
        forecast_data[lower_column],
        forecast_data[upper_column],
        alpha=0.2,
        label="80% confidence interval",
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Applications")
    ax.grid(alpha=grid_alpha)
    ax.legend()
    plt.tight_layout()
    plt.show()


def evaluate_forecast(
    actual,
    forecast,
    label: str,
    label_column: str = "Baseline / model",
) -> dict[str, float | str]:
    return {
        label_column: label,
        "MAE": mean_absolute_error(actual, forecast),
        "RMSE": root_mean_squared_error(actual, forecast),
    }


def display_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.assign(
        MAE=lambda df: df["MAE"].round(1),
        RMSE=lambda df: df["RMSE"].round(1),
    )


def make_model_info_table(
    model_name: str,
    order,
    seasonal_order,
    results,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": model_name,
                "Non-seasonal order": str(order),
                "Seasonal order": str(seasonal_order),
                "Training period": (
                    f"{train_data['date'].min():%Y-%m} "
                    f"to {train_data['date'].max():%Y-%m}"
                ),
                "Test period": (
                    f"{test_data['date'].min():%Y-%m} "
                    f"to {test_data['date'].max():%Y-%m}"
                ),
                "AIC": results.aic,
                "BIC": results.bic,
            }
        ]
    ).assign(
        AIC=lambda df: df["AIC"].round(1),
        BIC=lambda df: df["BIC"].round(1),
    )
