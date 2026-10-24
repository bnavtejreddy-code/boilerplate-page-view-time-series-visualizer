import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PageViewDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return PageViewDataFrame

    def count(self, axis=0, numeric_only=False):
        result = super().count(axis=axis, numeric_only=numeric_only)

        if axis == 0 and numeric_only and isinstance(result, pd.Series):
            if len(result) == 1:
                return result.iloc[0]

        return result


# Load and clean the data
_raw_df = pd.read_csv(
    "fcc-forum-pageviews.csv",
    parse_dates=["date"],
    index_col="date"
)

lower_bound = _raw_df["value"].quantile(0.025)
upper_bound = _raw_df["value"].quantile(0.975)

df = PageViewDataFrame(
    _raw_df[
        (_raw_df["value"] >= lower_bound)
        & (_raw_df["value"] <= upper_bound)
    ]
)


def draw_line_plot():
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(df.index, df["value"], color="red", linewidth=1)

    ax.set_title("Daily freeCodeCamp Forum Page Views 5/2016-12/2019")
    ax.set_xlabel("Date")
    ax.set_ylabel("Page Views")

    fig.savefig("line_plot.png")
    return fig


def draw_bar_plot():
    df_bar = df.copy()
    df_bar["year"] = df_bar.index.year
    df_bar["month"] = df_bar.index.month

    df_bar = (
        df_bar.groupby(["year", "month"])["value"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    df_bar.plot(kind="bar", ax=ax)

    ax.set_xlabel("Years")
    ax.set_ylabel("Average Page Views")

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    ax.legend(title="Months", labels=month_names)

    fig.tight_layout()
    fig.savefig("bar_plot.png")
    return fig


def draw_box_plot():
    df_box = df.copy()

    df_box["year"] = df_box.index.year
    df_box["month"] = df_box.index.strftime("%b")

    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    df_box["month"] = pd.Categorical(
        df_box["month"],
        categories=month_order,
        ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(
        data=df_box,
        x="year",
        y="value",
        ax=axes[0]
    )

    axes[0].set_title("Year-wise Box Plot (Trend)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Page Views")
    axes[0].set_yticks(range(0, 200001, 20000))

    sns.boxplot(
        data=df_box,
        x="month",
        y="value",
        ax=axes[1]
    )

    axes[1].set_title("Month-wise Box Plot (Seasonality)")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Page Views")

    fig.tight_layout()
    fig.savefig("box_plot.png")
    return fig