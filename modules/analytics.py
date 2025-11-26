# analytics.py
import pandas as pd
import plotly.express as px

def treemap(df, col_job, col_salary):
    fig = px.treemap(
        df,
        path=["Education_Level", col_job],
        values=col_salary,
        title="Treemap: Education → Jobs"
    )
    return fig

def bar_ai_exposure(df, col_job, col_ai, n):
    df2 = df.sort_values(col_ai, ascending=False).head(n)
    fig = px.bar(df2, x=col_job, y=col_ai, title=f"Top {n} AI Exposure")
    return fig

def bar_salary(df, col_job, col_salary, n):
    df2 = df.sort_values(col_salary, ascending=True).tail(n)
    fig = px.bar(df2, x=col_salary, y=col_job, orientation="h",
                 title=f"Top {n} Salaries")
    return fig

def skills_bar(df, skill_cols):
    avg_sk = df[skill_cols].mean().reset_index()
    avg_sk.columns = ["Skill", "Value"]
    fig = px.bar(avg_sk, x="Skill", y="Value", title="Average Skill Values")
    return fig
