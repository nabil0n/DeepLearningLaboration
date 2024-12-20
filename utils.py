import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_running_max(stats, title):
    if isinstance(stats, list):
        df = pd.concat([pd.read_csv(f) for f in stats])
    else:
        df = pd.read_csv(stats)
    
    return px.line(
        df,
        x='episode',
        y=['running_reward', "max_reward"],
        title=title,
        labels={
            'running_reward': 'Running Reward',
            'episode': 'Episode'
            },
        template='plotly_dark',
        hover_name='episode',
    )

def plot_all(stats, title, visibility='legendonly'):
    if isinstance(stats, list):
        df = pd.concat([pd.read_csv(f) for f in stats])
    else:
        df = pd.read_csv(stats)
    
    return px.scatter(
        df,
        x='time' if 'time' in df.columns else 'time_since_start',
        y=[i for i in df.columns if i not in ['episode', 'time', 'time_since_start']],
        title=title,
        labels={
            'running_reward': 'Running Reward',
            'episode': 'Episode',
            'time_since_start': 'Time Since Start',
            },
        template='plotly_dark',
        hover_name='episode',
        trendline='lowess',
    )
