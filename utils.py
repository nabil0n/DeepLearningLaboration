import pandas as pd
import plotly.express as px

def plot_model_stats(stats):
    if isinstance(stats, list):
        df = pd.concat([pd.read_csv(f) for f in stats])
    else:
        df = pd.read_csv(stats)
    
    return px.line(
        df,
        x='episode',
        y=['running_reward', "max_reward"],
        title='Running Reward over time',
        labels={
            'running_reward': 'Running Reward',
            'episode': 'Episode'
            },
        template='plotly_dark',
        hover_name='episode'
    )
