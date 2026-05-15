
# core/analytics.py
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

class Analytics:
    
    def match_score_gauge(self, score: int):
        """Beautiful Gauge Chart like the screenshot"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Match Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#22c55e"},
                'steps': [
                    {'range': [0, 50], 'color': "#ef4444"},
                    {'range': [50, 75], 'color': "#eab308"},
                    {'range': [75, 100], 'color': "#22c55e"}
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'value': score}
            }
        ))
        fig.update_layout(height=350, margin=dict(l=30, r=30, t=50, b=30))
        return fig

    def radar_chart(self, categories, values):
        """Spider/Radar Chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Strength',
            line_color='#22c55e'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Profile Strength Radar",
            height=450,
            showlegend=False
        )
        return fig

    def skill_comparison(self, matching, missing):
        categories = ['Matching Skills', 'Missing Skills']
        values = [len(matching), len(missing)]
        colors = ['#22c55e', '#ef4444']
        
        fig = px.bar(x=categories, y=values, text=values, color=categories,
                    color_discrete_sequence=colors, title="Matching vs Missing Skills")
        fig.update_layout(height=400, showlegend=False)
        return fig
