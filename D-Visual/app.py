import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Set the renderer for Plotly to open in the browser
pio.renderers.default = 'browser'

# Step 1: Generate Random Student Data
np.random.seed(42)

# Generating student IDs
student_ids = [f"S{i+1}" for i in range(100)]
names = [f"Student {i+1}" for i in range(100)]
ages = np.random.randint(18, 25, size=100)
genders = np.random.choice(['Male', 'Female'], size=100)
math_scores = np.random.randint(60, 100, size=100)
science_scores = np.random.randint(60, 100, size=100)
english_scores = np.random.randint(60, 100, size=100)

# Creating DataFrame
data = {
    'Student ID': student_ids,
    'Name': names,
    'Age': ages,
    'Gender': genders,
    'Math Score': math_scores,
    'Science Score': science_scores,
    'English Score': english_scores
}

df = pd.DataFrame(data)

# Step 2: Create Graphs

# 1. Histogram of Ages
fig1 = px.histogram(df, x='Age', title='Distribution of Ages')
fig1.show()

# 2. Pie Chart of Gender Distribution
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']
fig2 = px.pie(gender_counts, values='Count', names='Gender', title='Gender Distribution')
fig2.show()

# 3. Scatter Plot of Math vs Science Scores
fig3 = px.scatter(df, x='Math Score', y='Science Score', color='Gender',
                  title='Math vs Science Scores')
fig3.show()

# 4. Box Plot of Scores
fig4 = go.Figure()
fig4.add_trace(go.Box(y=df['Math Score'], name='Math Score'))
fig4.add_trace(go.Box(y=df['Science Score'], name='Science Score'))
fig4.add_trace(go.Box(y=df['English Score'], name='English Score'))
fig4.update_layout(title='Box Plot of Scores by Subject')
fig4.show()

# 5. Bar Chart of Average Scores by Gender
average_scores = df.groupby('Gender').mean().reset_index()
fig5 = px.bar(average_scores, x='Gender', y=['Math Score', 'Science Score', 'English Score'],
              title='Average Scores by Gender', barmode='group')
fig5.show()
