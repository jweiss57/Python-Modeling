import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

AthletesDF = pd.read_excel('2018-19 College Basketball Athletes.xlsx',
                            skiprows=(0,1), header=None,
                            names=['LastName','FirstName','College','Region',
                            'StateTheyPlayIn','PublicOrPrivate', 'Height', 'Assists', 'Blocks',
                            'Rebounds', 'Points', 'NumberofGamesPlayed'],
                            usecols='A:L')

SchoolsDF = pd.read_excel('Data for Top 50 Colleges.xlsx',
                          skiprows = (1), header = None,
                          names=['Name', 'PublicOrPrivate', 'Region', 'Ivy',
                                 'Age', 'Tuition', 'AvgClassSize', 'AvgStudentAge',
                                 'AcceptanceRate', 'AvgACTScore', 'AvgHighSchoolGPA'],
                          usecols='A:K')


fig = px.bar(SchoolsDF, x= 'Name', y= 'Tuition', title='Tuition Per School')
fig.update_layout(
    font_family="Courier New",
    font_color="black",
    title_font_family="Times New Roman",
    title_font_color="red")
fig.write_html('BarGraph.html', auto_open=True)

fig = px.scatter(SchoolsDF, x= 'AcceptanceRate', y= 'Tuition', title='Tuition by Acceptance Rate', hover_name="Name")
fig.update_layout(
    font_family="Courier New",
    font_color="black",
    title_font_family="Times New Roman",
    title_font_color="blue")
fig.write_html('ScatterPlot.html', auto_open=True)

fig = px.scatter(AthletesDF, x="Height", y="Rebounds",
                 size="Blocks", color="Region",
                 hover_name="LastName", log_x=True, size_max=60)
fig.write_html('BubbleGraph.html', auto_open=True)