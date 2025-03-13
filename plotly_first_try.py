import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, Input, Output

datos = pd.read_csv("datos.txt")
    #datos["group"] = datos["group"].astype(str)

    # Obtener todos los grupos posibles antes de la selección
#todos_los_grupos = datos["group"].astype(str).unique()
todos_los_grupos = pd.DataFrame({"group": datos["group"].astype(str).unique()})
print(type(todos_los_grupos))
app = dash.Dash(__name__)

app.layout= html.Div([html.H3("Selecciona puntos con el lazo para ver las gráficas"),
                          dcc.Graph(id="scatter-plot",figure = px.scatter(datos,x="x",y="y",color="group",hover_data="group")),
                          html.Hr(),
                          html.H3("Datos de los puntos seleccionados"),
                          dcc.Graph(id="selected-data")
                          ])


@app.callback(
        Output("selected-data", "figure"),
        Input("scatter-plot", "selectedData")
)

def update_graph(selectedData):

    group_initial = pd.DataFrame({"group": todos_los_grupos["group"], "count": [0] * len(todos_los_grupos)})

        #if not selectedData:

        # Extraer los puntos seleccionados
        #selected_points = selectedData["points"]
        #selected_points = selectedData.get("points", []) #Da error por el caso en el que todavía no se han seleccionado datos
    selected_points = selectedData["points"] if selectedData and "points" in selectedData else []
    print(len(selected_points))
    if selected_points:
        selected_indices = [point["pointIndex"] for point in selected_points]
        selected_df = datos.iloc[selected_indices].copy()
        selected_df["group"] = selected_df["group"].astype(str)
        group_counts = selected_df["group"].value_counts().reset_index()
        group_counts.columns = ["group", "count"]
        group_merge_left=group_initial.merge(group_counts,on="group",how="left", suffixes=("_initial", "_selected"))
        print(group_merge_left)

        group_merge_left["count_initial"] += group_merge_left["count_selected"].fillna(0)
        group_merge_left["count_initial"]=group_merge_left["count_initial"].astype(int)
        print(group_merge_left)

        group_merge_left.drop(columns=["count_selected"],inplace=True)
        print(group_merge_left)

        return px.bar(group_merge_left, x="group", y="count_initial", text_auto=True, title="Número de puntos seleccionados")

    # Ejecutar la app
if __name__ == "__main__":
        app.run_server(debug=True)
"""
fig.update_traces(marker=dict(size=10))
fig.show()
"""