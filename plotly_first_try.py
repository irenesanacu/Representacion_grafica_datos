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
    group_counts = todos_los_grupos.copy()
        #group_counts["count"] = 0
    group_counts = pd.DataFrame({"group": todos_los_grupos["group"], "count": [0] * len(todos_los_grupos)})

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
        #selected_df = pd.DataFrame(selected_points)
        counts = selected_df["group"].value_counts().reset_index()
        counts.columns = ["group", "count"]
        # Unir con el DataFrame base para asegurarnos de que todos los grupos aparecen
        #group_counts = group_counts.merge(counts, on="group", how="left").fillna(0)
        # Graficar los datos seleccionados
        counts["group"] = counts["group"].astype(str)
        counts["count"] = counts["count"].astype(int)
        print(counts)
        return px.bar(counts, x="group", y="count",title="Puntos seleccionados")
    # Ejecutar la app
if __name__ == "__main__":
        app.run_server(debug=True)
"""
fig.update_traces(marker=dict(size=10))
fig.show()
"""