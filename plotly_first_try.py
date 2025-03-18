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

    # Inicializar un DataFrame con todos los grupos y conteos en 0
    todos_los_grupos = datos["group"].astype(str).unique()
    group_initial = pd.DataFrame({"group": todos_los_grupos, "count": 0})

    # Obtener los puntos seleccionados (si los hay)
    selected_points = selectedData["points"] if selectedData and "points" in selectedData else []
    print(f"Número de puntos seleccionados: {len(selected_points)}")

    if selected_points:
        # Extraer los índices de los puntos seleccionados
        selected_indices = [point["pointIndex"] for point in selected_points]
        selected_df = datos.iloc[selected_indices].copy()
        selected_df["group"] = selected_df["group"].astype(str)

        # Contar los puntos seleccionados por grupo
        group_counts = selected_df["group"].value_counts().reset_index()
        group_counts.columns = ["group", "count"]
        print("Conteo de grupos seleccionados:")
        print(group_counts)

        # Combinar con el DataFrame inicial para incluir todos los grupos
        group_merge = group_initial.merge(group_counts, on="group", how="left", suffixes=("_initial", "_selected"))
        group_merge["count"] = group_merge["count_selected"].fillna(0).astype(int)
        group_merge.drop(columns=["count_selected"], inplace=True)
    else:
        # Si no hay puntos seleccionados, usar el DataFrame inicial
        group_merge = group_initial

    print("DataFrame final:")
    print(group_merge)

    # Crear el gráfico de barras
    return px.bar(group_merge, x="group", y="count", text_auto=True, title="Número de puntos seleccionados")
    # Ejecutar la app
if __name__ == "__main__":
        app.run_server(debug=True)
"""
fig.update_traces(marker=dict(size=10))
fig.show()
"""