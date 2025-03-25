
import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State

datos_original = pd.read_csv("datos.txt")
datos=datos_original.copy()
# Si no existe la columna "grupo_manual", créala
if "grupo_manual" not in datos.columns:
    datos["grupo_manual"] = datos["group"]  # Inicialmente, es igual a "group"
datos['id']=datos.index
todos_los_grupos = pd.DataFrame({"group": datos["group"].astype(str).unique()})
print(type(todos_los_grupos))
app = dash.Dash(__name__)

app.layout= html.Div([html.H3("Selecciona puntos con el lazo para ver las gráficas"),
                          dcc.Graph(id="scatter-plot",figure = px.scatter(datos,x="x",y="y",color="group",hover_data="group")),
                          html.Hr(),
                          html.H3("Datos de los puntos seleccionados"),
                          dcc.Graph(id="selected-data"),
                          html.H3("Asignar nuevo grupo a los puntos seleccionados"),
                          dcc.Input(id="nuevo-grupo", type="text", placeholder="Ingresa el nuevo grupo"),
                          html.Button("Asignar grupo", id="asignar-grupo-btn"),
                          html.Div(id="confirmacion-asignacion")
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

    # Convertir la columna "group" a numérica para ordenar
    group_merge["group"] = pd.to_numeric(group_merge["group"])

    # Ordenar el DataFrame por la columna "group"
    #group_merge = group_merge.sort_values(by="group")

    print("DataFrame final:")
    print(group_merge)

    # Crear el gráfico de barras
    return px.bar(group_merge, x="group", y="count", text_auto=True, title="Número de puntos seleccionados")

@app.callback(
    Output("confirmacion-asignacion", "children"),
    Output("scatter-plot", "figure"),
    Input("asignar-grupo-btn", "n_clicks"),
    State("nuevo-grupo", "value"),
    State("scatter-plot", "selectedData"),
    prevent_initial_call=True
    )
def asignar_grupo(n_clicks, nuevo_grupo, selectedData):
    if not nuevo_grupo:
        return "Por favor, ingresa un nuevo grupo.", dash.no_update

    if selectedData and "points" in selectedData:
        # Extraer los índices de los puntos seleccionados
        selected_indices = [point["pointIndex"] for point in selectedData["points"]]

        # Asignar el nuevo grupo a los puntos seleccionados
        datos.loc[selected_indices, "grupo_manual"] = nuevo_grupo

        # Actualizar el gráfico de dispersión con los nuevos grupos
        fig = px.scatter(datos, x="x", y="y", color="grupo_manual", hover_data=["group"])
        return f"Grupo '{nuevo_grupo}' asignado correctamente.", fig
    else:
        return "No hay puntos seleccionados.", dash.no_update

    # Ejecutar la app


if __name__ == "__main__":
        app.run_server(debug=True)
