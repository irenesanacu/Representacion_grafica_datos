import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

"""
Añadir botones para ir eligiendo que algoritmo se representa y diferentes datos.
"""


datos_formas = pd.read_csv("datos.txt")
datos = datos_formas.copy()
clust_algorithms=['none','DBscan']
datasets={
    'Formas': datos

}
app = dash.Dash(__name__)

# Si no existe la columna "grupo_manual", créala
if "grupo_DBscan" not in datos.columns:
    coords=datos[['x','y']]
    dbscan = DBSCAN(eps=1.1, min_samples=5)
    datos["grupo_DBscan"] =dbscan.fit_predict(coords)
if "grupo_manual" not in datos.columns:
    datos["grupo_manual"] = datos["group"]  # Inicialmente, es igual a "group"
datos['id'] = datos.index
todos_los_grupos = pd.DataFrame({"group": datos["group"].astype(str).unique()})

app.layout = html.Div([html.H3(html.H2("Visualizador de Datasets"),


                        # Selector de dataset

                        "Selecciona los datos y el metodo de clusterizacion"),
                        dcc.Dropdown(
                                id='data_dropdown',
                                options=[{'label': k, 'value': k} for k in datasets.keys()],
                                value=list(datasets.keys())[0]
                            ),
                        html.Div(id='dataset_selected'),
                        html.Label("Selecciona el algoritmo de clustering"),
                        dcc.Dropdown(
                            id='algorithm_dropdown',
                            options=[{'label': k, 'value': k} for k in clust_algorithms],
                            value='none'  # valor inicial, será actualizado dinámicamente
                        ),

                       html.Hr(),
                       html.H3("Datos de los puntos seleccionados"),
                       dcc.Graph(id="scatter-plot"),
                       html.H3("Asignar nuevo grupo a los puntos seleccionados"),
                       dcc.Input(id="nuevo-grupo", type="text", placeholder="Ingresa el nuevo grupo"),
                       html.Button("Asignar grupo", id="asignar-grupo-btn"),
                       html.Div(id="confirmacion-asignacion"),
                       html.Button("Calcular porcentaje", id="calcular_porcentaje"),
                       html.Div(id="porcentaje")
                       ])
@app.callback(
    Output('scatter-plot','figure'),
    Input('data_dropdown', 'value'),
    Input('algorithm_dropdown', 'value')

)
def print_dots(dataset_value,algorithm):
    df=datasets[dataset_value]

    if algorithm == 'none':

        figure = px.scatter(df, x="x", y="y", color="group",
                            hover_data=["group", "grupo_manual"])
    elif algorithm == 'DBscan':
        dbscan = DBSCAN(eps=1.1, min_samples=5)
        df["grupo_DBscan"] = dbscan.fit_predict(coords)
        figure = px.scatter(df, x="x", y="y", color="grupo_DBscan",
                            hover_data=["group", "grupo_manual", "grupo_DBscan"])
    return figure

@app.callback(
    Output('grafico', 'figure'),
    Input('data_dropdown', 'value'),
    Input('algorithm_dropdown', 'value')
)
def actualizar_grafico(dataset_seleccionado, columna_y):
    df = datasets[dataset_seleccionado]
    fig = px.scatter(df, x='x', y=columna_y, title=f'{columna_y} vs x en {dataset_seleccionado}')
    return fig
"""
@app.callback(
    Output('dataset_selected', 'children'),
    Output('scatter-plot','figure'),
    Input('data_dropdown', 'value'),
    Input('algorithm_dropdown', 'value')
)
def Data_selection(valor_seleccionado):
    datos = datasets[valor_seleccionado]
    return f"Seleccionaste: {valor_seleccionado}, con datos: {datos}"
    """

@app.callback(
    Output("selected-data", "figure"),
    Input("scatter-plot", "selectedData")
)
def update_graph(selectedData):
    print('update graph')
    print(selectedData)
    print('datos')
    print(datos)
    # Inicializar un DataFrame con todos los grupos y conteos en 0
    todos_los_grupos = datos["group"].astype(str).unique()
    group_initial = pd.DataFrame({"group": todos_los_grupos, "count": 0})

    # Obtener los puntos seleccionados (si los hay)
    selected_points = selectedData["points"] if selectedData and "points" in selectedData else []
    print(f"Número de puntos seleccionados: {len(selected_points)}")

    print('selected_points: ')
    print(selected_points)
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
    # group_merge = group_merge.sort_values(by="group")

    print("DataFrame final:")
    print(group_merge)
    # Crear el gráfico de barras
    return px.bar(group_merge, x="group", y="count", text_auto=True, title="Número de puntos seleccionados")

"""
@app.callback(
    Output("confirmacion-asignacion", "children"),
    Output("scatter-plot", "figure"),
    Input("asignar-grupo-btn", "n_clicks"),
    State("nuevo-grupo", "value"),
    State("scatter-plot", "selectedData"),
    prevent_initial_call=True
)
def asignar_grupo(n_clicks, nuevo_grupo, selectedData):
    print('assing group')
    print(selectedData)
    print('----------------------')
    if not nuevo_grupo:
        return "Por favor, ingresa un nuevo grupo.", dash.no_update

    if selectedData and "points" in selectedData:
        # Extraer los índices de los puntos seleccionados
        selected_indices = [point["pointIndex"] for point in selectedData["points"]]

        print('selected_indices')
        print(selected_indices)
        # Asignar el nuevo grupo a los puntos seleccionados
        datos.loc[selected_indices, "grupo_manual"] = nuevo_grupo
        print('datos')
        print(datos)

        # Actualizar el gráfico de dispersión con los nuevos grupos
        fig = px.scatter(datos, x="x", y="y", color="group", hover_data=["group","grupo_manual"])
        print('-------------')
        print(datos)
        return f"Grupo '{nuevo_grupo}' asignado correctamente.", fig
    else:
        return "No hay puntos seleccionados.", dash.no_update

    # Ejecutar la app

"""
@app.callback(
    Output("porcentaje", "children"),
    Input("calcular_porcentaje", "n_clicks"),
    prevent_initial_call=True
)
def percentage(click):
    aciertos = (datos["grupo_manual"] == datos["group"]).sum()

    porcent=aciertos/len(datos)*100

    return "El porcentaje de acierto es: ",porcent,"%"
if __name__ == "__main__":
    app.run(debug=True)

